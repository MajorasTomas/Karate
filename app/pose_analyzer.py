"""
Karate pose analysis using YOLOv11 pose estimation
"""

import os
import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Tuple, Dict
import logging
from collections import deque

logger = logging.getLogger(__name__)


class KaratePoseAnalyzer:
    """Analyzes karate poses using YOLOv11 pose estimation"""
    
    # COCO keypoint indices for body parts
    KEYPOINT_MAPPING = {
        'nose': 0,
        'left_eye': 1,
        'right_eye': 2,
        'left_ear': 3,
        'right_ear': 4,
        'left_shoulder': 5,
        'right_shoulder': 6,
        'left_elbow': 7,
        'right_elbow': 8,
        'left_wrist': 9,
        'right_wrist': 10,
        'left_hip': 11,
        'right_hip': 12,
        'left_knee': 13,
        'right_knee': 14,
        'left_ankle': 15,
        'right_ankle': 16
    }
    
    # Body parts to track (mapping to our requirements)
    TRACKED_PARTS = {
        'head': ['nose'],
        'chest': ['left_shoulder', 'right_shoulder'],
        'elbows': ['left_elbow', 'right_elbow'],
        'hands': ['left_wrist', 'right_wrist'],
        'hips': ['left_hip', 'right_hip'],
        'knees': ['left_knee', 'right_knee'],
        'toes': ['left_ankle', 'right_ankle']
    }
    
    def __init__(self, reference_video_path: str):
        """
        Initialize pose analyzer
        
        Args:
            reference_video_path: Path to professional karate video
        """
        logger.info("Initializing KaratePoseAnalyzer...")
        
        # Load YOLOv11 pose model
        self.model = YOLO('yolo11n-pose.pt')  # Using nano model for speed
        
        # Extract reference poses
        self.reference_video_path = reference_video_path
        self.reference_poses = self._extract_poses(reference_video_path)
        
        logger.info(
            f"Loaded {len(self.reference_poses)} reference poses"
        )
    
    def _extract_poses(self, video_path: str) -> List[np.ndarray]:
        """
        Extract pose keypoints from video
        
        Args:
            video_path: Path to video file
            
        Returns:
            List of pose keypoints for each frame
        """
        poses = []
        cap = cv2.VideoCapture(video_path)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run pose estimation
            results = self.model(frame, verbose=False)
            
            if len(results) > 0 and results[0].keypoints is not None:
                keypoints = results[0].keypoints.xy.cpu().numpy()
                if len(keypoints) > 0:
                    poses.append(keypoints[0])  # Take first person
        
        cap.release()
        return poses
    
    def _calculate_similarity(
        self, 
        user_keypoints: np.ndarray, 
        ref_keypoints: np.ndarray,
        part_indices: List[int]
    ) -> float:
        """
        Calculate similarity score for specific body parts
        
        Args:
            user_keypoints: User's keypoints
            ref_keypoints: Reference keypoints
            part_indices: Indices of keypoints to compare
            
        Returns:
            Similarity score (0-1, higher is better)
        """
        if len(user_keypoints) == 0 or len(ref_keypoints) == 0:
            return 0.0
        
        # Calculate normalized distance
        distances = []
        for idx in part_indices:
            if idx < len(user_keypoints) and idx < len(ref_keypoints):
                user_pt = user_keypoints[idx]
                ref_pt = ref_keypoints[idx]
                
                # Euclidean distance
                dist = np.linalg.norm(user_pt - ref_pt)
                distances.append(dist)
        
        if not distances:
            return 0.0
        
        # Normalize by image diagonal (assuming 640x480 default)
        img_diagonal = np.sqrt(640**2 + 480**2)
        avg_distance = np.mean(distances)
        
        # Convert to similarity score (0-1)
        similarity = max(0, 1 - (avg_distance / (img_diagonal * 0.2)))
        return similarity
    
    def _get_matching_reference_frame(self, frame_idx: int, total_frames: int) -> int:
        """
        Get matching reference frame index
        
        Args:
            frame_idx: Current frame index
            total_frames: Total frames in user video
            
        Returns:
            Matching reference frame index
        """
        if len(self.reference_poses) == 0:
            return 0
        
        # Map user frame to reference frame proportionally
        ref_idx = int(
            (frame_idx / total_frames) * len(self.reference_poses)
        )
        return min(ref_idx, len(self.reference_poses) - 1)
    
    def analyze_video(self, video_path: str) -> str:
        """
        Analyze user video and create annotated output
        
        Args:
            video_path: Path to user video
            
        Returns:
            Path to annotated video
        """
        logger.info(f"Analyzing video: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Output video path
        output_path = os.path.join('temp', 'analyzed_output.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Moving average for performance
        performance_scores = deque(maxlen=30)  # Last 30 frames
        frame_idx = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run pose estimation on user video
            results = self.model(frame, verbose=False)
            
            if len(results) > 0 and results[0].keypoints is not None:
                keypoints = results[0].keypoints.xy.cpu().numpy()
                
                if len(keypoints) > 0:
                    user_keypoints = keypoints[0]
                    
                    # Get matching reference pose
                    ref_idx = self._get_matching_reference_frame(
                        frame_idx, total_frames
                    )
                    ref_keypoints = self.reference_poses[ref_idx]
                    
                    # Analyze each body part
                    frame_scores = []
                    
                    for part_name, keypoint_names in self.TRACKED_PARTS.items():
                        # Get indices for this body part
                        indices = [
                            self.KEYPOINT_MAPPING[kp] 
                            for kp in keypoint_names
                        ]
                        
                        # Calculate similarity
                        similarity = self._calculate_similarity(
                            user_keypoints, ref_keypoints, indices
                        )
                        frame_scores.append(similarity)
                        
                        # Draw boxes around keypoints
                        color = (0, 255, 0) if similarity > 0.7 else (0, 0, 255)
                        
                        for idx in indices:
                            if idx < len(user_keypoints):
                                x, y = user_keypoints[idx]
                                if x > 0 and y > 0:
                                    cv2.rectangle(
                                        frame,
                                        (int(x) - 15, int(y) - 15),
                                        (int(x) + 15, int(y) + 15),
                                        color,
                                        2
                                    )
                                    cv2.putText(
                                        frame,
                                        part_name,
                                        (int(x) - 15, int(y) - 20),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.4,
                                        color,
                                        1
                                    )
                    
                    # Update moving average
                    if frame_scores:
                        performance_scores.append(np.mean(frame_scores))
            
            # Draw performance score
            if performance_scores:
                avg_score = np.mean(performance_scores) * 100
                cv2.rectangle(frame, (10, 10), (250, 60), (0, 0, 0), -1)
                cv2.putText(
                    frame,
                    f"Performance: {avg_score:.1f}%",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )
            
            out.write(frame)
            frame_idx += 1
        
        cap.release()
        out.release()
        
        logger.info(f"Analysis complete. Output saved to: {output_path}")
        return output_path
