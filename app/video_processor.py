"""
Video processing utilities
"""

import cv2
import logging
from typing import Tuple

logger = logging.getLogger(__name__)


class VideoProcessor:
    """Handles video processing operations"""
    
    @staticmethod
    def get_video_info(video_path: str) -> dict:
        """
        Get video information
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary with video metadata
        """
        cap = cv2.VideoCapture(video_path)
        
        info = {
            'fps': int(cap.get(cv2.CAP_PROP_FPS)),
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'duration': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / int(cap.get(cv2.CAP_PROP_FPS))
        }
        
        cap.release()
        return info
    
    @staticmethod
    def resize_video(
        input_path: str, 
        output_path: str, 
        target_size: Tuple[int, int]
    ) -> str:
        """
        Resize video to target size
        
        Args:
            input_path: Input video path
            output_path: Output video path
            target_size: (width, height) tuple
            
        Returns:
            Path to resized video
        """
        cap = cv2.VideoCapture(input_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, target_size)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            resized_frame = cv2.resize(frame, target_size)
            out.write(resized_frame)
        
        cap.release()
        out.release()
        
        return output_path
