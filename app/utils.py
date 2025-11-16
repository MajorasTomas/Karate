"""
Utility functions
"""

import os
import uuid
import logging
import requests
from typing import List, Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


def download_from_gcs(url: str) -> str:
    """
    Download video from Google Cloud Storage
    
    Args:
        url: GCS URL or signed URL
        
    Returns:
        Path to downloaded file
    """
    try:
        # Generate unique filename
        file_id = str(uuid.uuid4())
        temp_path = os.path.join('temp', f'{file_id}.mp4')
        
        # Ensure temp directory exists
        os.makedirs('temp', exist_ok=True)
        
        logger.info(f"Downloading from: {url}")
        
        # Download file
        response = requests.get(url, stream=True, timeout=300)
        response.raise_for_status()
        
        # Save to disk
        with open(temp_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        logger.info(f"Downloaded to: {temp_path}")
        return temp_path
        
    except Exception as e:
        logger.error(f"Failed to download video: {str(e)}")
        raise


def cleanup_temp_files(file_paths: List[Optional[str]]) -> None:
    """
    Clean up temporary files
    
    Args:
        file_paths: List of file paths to delete
    """
    for file_path in file_paths:
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
                logger.info(f"Cleaned up: {file_path}")
            except Exception as e:
                logger.warning(f"Failed to cleanup {file_path}: {str(e)}")


def validate_video_file(file_path: str) -> bool:
    """
    Validate if file is a valid video
    
    Args:
        file_path: Path to video file
        
    Returns:
        True if valid video, False otherwise
    """
    import cv2
    
    cap = cv2.VideoCapture(file_path)
    is_valid = cap.isOpened()
    cap.release()
    
    return is_valid
