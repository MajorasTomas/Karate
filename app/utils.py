"""
Utility functions
"""

import os
import uuid
import logging
import requests
from typing import List, Optional, Tuple
from google.cloud import storage
from datetime import timedelta

logger = logging.getLogger(__name__)


def download_from_gcs(url: str) -> str:
    """
    Download video from any URL (Google Cloud Storage or other)
    
    Args:
        url: Video URL (GCS signed URL, public URL, etc.)
        
    Returns:
        Path to downloaded file
    """
    try:
        # Generate unique filename
        file_id = str(uuid.uuid4())
        temp_path = os.path.join('temp', f'{file_id}.mp4')
        
        # Ensure temp directory exists
        os.makedirs('temp', exist_ok=True)
        
        logger.info(f"Downloading from: {url[:100]}...")
        
        # Download file with streaming
        response = requests.get(url, stream=True, timeout=300)
        response.raise_for_status()
        
        # Save to disk in chunks
        with open(temp_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        
        file_size = os.path.getsize(temp_path)
        logger.info(f"Downloaded {file_size} bytes to: {temp_path}")
        
        return temp_path
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to download video: {str(e)}")
        raise Exception(f"Download failed: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error during download: {str(e)}")
        raise


def upload_to_gcs(
    file_path: str,
    filename: str,
    user_id: str
) -> Tuple[str, str]:
    """
    Upload file to Google Cloud Storage
    
    Args:
        file_path: Path to local file to upload
        filename: Desired filename in GCS (e.g., "AnalyzedVideo.mp4")
        user_id: User identifier for organizing files
        
    Returns:
        Tuple of (video_url, gcs_path)
        - video_url: URL to access the uploaded video
        - gcs_path: Full path in the GCS bucket
    """
    try:
        # Get configuration from environment
        bucket_name = os.getenv("GCS_OUTPUT_BUCKET")
        
        if not bucket_name:
            raise ValueError(
                "GCS_OUTPUT_BUCKET environment variable not set. "
                "Please set it in Railway dashboard."
            )
        
        logger.info(f"Uploading to bucket: {bucket_name}")
        
        # Initialize GCS client
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        
        # Create blob path (organized by user)
        # Example: analyzed/john_doe/AnalyzedVideo.mp4
        blob_path = f"analyzed/{user_id}/{filename}"
        blob = bucket.blob(blob_path)
        
        logger.info(f"Uploading to GCS path: {blob_path}")
        
        # Upload file
        blob.upload_from_filename(
            file_path,
            content_type='video/mp4'
        )
        
        # Set custom metadata
        blob.metadata = {
            'user_id': user_id,
            'original_filename': filename,
            'processed_by': 'karate-pose-analyzer',
            'content_type': 'video/mp4'
        }
        blob.patch()
        
        logger.info("✅ File uploaded to GCS")
        
        # Determine whether to use signed URL or public URL
        use_signed_url = os.getenv("GCS_USE_SIGNED_URL", "true").lower() == "true"
        
        if use_signed_url:
            # Generate signed URL (private, expires in 24 hours)
            logger.info("Generating signed URL (expires in 24 hours)...")
            video_url = blob.generate_signed_url(
                version="v4",
                expiration=timedelta(hours=24),
                method="GET"
            )
            logger.info("✅ Generated signed URL")
        else:
            # Make blob public and return public URL
            logger.info("Making blob public...")
            blob.make_public()
            video_url = blob.public_url
            logger.info("✅ Blob is now public")
        
        return video_url, blob_path
        
    except Exception as e:
        logger.error(f"Failed to upload to GCS: {str(e)}", exc_info=True)
        raise Exception(f"GCS upload failed: {str(e)}")


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
                logger.info(f"🗑️  Cleaned up: {file_path}")
            except Exception as e:
                logger.warning(f"⚠️  Failed to cleanup {file_path}: {str(e)}")
