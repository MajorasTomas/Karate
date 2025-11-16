"""
FastAPI application for karate pose analysis
"""

import os
import json
import logging
from fastapi import FastAPI, HTTPException, BackgroundTasks, status
from fastapi.responses import Response
from pydantic import BaseModel, HttpUrl
from typing import Optional
import uvicorn

from app.pose_analyzer import KaratePoseAnalyzer
from app.utils import download_from_gcs, cleanup_temp_files, upload_to_gcs

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_gcs_credentials():
    """Write GCS credentials from environment variable to file"""
    creds_json = os.getenv("GCS_CREDENTIALS_JSON")
    if creds_json:
        try:
            creds_path = "/tmp/gcs-credentials.json"
            
            # Parse JSON if it's a string
            if isinstance(creds_json, str):
                creds_data = json.loads(creds_json)
            else:
                creds_data = creds_json
            
            # Write to file
            with open(creds_path, 'w') as f:
                json.dump(creds_data, f)
            
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds_path
            logger.info(f"✅ GCS credentials file created at {creds_path}")
        except Exception as e:
            logger.error(f"❌ Failed to setup GCS credentials: {str(e)}")
    else:
        logger.warning("⚠️  GCS_CREDENTIALS_JSON not found in environment variables")


# Setup credentials before starting app
setup_gcs_credentials()

# Initialize FastAPI app
app = FastAPI(
    title="Karate Pose Analyzer API",
    description="AI-powered karate pose analysis using YOLOv11",
    version="1.0.0"
)

# Initialize analyzer (lazy loading)
analyzer: Optional[KaratePoseAnalyzer] = None


def get_analyzer() -> KaratePoseAnalyzer:
    """Get or create analyzer instance"""
    global analyzer
    if analyzer is None:
        reference_video_path = os.getenv(
            "REFERENCE_VIDEO_PATH", 
            "reference_videos/Nika3.mp4"
        )
        analyzer = KaratePoseAnalyzer(reference_video_path)
    return analyzer


class AnalyzeRequest(BaseModel):
    """Request model for video analysis"""
    video_url: HttpUrl
    user_id: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str


@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint"""
    return HealthResponse(status="healthy", version="1.0.0")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(status="healthy", version="1.0.0")


@app.post("/analyze", status_code=status.HTTP_204_NO_CONTENT)
async def analyze_video(
    request: AnalyzeRequest,
    background_tasks: BackgroundTasks
):
    """
    Analyze karate pose from user video and upload to Google Cloud Storage
    
    This endpoint does not return any data. It processes the video and
    uploads the analyzed result to Google Cloud Storage.
    
    Args:
        request: Contains video_url and optional user_id
        
    Returns:
        204 No Content (empty response)
    """
    temp_input_path = None
    temp_output_path = None
    
    try:
        user_id = request.user_id or "anonymous"
        logger.info(f"📥 Received analysis request for user: {user_id}")
        logger.info(f"   Input video URL: {request.video_url}")
        
        # Step 1: Download video
        logger.info("⬇️  Downloading video...")
        temp_input_path = download_from_gcs(str(request.video_url))
        
        if not os.path.exists(temp_input_path):
            raise HTTPException(status_code=400, detail="Failed to download video")
        
        logger.info(f"✅ Downloaded to: {temp_input_path}")
        
        # Step 2: Get analyzer instance
        pose_analyzer = get_analyzer()
        
        # Step 3: Analyze video
        logger.info("🤖 Analyzing video with YOLO pose detection...")
        temp_output_path = pose_analyzer.analyze_video(temp_input_path)
        
        if not os.path.exists(temp_output_path):
            raise HTTPException(status_code=500, detail="Video processing failed")
        
        logger.info(f"✅ Analysis complete: {temp_output_path}")
        
        # Step 4: Upload to Google Cloud Storage
        filename = "AnalyzedVideo.mp4"  # Hardcoded name
        logger.info(f"⬆️  Uploading to Google Cloud Storage as '{filename}'...")
        
        video_url, gcs_path = upload_to_gcs(
            file_path=temp_output_path,
            filename=filename,
            user_id=user_id
        )
        
        logger.info(f"✅ Uploaded to GCS: {gcs_path}")
        logger.info(f"🔗 Video URL: {video_url}")
        
        # Step 5: Schedule cleanup in background
        background_tasks.add_task(
            cleanup_temp_files, 
            [temp_input_path, temp_output_path]
        )
        
        logger.info("🎉 Process completed successfully!")
        
        # Return nothing (204 No Content)
        return Response(status_code=status.HTTP_204_NO_CONTENT)
        
    except HTTPException:
        # Re-raise HTTP exceptions
        cleanup_temp_files([temp_input_path, temp_output_path])
        raise
        
    except Exception as e:
        logger.error(f"❌ Error during video analysis: {str(e)}", exc_info=True)
        cleanup_temp_files([temp_input_path, temp_output_path])
        raise HTTPException(
            status_code=500,
            detail=f"Video analysis failed: {str(e)}"
        )


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=port,
        reload=False
    )
