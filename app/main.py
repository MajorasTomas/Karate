"""
FastAPI application for karate pose analysis
"""

import os
import logging
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import Response
from pydantic import BaseModel, HttpUrl
from typing import Optional
import uvicorn

from app.pose_analyzer import KaratePoseAnalyzer
from app.video_processor import VideoProcessor
from app.utils import download_from_gcs, cleanup_temp_files

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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


@app.post("/analyze")
async def analyze_video(
    request: AnalyzeRequest,
    background_tasks: BackgroundTasks
):
    """
    Analyze karate pose from user video
    
    Args:
        request: Contains video_url from Google Cloud Storage
        background_tasks: FastAPI background tasks for cleanup
        
    Returns:
        Video buffer with pose analysis overlay
        Filename in response headers
    """
    temp_input_path = None
    temp_output_path = None
    
    try:
        logger.info(f"Received analysis request for video: {request.video_url}")
        
        # Download video from GCS
        logger.info("Downloading video from Google Cloud Storage...")
        temp_input_path = download_from_gcs(str(request.video_url))
        
        if not os.path.exists(temp_input_path):
            raise HTTPException(status_code=400, detail="Failed to download video")
        
        # Get analyzer instance
        pose_analyzer = get_analyzer()
        
        # Process video
        logger.info("Processing video with pose analysis...")
        temp_output_path = pose_analyzer.analyze_video(temp_input_path)
        
        if not os.path.exists(temp_output_path):
            raise HTTPException(
                status_code=500, 
                detail="Video processing failed"
            )
        
        # Read processed video as buffer
        logger.info("Reading processed video...")
        with open(temp_output_path, 'rb') as video_file:
            video_buffer = video_file.read()
        
        # Generate dynamic filename based on user_id or timestamp
        filename = _generate_output_filename(request.user_id)
        
        # Schedule cleanup in background
        background_tasks.add_task(
            cleanup_temp_files, 
            [temp_input_path, temp_output_path]
        )
        
        logger.info(f"Video analysis completed successfully: {filename}")
        
        # Return video buffer with filename in headers
        return Response(
            content=video_buffer,
            media_type="video/mp4",
            headers={
                "Content-Disposition": f"attachment; filename={filename}",
                "X-Filename": filename,  # Easy-to-access custom header
                "X-File-Size": str(len(video_buffer)),
                "X-User-ID": request.user_id or "anonymous"
            }
        )
        
    except Exception as e:
        logger.error(f"Error during video analysis: {str(e)}", exc_info=True)
        
        # Cleanup on error
        cleanup_temp_files([temp_input_path, temp_output_path])
        
        raise HTTPException(
            status_code=500,
            detail=f"Video analysis failed: {str(e)}"
        )


def _generate_output_filename(user_id: Optional[str]) -> str:
    """
    Generate output filename based on user_id or timestamp
    
    Args:
        user_id: Optional user identifier
        
    Returns:
        Generated filename (e.g., "analyzed_user123.mp4" or "analyzed_20251116_143052.mp4")
    """
    from datetime import datetime
    
    if user_id:
        # Use user ID if provided
        # Clean user_id to make it safe for filenames
        safe_user_id = "".join(c for c in user_id if c.isalnum() or c in ('-', '_'))
        filename = f"analyzed_{safe_user_id}.mp4"
    else:
        # Use timestamp if no user_id
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"analyzed_{timestamp}.mp4"
    
    return filename



if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=port,
        reload=False
    )
