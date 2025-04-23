from fastapi import FastAPI, UploadFile, File, BackgroundTasks, Request
from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import cv2
import numpy as np
import threading
import shutil
import os
from typing import Optional

app = FastAPI()

# Create directories
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs("static", exist_ok=True)
os.makedirs("templates", exist_ok=True)

# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Setup templates
templates = Jinja2Templates(directory="templates")

# Global variables
video_path: Optional[str] = None
processing_active = False
vehicle_counts = {"car": 0, "truck": 0, "bus": 0, "motorcycle": 0, "bicycle": 0}
frame_buffer = None
frame_lock = threading.Lock()

# Check if YOLO is available, otherwise use a mock
try:
    from ultralytics import YOLO
    model = YOLO("yolov8n.pt")  # Load YOLO model
    MOCK_MODEL = False
except ImportError:
    print("YOLO not installed. Using mock detection.")
    MOCK_MODEL = True
    
    class MockModel:
        def __call__(self, frame):
            class MockResults:
                def __init__(self):
                    self.boxes = None
                    self.names = {0: "car", 1: "truck", 2: "bus", 3: "motorcycle", 4: "bicycle"}
            return [MockResults()]
    
    model = MockModel()

def process_video_background(video_file):
    global vehicle_counts, frame_buffer, processing_active
    
    if not os.path.exists(video_file):
        processing_active = False
        return
    
    processing_active = True
    cap = cv2.VideoCapture(video_file)
    
    while cap.isOpened() and processing_active:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame with YOLO
        results = model(frame)
        frame_counts = {"car": 0, "truck": 0, "bus": 0, "motorcycle": 0, "bicycle": 0}
        
        if not MOCK_MODEL and results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            cls_ids = results[0].boxes.cls.int().cpu().numpy()
            
            for box, cls_id in zip(boxes, cls_ids):
                x1, y1, x2, y2 = box.astype(int)
                class_name = results[0].names[cls_id]
                if class_name in frame_counts:
                    frame_counts[class_name] += 1
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, class_name, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        else:
            # Mock detection for testing
            frame_counts = {"car": 2, "truck": 1, "bus": 0, "motorcycle": 1, "bicycle": 0}
            cv2.putText(frame, "Mock Detection", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Update counts and frame buffer
        vehicle_counts = frame_counts
        
        with frame_lock:
            _, buffer = cv2.imencode(".jpg", frame)
            frame_buffer = buffer.tobytes()
        
    cap.release()
    processing_active = False

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload/")
async def upload_video(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    global video_path, processing_active
    
    # Stop any existing processing
    processing_active = False
    
    # Save the uploaded file
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    video_path = file_path
    
    # Start processing in background
    background_tasks.add_task(process_video_background, video_path)
    
    return {"message": "Video uploaded successfully", "filename": file.filename}

@app.get("/video_feed")
async def video_feed():
    async def generate():
        global frame_buffer
        while processing_active:
            with frame_lock:
                if frame_buffer is not None:
                    yield (b"--frame\r\n"
                           b"Content-Type: image/jpeg\r\n\r\n" + frame_buffer + b"\r\n")
            await asyncio.sleep(0.1)
    
    if video_path is None or not processing_active:
        return JSONResponse({"error": "No video is being processed"}, status_code=404)
    
    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/vehicle_counts")
async def get_vehicle_counts():
    return JSONResponse(vehicle_counts)

@app.get("/process_latest_video")
async def process_latest_video(background_tasks: BackgroundTasks):
    global video_path, processing_active
    
    # Stop any existing processing
    processing_active = False
    
    # Find the latest uploaded file
    files = [f for f in os.listdir(UPLOAD_DIR) if f.endswith((".mp4", ".avi", ".mov"))]
    if not files:
        return JSONResponse({"error": "No video found in uploads folder"}, status_code=404)
    
    latest_video = max(files, key=lambda f: os.path.getctime(os.path.join(UPLOAD_DIR, f)))
    video_path = os.path.join(UPLOAD_DIR, latest_video)
    
    # Start processing in background
    background_tasks.add_task(process_video_background, video_path)
    
    return {"message": f"Processing {latest_video}"}

if __name__ == "__main__":
    import uvicorn
    import asyncio
    uvicorn.run(app, host="0.0.0.0", port=8000)
else:
    import asyncio

