from fastapi import FastAPI, UploadFile, File, BackgroundTasks, Request, Form
from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import cv2
import numpy as np
import threading
import shutil
import os
import time
import asyncio
from typing import Optional, Dict, List
import io
import random

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
video_paths = {
    "right": None,
    "down": None,
    "left": None,
    "up": None
}

processing_active = False
vehicle_counts = {
    "right": {"car": 0, "truck": 0, "bus": 0, "motorcycle": 0, "bicycle": 0},
    "down": {"car": 0, "truck": 0, "bus": 0, "motorcycle": 0, "bicycle": 0},
    "left": {"car": 0, "truck": 0, "bus": 0, "motorcycle": 0, "bicycle": 0},
    "up": {"car": 0, "truck": 0, "bus": 0, "motorcycle": 0, "bicycle": 0}
}

# Store processed frames
processed_frames = {
    "right": None,
    "down": None,
    "left": None,
    "up": None
}

# Combined frame for all directions
combined_frame = None
frame_lock = threading.Lock()

# Define counting regions for each direction (x1, y1, x2, y2 in relative coordinates)
counting_regions = {
    "right": {"x1": 0.1, "y1": 0.5, "x2": 0.7, "y2": 1.0},
    "down": {"x1": 0.1, "y1": 0.5, "x2": 0.7, "y2": 1.0},
    "left": {"x1": 0.1, "y1": 0.5, "x2": 0.7, "y2": 1.0},
    "up": {"x1": 0.1, "y1": 0.5, "x2": 0.7, "y2": 1.0}
}

# Check if YOLO is available, otherwise use a mock
try:
    from ultralytics import YOLO
    model = YOLO("yolov8n.pt")  # Load YOLO model
    MOCK_MODEL = False
    print("YOLO model loaded successfully")
except ImportError:
    print("YOLO not installed. Using mock detection.")
    MOCK_MODEL = True
    
    class MockModel:
        def __init__(self):
            self.vehicle_classes = {0: "car", 1: "truck", 2: "bus", 3: "motorcycle", 4: "bicycle"}
            
        def __call__(self, frame):
            class MockResults:
                def __init__(self, frame_shape):
                    self.names = {0: "car", 1: "truck", 2: "bus", 3: "motorcycle", 4: "bicycle"}
                    
                    # Create mock boxes
                    height, width = frame_shape[:2]
                    num_objects = random.randint(3, 8)  # Random number of objects
                    
                    # Create mock boxes with numpy arrays to match YOLO format
                    boxes_list = []
                    cls_list = []
                    
                    for _ in range(num_objects):
                        # Random box dimensions
                        box_width = random.randint(50, 150)
                        box_height = random.randint(50, 100)
                        
                        # Random position
                        x1 = random.randint(0, width - box_width)
                        y1 = random.randint(0, height - box_height)
                        x2 = x1 + box_width
                        y2 = y1 + box_height
                        
                        boxes_list.append([x1, y1, x2, y2])
                        cls_list.append(random.randint(0, 4))  # Random vehicle class
                    
                    # Convert to numpy arrays
                    self.boxes = type('obj', (), {
                        'xyxy': type('obj', (), {
                            'cpu': lambda: type('obj', (), {
                                'numpy': lambda: np.array(boxes_list)
                            })()
                        })(),
                        'cls': type('obj', (), {
                            'int': lambda: type('obj', (), {
                                'cpu': lambda: type('obj', (), {
                                    'numpy': lambda: np.array(cls_list)
                                })()
                            })()
                        })()
                    })
            
            return [MockResults(frame.shape)]
    
    model = MockModel()

def is_in_region(box, region, frame_height, frame_width):
    """Check if a bounding box is inside the counting region"""
    x1, y1, x2, y2 = box
    
    # Convert region coordinates from relative to absolute
    region_x1 = int(region["x1"] * frame_width)
    region_y1 = int(region["y1"] * frame_height)
    region_x2 = int(region["x2"] * frame_width)
    region_y2 = int(region["y2"] * frame_height)
    
    # Calculate center of the box
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    
    # Check if center is inside the region
    if (region_x1 <= center_x <= region_x2) and (region_y1 <= center_y <= region_y2):
        return True
    
    return False

def process_video(direction):
    """Process a video stream and count vehicles in a region"""
    global vehicle_counts, processed_frames, processing_active
    
    video_path = video_paths[direction]
    if not video_path or not os.path.exists(video_path):
        print(f"Video path for {direction} is not valid: {video_path}")
        return
    
    try:
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Error opening video file: {video_path}")
            return
        
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Get region coordinates for this direction
        region = counting_regions[direction]
        
        # Convert region coordinates from relative to absolute
        region_x1 = int(region["x1"] * frame_width)
        region_y1 = int(region["y1"] * frame_height)
        region_x2 = int(region["x2"] * frame_width)
        region_y2 = int(region["y2"] * frame_height)
        
        frame_count = 0
        
        while cap.isOpened() and processing_active:
            ret, frame = cap.read()
            if not ret:
                # If video ended, loop back to beginning
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            
            frame_count += 1
            
            # Process every 3rd frame to improve performance but still show smooth detection
            if frame_count % 3 != 0:
                continue
            
            # Create a copy for processing
            process_frame = frame.copy()
            
            # Draw the counting region
            cv2.rectangle(process_frame, (region_x1, region_y1), (region_x2, region_y2), (0, 255, 255), 2)
            
            # Reset counts for this frame
            frame_counts = {
                "car": 0,
                "truck": 0,
                "bus": 0,
                "motorcycle": 0,
                "bicycle": 0
            }
            
            # Run detection (either YOLO or mock)
            results = model(process_frame)
            
            if results[0].boxes is not None:
                try:
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    cls_ids = results[0].boxes.cls.int().cpu().numpy()
                    
                    for box, cls_id in zip(boxes, cls_ids):
                        x1, y1, x2, y2 = box.astype(int)
                        
                        # Get class name
                        class_name = results[0].names[cls_id]
                        
                        # Only process vehicle classes we're interested in
                        if class_name in frame_counts:
                            # Check if this object is in the counting region
                            if is_in_region(box, region, frame_height, frame_width):
                                # Increment count for this frame
                                frame_counts[class_name] += 1
                                
                                # Draw bounding box with different color for vehicles in region
                                cv2.rectangle(process_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                                
                                # Add label
                                label = f"{class_name}"
                                cv2.putText(process_frame, label, (x1, y1 - 10), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        #    else:
                         #       # Draw regular bounding box for other vehicles
                          #      cv2.rectangle(process_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                
                           #     # Add label
                            #    label = f"{class_name}"
                             #   cv2.putText(process_frame, label, (x1, y1 - 10), 
                              #              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                except Exception as e:
                    print(f"Error processing detection results: {e}")
            
            # Update the global vehicle counts with the current frame counts
            with frame_lock:
                vehicle_counts[direction] = frame_counts.copy()
            
            # Add count text to frame
            y_offset = 30
            for vehicle_type, count in frame_counts.items():
                text = f"{vehicle_type}: {count}"
                cv2.putText(process_frame, text, (10, y_offset), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                y_offset += 30
            
            # Add direction label
            cv2.putText(process_frame, direction.upper(), (frame_width - 100, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Resize frame for display
            resized_frame = cv2.resize(process_frame, (416, 416))
            
            # Store the processed frame
            with frame_lock:
                processed_frames[direction] = resized_frame.copy()
            
            # Small delay to reduce CPU usage
            time.sleep(0.03)
            
    except Exception as e:
        print(f"Error processing video {direction}: {e}")
    finally:
        if 'cap' in locals() and cap.isOpened():
            cap.release()

def create_combined_frame():
    """Create a combined frame with all four video streams"""
    global processed_frames, combined_frame, processing_active
    
    while processing_active:
        try:
            # Create a blank canvas
            canvas = np.ones((900, 900, 3), dtype=np.uint8) * 255
            
            # Positions for each direction
            positions = {
                "right": (450, 450),  # Right side (right middle)
                "down": (450, 0),     # Down (top right)
                "left": (0, 0),       # Left side (top left)
                "up": (0, 450)        # Up (bottom left)
            }
            
            with frame_lock:
                # Check if we have any processed frames - FIX: use proper check for NumPy arrays
                any_frames = False
                for frame in processed_frames.values():
                    if frame is not None:
                        any_frames = True
                        break
                
                if not any_frames:
                    # If no frames are available yet, add a message
                    cv2.putText(canvas, "Processing videos, please wait...", (250, 450), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
                else:
                    # Place each processed frame on the canvas
                    for direction, (x, y) in positions.items():
                        if processed_frames[direction] is not None:
                            try:
                                frame = processed_frames[direction].copy()  # Make a copy to avoid issues
                                h, w = frame.shape[:2]
                                
                                # Check if the frame will fit in the allocated space
                                if y + h <= canvas.shape[0] and x + w <= canvas.shape[1]:
                                    canvas[y:y+h, x:x+w] = frame
                            except Exception as e:
                                print(f"Error placing frame for {direction}: {e}")
                        else:
                            # Add placeholder for missing direction
                            placeholder = np.ones((416, 416, 3), dtype=np.uint8) * 240
                            cv2.putText(placeholder, f"No {direction} video", (120, 200), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                            canvas[y:y+416, x:x+416] = placeholder
                
                # Add title and timestamp
                cv2.putText(canvas, "Traffic Junction Monitoring", (300, 850), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                cv2.putText(canvas, timestamp, (300, 880), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                
                # Update the combined frame
                _, buffer = cv2.imencode(".jpg", canvas)
                combined_frame = buffer.tobytes()
            
            # Small delay to reduce CPU usage
            time.sleep(0.1)
            
        except Exception as e:
            print(f"Error in combined frame creation: {e}")

def start_processing():
    """Start processing all videos"""
    global processing_active, combined_frame
    
    # Stop any existing processing
    processing_active = False
    time.sleep(1)  # Give time for threads to stop
    
    # Create initial placeholder frame
    canvas = np.ones((900, 900, 3), dtype=np.uint8) * 255
    cv2.putText(canvas, "Starting video processing...", (250, 450), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    _, buffer = cv2.imencode(".jpg", canvas)
    combined_frame = buffer.tobytes()
    
    # Start new processing
    processing_active = True
    
    # Start video processing threads
    threads = []
    videos_started = 0
    
    for direction in video_paths:
        if video_paths[direction] and os.path.exists(video_paths[direction]):
            thread = threading.Thread(target=process_video, args=(direction,))
            thread.daemon = True
            thread.start()
            threads.append(thread)
            videos_started += 1
            print(f"Started processing thread for {direction} video")
    
    print(f"Started {videos_started} video processing threads")
    
    # Start combined frame creation thread
    combined_thread = threading.Thread(target=create_combined_frame)
    combined_thread.daemon = True
    combined_thread.start()
    print("Started combined frame creation thread")

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index2.html", {"request": request})

@app.post("/upload_videos/")
async def upload_videos(
    background_tasks: BackgroundTasks,
    right_video: Optional[UploadFile] = File(None),
    down_video: Optional[UploadFile] = File(None),
    left_video: Optional[UploadFile] = File(None),
    up_video: Optional[UploadFile] = File(None),
    right_roi: Optional[str] = Form("0.1,0.5,0.7,1.0"),
    down_roi: Optional[str] = Form("0.1,0.5,0.7,1.0"),
    left_roi: Optional[str] = Form("0.1,0.5,0.7,1.0"),
    up_roi: Optional[str] = Form("0.1,0.5,0.7,1.0")
):
    global video_paths, counting_regions, processing_active
    
    # Stop any existing processing
    processing_active = False
    time.sleep(1)  # Give time for threads to stop
    
    # Process uploaded videos
    uploaded_videos = {
        "right": right_video,
        "down": down_video,
        "left": left_video,
        "up": up_video
    }
    
    # Process ROI values
    roi_values = {
        "right": right_roi,
        "down": down_roi,
        "left": left_roi,
        "up": up_roi
    }
    
    # Update ROI settings
    for direction, roi_str in roi_values.items():
        try:
            x1, y1, x2, y2 = map(float, roi_str.split(','))
            counting_regions[direction] = {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
        except Exception as e:
            print(f"Error parsing ROI for {direction}: {e}")
    
    # Track if any videos were uploaded
    any_videos_uploaded = False
    
    # Save uploaded videos
    for direction, file in uploaded_videos.items():
        if file and file.filename:
            try:
                file_path = os.path.join(UPLOAD_DIR, f"{direction}_{file.filename}")
                with open(file_path, "wb") as buffer:
                    shutil.copyfileobj(file.file, buffer)
                video_paths[direction] = file_path
                any_videos_uploaded = True
                print(f"Saved {direction} video to {file_path}")
            except Exception as e:
                print(f"Error saving {direction} video: {e}")
    
    if not any_videos_uploaded:
        return {"error": "No videos were uploaded", "status": "failed"}
    
    # Start processing in background
    background_tasks.add_task(start_processing)
    
    return {
        "message": "Videos uploaded successfully",
        "uploaded": {direction: (file.filename if file and file.filename else None) for direction, file in uploaded_videos.items()},
        "roi_settings": counting_regions
    }

@app.get("/video_feed")
async def video_feed():
    async def generate():
        global combined_frame, processing_active
        
        # Add a counter to track empty frames
        empty_frames = 0
        max_empty_frames = 50  # Wait for this many empty frames before giving up
        
        while processing_active:
            if combined_frame is not None:
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" + combined_frame + b"\r\n")
                empty_frames = 0  # Reset counter when we get a frame
            else:
                empty_frames += 1
                if empty_frames > max_empty_frames:
                    print("No frames received after waiting, stopping stream")
                    break
            await asyncio.sleep(0.1)
    
    # Check if any videos have been uploaded
    if not any(video_paths.values()):
        print("No videos have been uploaded yet")
        # Instead of returning 404, return a placeholder image
        placeholder = np.ones((600, 800, 3), dtype=np.uint8) * 240
        cv2.putText(placeholder, "No videos uploaded yet", (200, 300), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        _, buffer = cv2.imencode(".jpg", placeholder)
        return StreamingResponse(io.BytesIO(buffer.tobytes()), media_type="image/jpeg")
    
    # Log which videos are available
    print(f"Available videos: {[k for k, v in video_paths.items() if v]}")
    
    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/vehicle_counts")
async def get_vehicle_counts():
    with frame_lock:
        return JSONResponse(vehicle_counts)

@app.get("/update_roi/{direction}")
async def update_roi(direction: str, x1: float, y1: float, x2: float, y2: float):
    if direction not in counting_regions:
        return JSONResponse({"error": f"Invalid direction: {direction}"}, status_code=400)
    
    counting_regions[direction] = {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
    return {"message": f"ROI for {direction} updated successfully", "roi": counting_regions[direction]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
else:
    import asyncio