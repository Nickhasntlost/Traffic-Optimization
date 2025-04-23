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

# Create directories before app initialization
UPLOAD_DIR = "uploads"
IMAGES_DIR = "images"
STATIC_DIR = "static"
TEMPLATES_DIR = "templates"

# Ensure all required directories exist
for directory in [UPLOAD_DIR, IMAGES_DIR, STATIC_DIR, TEMPLATES_DIR]:
    os.makedirs(directory, exist_ok=True)

# Create a CSS file if it doesn't exist
css_file_path = os.path.join(STATIC_DIR, "style.css")
if not os.path.exists(css_file_path):
    with open(css_file_path, "w") as f:
        f.write("""/* Main styles for the traffic monitoring application */
body {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif;
}

.video-container {
    width: 100%;
    min-height: 300px;
    display: flex;
    justify-content: center;
    align-items: center;
    border: 1px solid #e5e7eb;
    border-radius: 0.5rem;
    overflow: hidden;
    background-color: #f9fafb;
}

.traffic-light-container {
    width: 100%;
    min-height: 400px;
    display: flex;
    justify-content: center;
    align-items: center;
    border: 1px solid #e5e7eb;
    border-radius: 0.5rem;
    overflow: hidden;
    background-color: #f9fafb;
}

.count-card {
    transition: all 0.3s ease;
}

.count-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
}

/* Loading animation */
.loading {
    display: inline-block;
    width: 20px;
    height: 20px;
    border: 3px solid rgba(0, 0, 0, 0.1);
    border-radius: 50%;
    border-top-color: #3b82f6;
    animation: spin 1s ease-in-out infinite;
}

@keyframes spin {
    to {
        transform: rotate(360deg);
    }
}

/* Responsive adjustments */
@media (max-width: 768px) {
    .video-container, .traffic-light-container {
        min-height: 200px;
    }
}""")

app = FastAPI()

# Mount static files directory
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/images", StaticFiles(directory=IMAGES_DIR), name="images")

# Setup templates
templates = Jinja2Templates(directory=TEMPLATES_DIR)

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

# Traffic light simulation variables
traffic_light_active = False
traffic_light_frame = None
traffic_light_lock = threading.Lock()

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
    import random  # Add import for random module used in MockModel

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
                            else:
                                # Draw regular bounding box for other vehicles
                                cv2.rectangle(process_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                
                                # Add label
                                label = f"{class_name}"
                                cv2.putText(process_frame, label, (x1, y1 - 10), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
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
                
                # Save the latest frame for traffic light simulation
                if direction == "right":
                    cv2.imwrite(os.path.join(IMAGES_DIR, "North.jpg"), resized_frame)
                elif direction == "down":
                    cv2.imwrite(os.path.join(IMAGES_DIR, "South.jpg"), resized_frame)
                elif direction == "left":
                    cv2.imwrite(os.path.join(IMAGES_DIR, "East.jpg"), resized_frame)
                elif direction == "up":
                    cv2.imwrite(os.path.join(IMAGES_DIR, "West.jpg"), resized_frame)
            
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

# Traffic Light Simulation Functions
def detect_vehicles_in_image(image_path):
    """Detect vehicles in an image and return count"""
    if not os.path.exists(image_path):
        return 0
    
    try:
        image = cv2.imread(image_path)
        if image is None:
            return 0
        
        results = model(image)
        count = 0
        
        if not MOCK_MODEL and results[0].boxes is not None:
            for r in results:
                for box in r.boxes:
                    cls = int(box.cls.item())
                    if cls in [2, 3, 5, 7]:
                        count += 1
        else:
            # Mock detection
            count = np.random.randint(5, 20)
            
        return count
    except Exception as e:
        print(f"Error detecting vehicles in {image_path}: {e}")
        return 0

def calculate_green_time(count):
    """Calculate green light duration based on vehicle count"""
    base_time = 20  # Minimum green time (seconds)
    extra_time = (count // 10) * 5  # Additional 5 sec per 10 vehicles
    return min(base_time + extra_time, 60)  # Max green time = 60 sec

def run_traffic_light_simulation():
    """Run the traffic light simulation"""
    global traffic_light_active, traffic_light_frame
    
    try:
        traffic_light_active = True
        
        # Create the images directory if it doesn't exist
        os.makedirs(IMAGES_DIR, exist_ok=True)
        
        # Define image paths for each direction
        photo_paths = {
            "North": os.path.join(IMAGES_DIR, "North.jpg"),
            "South": os.path.join(IMAGES_DIR, "South.jpg"),
            "East": os.path.join(IMAGES_DIR, "East.jpg"),
            "West": os.path.join(IMAGES_DIR, "West.jpg")
        }
        
        # Create placeholder images if they don't exist
        for direction, path in photo_paths.items():
            try:
                if not os.path.exists(path):
                    placeholder = np.ones((416, 416, 3), dtype=np.uint8) * 240
                    cv2.putText(placeholder, f"No {direction} video", (120, 200), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                    cv2.imwrite(path, placeholder)
                    print(f"Created placeholder image for {direction} at {path}")
            except Exception as e:
                print(f"Error creating placeholder for {direction}: {e}")
                # Create a fallback placeholder in memory
                placeholder = np.ones((416, 416, 3), dtype=np.uint8) * 240
                cv2.putText(placeholder, f"Error: {direction}", (120, 200), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                _, buffer = cv2.imencode(".jpg", placeholder)
                with open(path, "wb") as f:
                    f.write(buffer)
        
        # Detect vehicles in each direction
        traffic_counts = {}
        for direction, path in photo_paths.items():
            try:
                count = detect_vehicles_in_image(path)
                traffic_counts[direction] = count
                print(f"Detected {count} vehicles in {direction} direction")
            except Exception as e:
                print(f"Error detecting vehicles in {direction}: {e}")
                # Use a default count if detection fails
                traffic_counts[direction] = 5
        
        # Determine traffic light timing
        green_times = {road: calculate_green_time(count) for road, count in traffic_counts.items()}
        
        # Sort roads by vehicle count (highest first)
        sorted_roads = sorted(green_times, key=lambda x: traffic_counts[x], reverse=True)
        
        # Run simulation for each road
        for road in sorted_roads:
            if not traffic_light_active:
                break
                
            countdowns = {r: green_times[r] if r == road else 0 for r in green_times}
            
            for t in range(green_times[road], 0, -1):
                if not traffic_light_active:
                    break
                    
                countdowns[road] = t
                
                # Create simulation frame
                canvas = np.ones((750, 1200, 3), dtype=np.uint8) * 255
                positions = {"North": (275, 50), "South": (275, 450), "East": (650, 250), "West": (50, 250)}
                signals = {r: (0, 255, 0) if r == road else (0, 0, 255) for r in traffic_counts}
                
                for r, (x, y) in positions.items():
                    try:
                        img = cv2.imread(photo_paths[r])
                        if img is not None:
                            img = cv2.resize(img, (250, 200))
                            canvas[y:y+200, x:x+250] = img
                        else:
                            # Handle case where image couldn't be read
                            placeholder = np.ones((200, 250, 3), dtype=np.uint8) * 240
                            cv2.putText(placeholder, f"No {r} image", (50, 100), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                            canvas[y:y+200, x:x+250] = placeholder
                    except Exception as e:
                        print(f"Error placing {r} image in simulation: {e}")
                        # Add placeholder for error
                        placeholder = np.ones((200, 250, 3), dtype=np.uint8) * 240
                        cv2.putText(placeholder, f"Error: {r}", (50, 100), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                        canvas[y:y+200, x:x+250] = placeholder
                    
                    # Traffic light
                    cv2.circle(canvas, (x+125, y+220), 20, signals[r], -1)
                    
                    # Countdown
                    cv2.putText(canvas, f"{countdowns[r]}s", (x+100, y+250), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
                    
                    # Vehicle count
                    cv2.putText(canvas, f"Vehicles: {traffic_counts[r]}", (x+80, y+280), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
                
                # Add title
                cv2.putText(canvas, "Traffic Light Simulation", (450, 700), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
                
                # Update the traffic light frame
                with traffic_light_lock:
                    _, buffer = cv2.imencode(".jpg", canvas)
                    traffic_light_frame = buffer.tobytes()
                
                # Wait 1 second
                time.sleep(1)
        
        # Simulation complete
        canvas = np.ones((750, 1200, 3), dtype=np.uint8) * 255
        cv2.putText(canvas, "Simulation Complete - Click 'Start Simulation' to run again", (300, 375), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        
        with traffic_light_lock:
            _, buffer = cv2.imencode(".jpg", canvas)
            traffic_light_frame = buffer.tobytes()
            
        traffic_light_active = False
        
    except Exception as e:
        print(f"Error in traffic light simulation: {e}")
        # Create an error message frame
        try:
            canvas = np.ones((750, 1200, 3), dtype=np.uint8) * 255
            cv2.putText(canvas, f"Error in simulation: {str(e)}", (300, 375), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            cv2.putText(canvas, "Click 'Start Simulation' to try again", (300, 425), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            
            with traffic_light_lock:
                _, buffer = cv2.imencode(".jpg", canvas)
                traffic_light_frame = buffer.tobytes()
        except:
            pass
        
        traffic_light_active = False

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/traffic_light", response_class=HTMLResponse)
async def traffic_light(request: Request):
    return templates.TemplateResponse("traffic_light.html", {"request": request})

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

    print("✅ /upload_videos/ endpoint called.")
    print(f"Uploaded files: right={right_video.filename if right_video else None}, "
      f"down={down_video.filename if down_video else None}, "
      f"left={left_video.filename if left_video else None}, "
      f"up={up_video.filename if up_video else None}")
    
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
    print("✅ ROI values parsed:")
    for direction, region in counting_regions.items():
        print(f"  {direction}: {region}")

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
    print("✅ Video paths updated:")
    for direction, path in video_paths.items():
        print(f"  {direction}: {path}")
    
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

@app.get("/traffic_light_feed")
async def traffic_light_feed():
    async def generate():
        global traffic_light_frame, traffic_light_active
        
        # Add a counter to track empty frames
        empty_frames = 0
        max_empty_frames = 50  # Wait for this many empty frames before giving up
        
        while traffic_light_active:
            try:
                with traffic_light_lock:
                    if traffic_light_frame is not None:
                        yield (b"--frame\r\n"
                               b"Content-Type: image/jpeg\r\n\r\n" + traffic_light_frame + b"\r\n")
                        empty_frames = 0  # Reset counter when we get a frame
                    else:
                        empty_frames += 1
                        if empty_frames > max_empty_frames:
                            print("No traffic light frames received after waiting, stopping stream")
                            break
                await asyncio.sleep(0.1)
            except Exception as e:
                print(f"Error in traffic light feed stream: {e}")
                # Try to continue despite errors
                await asyncio.sleep(0.5)
    
    try:
        # Create initial placeholder
        placeholder = np.ones((750, 1200, 3), dtype=np.uint8) * 240
        cv2.putText(placeholder, "Starting traffic light simulation...", (400, 375), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        _, buffer = cv2.imencode(".jpg", placeholder)
        
        # If simulation is not active, return placeholder
        if not traffic_light_active:
            return StreamingResponse(io.BytesIO(buffer.tobytes()), media_type="image/jpeg")
        
        return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")
    except Exception as e:
        print(f"Error setting up traffic light feed: {e}")
        # Return a static error image instead of failing
        error_img = np.ones((750, 1200, 3), dtype=np.uint8) * 240
        cv2.putText(error_img, f"Error: {str(e)}", (400, 375), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        _, buffer = cv2.imencode(".jpg", error_img)
        return StreamingResponse(io.BytesIO(buffer.tobytes()), media_type="image/jpeg")

@app.post("/start_traffic_simulation")
async def start_traffic_simulation(background_tasks: BackgroundTasks):
    try:
        global traffic_light_active, traffic_light_frame

        # Stop any existing simulation
        traffic_light_active = False
        time.sleep(1)
        
        # Create an initial frame to show while starting
        canvas = np.ones((750, 1200, 3), dtype=np.uint8) * 255
        cv2.putText(canvas, "Starting traffic light simulation...", (400, 375), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        
        with traffic_light_lock:
            _, buffer = cv2.imencode(".jpg", canvas)
            traffic_light_frame = buffer.tobytes()

        # Start new simulation in background
        background_tasks.add_task(run_traffic_light_simulation)

        return {"message": "Traffic light simulation started"}
    except Exception as e:
        print(f"Error starting simulation: {e}")
        return JSONResponse({"error": f"Failed to start simulation: {str(e)}"}, status_code=500)


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
