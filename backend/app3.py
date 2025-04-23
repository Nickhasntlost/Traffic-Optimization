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
import subprocess
from typing import Optional, Dict, List
import io
import json

import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

#RL
import random
from collections import defaultdict

class RLAgent:
    def __init__(self, actions, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.q_table = defaultdict(lambda: {a: 0.0 for a in actions})
        self.alpha = alpha      # learning rate
        self.gamma = gamma      # discount factor
        self.epsilon = epsilon  # exploration probability
        self.actions = actions

    def get_state(self, traffic_counts):
        return tuple(traffic_counts[r] // 5 for r in ["North", "South", "East", "West"])

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        return max(self.q_table[state], key=self.q_table[state].get)

    def update(self, state, action, reward, next_state):
        old_value = self.q_table[state][action]
        future = max(self.q_table[next_state].values())
        self.q_table[state][action] = old_value + self.alpha * (reward + self.gamma * future - old_value)

# GLOBAL AGENT
rl_agent = RLAgent(actions=["North", "South", "East", "West"])

congestion_history = []
reward_history = []
epsilon_history = []
action_counts = {"North": 0, "South": 0, "East": 0, "West": 0}


# Save model
def save_model(agent, path="dqn_model.pth"):
    torch.save(agent.model.state_dict(), path)
    print(f"✅ Model saved to {path}")

# Load model
def load_model(agent, path="dqn_model.pth"):
    if os.path.exists(path):
        agent.model.load_state_dict(torch.load(path))
        agent.target_model.load_state_dict(agent.model.state_dict())
        print(f"✅ Model loaded from {path}")
    else:
        print("⚠️ No saved model found to load.")

class DQN(nn.Module):
    def __init__(self, input_size=4, output_size=4):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.out = nn.Linear(64, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.out(x)

class DQNAgent:
    def __init__(self):
        self.model = DQN()
        self.target_model = DQN()
        self.memory = deque(maxlen=5000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.batch_size = 32
        self.update_target_steps = 10
        self.steps = 0

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(4)
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            return torch.argmax(self.model(state_tensor)).item()

    def remember(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states = zip(*batch)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)

        q_values = self.model(states).gather(1, actions)
        next_q_values = self.target_model(next_states).max(1)[0].unsqueeze(1)
        target = rewards + self.gamma * next_q_values

        loss = nn.MSELoss()(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.steps += 1
        if self.steps % self.update_target_steps == 0:
            self.target_model.load_state_dict(self.model.state_dict())
            if self.steps % 100 == 0:
                save_model(self)


        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay



# Ensure DQNAgent and load_model are defined before usage
dqn_agent = DQNAgent()
try:
    load_model(dqn_agent)
except Exception as e:
    print(f"Error loading model: {e}")
    print("Initializing new model")


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

persistent_state = {
    "videos_uploaded": False,
    "last_upload_time": None
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

# Modified cases simulation variables
modified_cases_active = False
modified_cases_process = None
modified_cases_frame = None
modified_cases_lock = threading.Lock()

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
            try:
                ret, frame = cap.read()
                if not ret:
                    # If video ended, loop back to beginning
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                
                frame_count += 1
                
                # Process every 5th frame to improve performance (was 3rd frame)
                if frame_count % 5 != 0:
                    continue
                
                # Create a copy for processing
                process_frame = frame.copy()
                
                # Resize frame for faster processing if it's large
                h, w = process_frame.shape[:2]
                if w > 1280:  # If width is greater than 1280px
                    scale_factor = 1280 / w
                    process_frame = cv2.resize(process_frame, (int(w * scale_factor), int(h * scale_factor)))
                
                # Draw the counting region
                region_x1 = int(region["x1"] * process_frame.shape[1])
                region_y1 = int(region["y1"] * process_frame.shape[0])
                region_x2 = int(region["x2"] * process_frame.shape[1])
                region_y2 = int(region["y2"] * process_frame.shape[0])
                
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
                                if is_in_region(box, region, process_frame.shape[0], process_frame.shape[1]):
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
                cv2.putText(process_frame, direction.upper(), (process_frame.shape[1] - 100, 30), 
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
            
            except Exception as e:
                print(f"Error processing frame for {direction}: {e}")
                # Continue processing despite errors
                continue
                
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

        # Run the simulation continuously until stopped
        while traffic_light_active:
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
                    traffic_counts[direction] = 5  # Default to 5 vehicles instead of 0
            
            # Sort roads by vehicle count (highest first)
            sorted_roads = sorted(traffic_counts.items(), key=lambda x: x[1], reverse=True)
            
            # Process each road in order of vehicle count
            for road, count in sorted_roads:
                # Skip roads with no vehicles
                if count <= 0:
                    continue
                    
                # Calculate green time based on vehicle count
                green_time = calculate_green_time(count)
                
                # Set all roads to red except the current one
                countdowns = {r: 0 for r in traffic_counts}
                countdowns[road] = green_time
                
                # Use DQN agent
                state = [traffic_counts[r] / 100 for r in ["North", "South", "East", "West"]]
                action = dqn_agent.act(state)
                selected_road = ["North", "South", "East", "West"][action]
                
                # If DQN selected a road with no vehicles, use our sorted selection instead
                if traffic_counts[selected_road] <= 0:
                    selected_road = road
                
                # Run the countdown for this road
                for t in range(green_time, 0, -1):
                    if not traffic_light_active:
                        break
                        
                    countdowns[selected_road] = t
                    
                    # Simulate traffic effect: reduce vehicle count for green road
                    next_counts = traffic_counts.copy()
                    next_counts[selected_road] = max(0, next_counts[selected_road] - 1)

                    # Calculate total congestion
                    total_congestion = sum(traffic_counts.values())
                    congestion_history.append(total_congestion)

                    # Keep only the latest 100 data points
                    if len(congestion_history) > 100:
                        congestion_history.pop(0)

                    
                    next_state = [next_counts[r] / 100 for r in ["North", "South", "East", "West"]]
                    reward = -sum(next_counts[r] for r in traffic_counts if r != selected_road)
                    
                    # Remember this transition
                    dqn_agent.remember(state, action, reward, next_state)

                    # Record for charts
                    reward_history.append(reward)
                    epsilon_history.append(dqn_agent.epsilon)

                    # Track how many times each direction was selected
                    action_counts[selected_road] += 1

                    # Optionally, limit to last 100 records
                    if len(reward_history) > 100:
                        reward_history.pop(0)
                    if len(epsilon_history) > 100:
                        epsilon_history.pop(0)

                    
                    # Train the model
                    dqn_agent.train_step()
                    
                    # Create simulation frame
                    canvas = np.ones((750, 1200, 3), dtype=np.uint8) * 255
                    
                    # Slightly adjusted positions for better visual balance
                    positions = {
                        "North": (475, 50),
                        "South": (475, 450),
                        "East": (850, 250),
                        "West": (100, 250)
                    }
                    
                    signals = {r: (0, 255, 0) if r == selected_road else (0, 0, 255) for r in traffic_counts}
                    
                    for r, (x, y) in positions.items():
                        try:
                            img = cv2.imread(photo_paths[r])
                            if img is not None:
                                img = cv2.resize(img, (250, 200))
                                canvas[y:y+200, x:x+250] = img
                            else:
                                # Handle case where image couldn't be read
                                placeholder = np.ones((200, 250, 3), dtype=np.uint8) * 240
                                cv2.putText(placeholder, f"No {r} image", (30, 100), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                                canvas[y:y+200, x:x+250] = placeholder
                        except Exception as e:
                            print(f"Error placing {r} image in simulation: {e}")
                            # Add placeholder for error
                            placeholder = np.ones((200, 250, 3), dtype=np.uint8) * 240
                            cv2.putText(placeholder, f"Error: {r}", (50, 100), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                            canvas[y:y+200, x:x+250] = placeholder
                            
                        # Adjusted offsets for better text/light alignment
                        cv2.circle(canvas, (x+125, y+210), 20, signals[r], -1)
                        
                        cv2.putText(canvas, f"{countdowns[r]}s", (x+90, y+250), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
                        
                        cv2.putText(canvas, f"Vehicles: {traffic_counts[r]}", (x+60, y+280), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
                    
                    # Add title and info
                    cv2.putText(canvas, "Traffic Light Simulation", (450, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
                    
                    cv2.putText(canvas, f"Current Green: {selected_road} - {countdowns[selected_road]}s remaining", (400, 720), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
                    
                    # Update the traffic light frame
                    with traffic_light_lock:
                        _, buffer = cv2.imencode(".jpg", canvas)
                        traffic_light_frame = buffer.tobytes()
                    
                    # Wait 1 second
                    time.sleep(1)
            
            # If no roads have vehicles, show a message
            if all(count <= 0 for count in traffic_counts.values()):
                canvas = np.ones((750, 1200, 3), dtype=np.uint8) * 255
                cv2.putText(canvas, "No vehicles detected in any direction", (350, 375), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
                
                with traffic_light_lock:
                    _, buffer = cv2.imencode(".jpg", canvas)
                    traffic_light_frame = buffer.tobytes()
                
                # Wait a bit before checking again
                time.sleep(3)
        
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

# Modified Cases Simulation Functions
def run_modified_cases_simulation():
    """Run the modified cases simulation"""
    global modified_cases_active, modified_cases_frame, modified_cases_process, vehicle_counts
    
    try:
        modified_cases_active = True
        
        # Create initial frame
        canvas = np.ones((750, 1200, 3), dtype=np.uint8) * 255
        cv2.putText(canvas, "Starting Modified Cases Simulation...", (350, 375), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        
        with modified_cases_lock:
            _, buffer = cv2.imencode(".jpg", canvas)
            modified_cases_frame = buffer.tobytes()
        
        # Start the modified_cases.py process
        cmd = ["python", "modified_cases.py"]
        modified_cases_process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Create a thread to read output from the process
        def read_output():
            while modified_cases_active and modified_cases_process.poll() is None:
                try:
                    line = modified_cases_process.stdout.readline().strip()
                    if line:
                        # Check if the line contains simulation data
                        if line.startswith("SIMULATION_DATA:"):
                            try:
                                # Extract the JSON data
                                json_str = line[len("SIMULATION_DATA:"):].strip()
                                data = json.loads(json_str)
                                
                                # Create a visualization of the simulation state
                                canvas = np.ones((750, 1200, 3), dtype=np.uint8) * 255
                                
                                # Draw title
                                cv2.putText(canvas, "Modified Cases Simulation", (450, 30), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
                                
                                # Draw elapsed time
                                cv2.putText(canvas, f"Elapsed Time: {data['elapsedTime']} seconds", (450, 60), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                                
                                # Draw traffic signals
                                signal_positions = {
                                    0: (300, 300),  # Right
                                    1: (600, 300),  # Down
                                    2: (600, 500),  # Left
                                    3: (300, 500)   # Up
                                }
                                
                                for signal in data["signalStatus"]:
                                    signal_id = signal["id"]
                                    state = signal["state"]
                                    time_remaining = signal["timeRemaining"]
                                    
                                    x, y = signal_positions[signal_id]
                                    
                                    # Draw signal circle with appropriate color
                                    if state == "red":
                                        color = (0, 0, 255)
                                    elif state == "yellow":
                                        color = (0, 255, 255)
                                    else:  # green
                                        color = (0, 255, 0)
                                    
                                    cv2.circle(canvas, (x, y), 30, color, -1)
                                    
                                    # Draw signal ID and time
                                    cv2.putText(canvas, f"Signal {signal_id+1}", (x-40, y-40), 
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                                    cv2.putText(canvas, f"{time_remaining}s", (x-15, y+5), 
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                                
                                # Draw vehicle counts
                                direction_names = {
                                    "right": "East",
                                    "down": "South",
                                    "left": "West",
                                    "up": "North"
                                }
                                
                                count_positions = {
                                    "right": (900, 200),
                                    "down": (900, 300),
                                    "left": (900, 400),
                                    "up": (900, 500)
                                }
                                
                                cv2.putText(canvas, "Vehicle Counts:", (800, 150), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                                
                                for direction, counts in data["vehicleCounts"].items():
                                    x, y = count_positions[direction]
                                    display_name = direction_names.get(direction, direction)
                                    
                                    cv2.putText(canvas, f"{display_name}:", (x-100, y), 
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                                    
                                    y_offset = 0
                                    for vehicle_type, count in counts.items():
                                        cv2.putText(canvas, f"  {vehicle_type}: {count}", (x, y + y_offset), 
                                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                                        y_offset += 25
                                
                                # Draw intersection diagram
                                cv2.rectangle(canvas, (400, 350), (500, 450), (200, 200, 200), -1)
                                
                                # Draw roads
                                cv2.rectangle(canvas, (200, 380), (400, 420), (100, 100, 100), -1)  # Left road
                                cv2.rectangle(canvas, (500, 380), (700, 420), (100, 100, 100), -1)  # Right road
                                cv2.rectangle(canvas, (430, 200), (470, 350), (100, 100, 100), -1)  # Top road
                                cv2.rectangle(canvas, (430, 450), (470, 600), (100, 100, 100), -1)  # Bottom road
                                
                                # Draw road markings
                                for i in range(200, 700, 30):
                                    if i < 400 or i >= 500:
                                        cv2.line(canvas, (i, 400), (i+15, 400), (255, 255, 255), 2)
                                
                                for i in range(200, 600, 30):
                                    if i < 350 or i >= 450:
                                        cv2.line(canvas, (450, i), (450, i+15), (255, 255, 255), 2)
                                
                                # Update the frame
                                with modified_cases_lock:
                                    _, buffer = cv2.imencode(".jpg", canvas)
                                    modified_cases_frame = buffer.tobytes()
                                
                            except json.JSONDecodeError as e:
                                print(f"Error parsing simulation data: {e}")
                            except Exception as e:
                                print(f"Error processing simulation data: {e}")
                        else:
                            print(f"Modified Cases Output: {line}")
                except Exception as e:
                    print(f"Error reading from modified_cases process: {e}")
                    break
        
        # Start the output reading thread
        output_thread = threading.Thread(target=read_output)
        output_thread.daemon = True
        output_thread.start()
        
        # Send vehicle counts to the simulation periodically
        while modified_cases_active and modified_cases_process.poll() is None:
            try:
                # Get current vehicle counts
                with frame_lock:
                    current_counts = vehicle_counts.copy()
                
                # Send counts to the simulation via API
                try:
                    import requests
                    response = requests.post("http://localhost:8000/vehicle_counts", json={"counts": current_counts})
                    if response.status_code != 200:
                        print(f"Error sending vehicle counts to simulation: {response.status_code}")
                except Exception as e:
                    print(f"Error sending vehicle counts to simulation: {e}")
                
                # Wait before sending next update
                time.sleep(1)
            except Exception as e:
                print(f"Error in modified cases update loop: {e}")
        
        # Simulation ended
        if modified_cases_process.poll() is not None:
            print(f"Modified cases process exited with code {modified_cases_process.returncode}")
            stderr = modified_cases_process.stderr.read()
            if stderr:
                print(f"Modified cases stderr: {stderr}")
        
        # Create final frame
        canvas = np.ones((750, 1200, 3), dtype=np.uint8) * 255
        cv2.putText(canvas, "Modified Cases Simulation Ended", (400, 375), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.putText(canvas, "Click 'Start Modified Cases' to run again", (400, 425), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        
        with modified_cases_lock:
            _, buffer = cv2.imencode(".jpg", canvas)
            modified_cases_frame = buffer.tobytes()
        
        modified_cases_active = False
        
    except Exception as e:
        print(f"Error in modified cases simulation: {e}")
        # Create an error message frame
        try:
            canvas = np.ones((750, 1200, 3), dtype=np.uint8) * 255
            cv2.putText(canvas, f"Error in simulation: {str(e)}", (300, 375), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            cv2.putText(canvas, "Click 'Start Modified Cases' to try again", (300, 425), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            
            with modified_cases_lock:
                _, buffer = cv2.imencode(".jpg", canvas)
                modified_cases_frame = buffer.tobytes()
        except:
            pass
        
        modified_cases_active = False

@app.post("/restart_processing")
async def restart_processing(background_tasks: BackgroundTasks):
    """Restart video processing"""
    global processing_active
    
    if not any(video_paths.values()):
        return JSONResponse({"error": "No videos available to process"}, status_code=400)
    
    # Start processing in background
    background_tasks.add_task(start_processing)
    
    return {"message": "Processing restarted successfully"}

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    # Pass persistent state to the template
    return templates.TemplateResponse("index.html", {
        "request": request,
        "persistent_state": persistent_state
    })

@app.get("/traffic_light", response_class=HTMLResponse)
async def traffic_light(request: Request):
    # Pass persistent state to the template
    return templates.TemplateResponse("traffic_light.html", {
        "request": request,
        "persistent_state": persistent_state
    })

@app.get("/modified_cases", response_class=HTMLResponse)
async def modified_cases_page(request: Request):
    # Pass persistent state to the template
    return templates.TemplateResponse("modified_cases.html", {
        "request": request,
        "persistent_state": persistent_state
    })

@app.get("/state")
async def get_state():
    """Return the current application state"""
    global persistent_state, video_paths, processing_active, traffic_light_active, modified_cases_active
    
    # Check if videos actually exist on disk
    videos_exist = {}
    for direction, path in video_paths.items():
        videos_exist[direction] = path is not None and os.path.exists(path)
    
    # Check if any videos exist
    any_videos_exist = any(videos_exist.values())
    
    # Update persistent state based on actual file existence
    if any_videos_exist and not persistent_state["videos_uploaded"]:
        persistent_state["videos_uploaded"] = True
        print("✅ Updated persistent state based on file existence check")
    
    return {
        "persistent_state": persistent_state,
        "videos_available": videos_exist,
        "any_videos_available": any_videos_exist,
        "processing_active": processing_active,
        "traffic_light_active": traffic_light_active,
        "modified_cases_active": modified_cases_active,
        "timestamp": time.time()
    }

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
    global video_paths, counting_regions, processing_active, persistent_state

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
                # Delete old video if it exists
                if video_paths[direction] and os.path.exists(video_paths[direction]):
                    try:
                        os.remove(video_paths[direction])
                        print(f"Deleted old {direction} video: {video_paths[direction]}")
                    except Exception as e:
                        print(f"Error deleting old {direction} video: {e}")
                
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
    
    # Upadte persistent state 
    persistent_state["videos_uploaded"] = True
    persistent_state["last_upload_time"] = time.time()
    print(f"✅ Updated persistent state: {persistent_state}")

    # Start processing in background
    background_tasks.add_task(start_processing)

    return {
        "message": "Videos uploaded successfully",
        "uploaded": {direction: (file.filename if file and file.filename else None) 
         for direction, file in uploaded_videos.items()},
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
            try:
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
            except Exception as e:
                print(f"Error in video feed stream: {e}")
                await asyncio.sleep(0.5)  # Add a delay before retrying
    
    # Check if any videos have been uploaded
    if not any(video_paths.values()):
        print("No videos have been uploaded yet")
        # Instead of returning 404, return a placeholder image
        placeholder = np.ones((600, 800, 3), dtype=np.uint8) * 240
        cv2.putText(placeholder, "No videos uploaded yet", (200, 300), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        _, buffer = cv2.imencode(".jpg", placeholder)
        return StreamingResponse(io.BytesIO(buffer.tobytes()), 
                                media_type="image/jpeg",
                                headers={"Cache-Control": "no-cache, no-store, must-revalidate"})
    
    # Log which videos are available
    print(f"Available videos: {[k for k, v in video_paths.items() if v]}")
    
    return StreamingResponse(generate(), 
                            media_type="multipart/x-mixed-replace; boundary=frame",
                            headers={"Cache-Control": "no-cache, no-store, must-revalidate"})

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

@app.get("/modified_cases_feed")
async def modified_cases_feed():
    async def generate():
        global modified_cases_frame, modified_cases_active
        
        # Add a counter to track empty frames
        empty_frames = 0
        max_empty_frames = 50  # Wait for this many empty frames before giving up
        
        while modified_cases_active:
            try:
                with modified_cases_lock:
                    if modified_cases_frame is not None:
                        yield (b"--frame\r\n"
                               b"Content-Type: image/jpeg\r\n\r\n" + modified_cases_frame + b"\r\n")
                        empty_frames = 0  # Reset counter when we get a frame
                    else:
                        empty_frames += 1
                        if empty_frames > max_empty_frames:
                            print("No modified cases frames received after waiting, stopping stream")
                            break
                await asyncio.sleep(0.1)
            except Exception as e:
                print(f"Error in modified cases feed stream: {e}")
                # Try to continue despite errors
                await asyncio.sleep(0.5)
    
    try:
        # Create initial placeholder
        placeholder = np.ones((750, 1200, 3), dtype=np.uint8) * 240
        cv2.putText(placeholder, "Starting modified cases simulation...", (400, 375), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        _, buffer = cv2.imencode(".jpg", placeholder)
        
        # If simulation is not active, return placeholder
        if not modified_cases_active:
            return StreamingResponse(io.BytesIO(buffer.tobytes()), media_type="image/jpeg")
        
        return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")
    except Exception as e:
        print(f"Error setting up modified cases feed: {e}")
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

        # Set traffic_light_active to True before starting the simulation
        traffic_light_active = True
        
        # Start new simulation in background
        background_tasks.add_task(run_traffic_light_simulation)

        return {"message": "Traffic light simulation started"}
    except Exception as e:
        print(f"Error starting simulation: {e}")
        return JSONResponse({"error": f"Failed to start simulation: {str(e)}"}, status_code=500)

@app.post("/start_modified_cases")
async def start_modified_cases(background_tasks: BackgroundTasks):
    try:
        global modified_cases_active, modified_cases_frame, modified_cases_process
        
        # Stop any existing simulation
        if modified_cases_active and modified_cases_process is not None:
            try:
                modified_cases_active = False
                modified_cases_process.terminate()
                modified_cases_process.wait(timeout=5)
            except Exception as e:
                print(f"Error stopping existing modified cases process: {e}")
        
        time.sleep(1)
        
        # Create an initial frame to show while starting
        canvas = np.ones((750, 1200, 3), dtype=np.uint8) * 255
        cv2.putText(canvas, "Starting modified cases simulation...", (400, 375), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        
        with modified_cases_lock:
            _, buffer = cv2.imencode(".jpg", canvas)
            modified_cases_frame = buffer.tobytes()
        
        # Start new simulation in background
        background_tasks.add_task(run_modified_cases_simulation)
        
        return {"message": "Modified cases simulation started"}
    except Exception as e:
        print(f"Error starting modified cases: {e}")
        return JSONResponse({"error": f"Failed to start modified cases: {str(e)}"}, status_code=500)

@app.get("/update_roi/{direction}")
async def update_roi(direction: str, x1: float, y1: float, x2: float, y2: float):
    if direction not in counting_regions:
        return JSONResponse({"error": f"Invalid direction: {direction}"}, status_code=400)
    
    counting_regions[direction] = {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
    return {"message": f"ROI for {direction} updated successfully", "roi": counting_regions[direction]}

@app.post("/save_model")
async def save_dqn_model():
    try:
        save_model(dqn_agent)
        return {"message": "Model saved successfully"}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi import Request

templates = Jinja2Templates(directory="templates")

@app.get("/dqn_graphs", response_class=HTMLResponse)
def show_dqn_graphs(request: Request):
    return templates.TemplateResponse("dqn_graphs.html", {"request": request})


@app.get("/dqn_stats")
def get_dqn_stats():
    return {
        "reward_history": reward_history,
        "action_counts": action_counts,
        "epsilon_history": epsilon_history,
        "congestion_history": congestion_history
    }


@app.get("/vehicle_counts")
async def get_vehicle_counts():
    with frame_lock:
        return JSONResponse({
            "counts": vehicle_counts,
            "epsilon": round(dqn_agent.epsilon, 3)  # show exploration level
        })
    
@app.post("/stop_modified_cases")
async def stop_modified_cases():
    """Stop the modified cases simulation"""
    global modified_cases_active, modified_cases_process
    
    try:
        if modified_cases_active and modified_cases_process is not None:
            try:
                modified_cases_active = False
                modified_cases_process.terminate()
                modified_cases_process.wait(timeout=5)
            except Exception as e:
                print(f"Error stopping modified cases process: {e}")
                # Force kill if needed
                try:
                    modified_cases_process.kill()
                except:
                    pass
            
            modified_cases_process = None
            
            # Create a final frame
            canvas = np.ones((750, 1200, 3), dtype=np.uint8) * 255
            cv2.putText(canvas, "Modified Cases Simulation Stopped", (400, 375), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            
            with modified_cases_lock:
                _, buffer = cv2.imencode(".jpg", canvas)
                modified_cases_frame = buffer.tobytes()
            
            return {"message": "Simulation stopped successfully"}
        else:
            return {"message": "No simulation is currently running"}
    except Exception as e:
        print(f"Error stopping modified cases: {e}")
        return JSONResponse({"error": f"Failed to stop simulation: {str(e)}"}, status_code=500)    

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)