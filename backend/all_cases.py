import random
import math
import time
import threading
import pygame
import sys
import os
import json
import requests

# Default values of signal times
defaultRed = 150
defaultYellow = 5
defaultGreen = 20
defaultMinimum = 10
defaultMaximum = 60

signals = []
noOfSignals = 4
simTime = 300
timeElapsed = 0

currentGreen = 0  
nextGreen = (currentGreen + 1) % noOfSignals
currentYellow = 0  

# Average times for vehicles to pass the intersection
carTime = 2
bikeTime = 1
rickshawTime = 2.25 
busTime = 2.5
truckTime = 2.5

# Count of cars at a traffic signal
noOfCars = 0
noOfBikes = 0
noOfBuses = 0
noOfTrucks = 0
noOfRickshaws = 0
noOfLanes = 2

# Red signal time at which cars will be detected at a signal
detectionTime = 5

speeds = {'car': 0.5, 'bus': 0.5, 'truck': 0.5, 'rickshaw': 0.5, 'bike': 0.5}

# Coordinates of start
x = {'right': [0, 0, 0], 'down': [710, 680, 650], 'left': [1200, 1200, 1200], 'up': [550, 580, 610]}    
y = {'right': [335, 355, 380], 'down': [0, 0, 0], 'left': [498, 466, 436], 'up': [800, 800, 800]}

vehicles = {'right': {0: [], 1: [], 2: [], 'crossed': 0},
            'down': {0: [], 1: [], 2: [], 'crossed': 0},
            'left': {0: [], 1: [], 2: [], 'crossed': 0},
            'up': {0: [], 1: [], 2: [], 'crossed': 0}}

stopped_vehicles = {'right': 0, 'down': 0, 'left': 0, 'up': 0}  # Count of vehicles stopped at signals
vehicle_counts = {'right': 0, 'down': 0, 'left': 0, 'up': 0}    # Total count of vehicles in each lane

vehicleTypes = {0: 'car', 1: 'bus', 2: 'truck', 3: 'rickshaw', 4: 'bike'}
directionNumbers = {0: 'right', 1: 'down', 2: 'left', 3: 'up'}

# Coordinates of signal image, timer, and vehicle count
signalCoods = [(480, 225), (750, 225), (750, 570), (480, 570)]
signalTimerCoods = [(480, 190), (750, 190), (750, 540), (480, 540)]
vehicleCountCoods = [(440, 190), (810, 190), (810, 540), (440, 540)]
vehicleCountTexts = ["0", "0", "0", "0"]

# Coordinates of stop lines
stopLines = {'right': 540, 'down': 320, 'left': 740, 'up': 520}
defaultStop = {'right': 530, 'down': 310, 'left': 750, 'up': 530}
stops = {'right': [530, 530, 530], 'down': [310, 310, 310], 'left': [750, 750, 750], 'up': [530, 530, 530]}

mid = {'right': {'x': 670, 'y': 420}, 'down': {'x': 690, 'y': 450}, 'left': {'x': 650, 'y': 410}, 'up': {'x': 750, 'y': 420}}
rotationAngle = 3

# Gap between vehicles
gap = 15    # stopping gap
gap2 = 15   # moving gap

# Simulation mode
ALL_CASES_MODE = 2
simulation_mode = ALL_CASES_MODE

# Empty lane configuration for ALL_CASES_MODE
# This will be set to one of the directions
empty_lanes = []  # Will store current empty lanes
max_empty_lanes = 1  # Maximum number of empty lanes at any time
empty_lane_change_interval = 30  # Time in seconds to change empty lanes
signal_skip = True   # Whether to skip signals for empty lanes

# Vehicle generation settings
vehicle_generation_interval = {
    'right': random.randint(2, 5),  # Slower generation (2-5 seconds)
    'down': random.randint(2, 5),
    'left': random.randint(2, 5),
    'up': random.randint(2, 5)
}
last_vehicle_generated = {
    'right': 0,
    'down': 0,
    'left': 0,
    'up': 0
}

# Track which lane is getting preference
preferred_lane = None
preference_reason = ""

pygame.init()
simulation = pygame.sprite.Group()

# Add this function to output simulation data in JSON format
def output_simulation_data():
    global timeElapsed, currentGreen, currentYellow, signals, vehicles, stopped_vehicles, empty_lanes, preferred_lane
    
    # Format the signal status
    signal_status = []
    for i in range(len(signals)):
        state = "red"
        time_remaining = signals[i].red
        
        if i == currentGreen:
            if currentYellow == 1:
                state = "yellow"
                time_remaining = signals[i].yellow
            else:
                state = "green"
                time_remaining = signals[i].green
        
        signal_status.append({
            "id": i,
            "state": state,
            "timeRemaining": time_remaining
        })
    
    # Format vehicle counts
    vehicle_counts = {}
    for direction in directionNumbers.values():
        vehicle_counts[direction] = {
            "car": 0,
            "truck": 0,
            "bus": 0,
            "bike": 0
        }
        
        for lane in range(3):
            for vehicle in vehicles[direction][lane]:
                if vehicle.crossed == 0:
                    vehicle_type = vehicle.vehicleClass
                    if vehicle_type in ["motorcycle", "bicycle"]:
                        vehicle_counts[direction]["bike"] += 1
                    else:
                        vehicle_counts[direction][vehicle_type] += 1
    
    # Create the output data
    output_data = {
        "isRunning": True,
        "elapsedTime": timeElapsed,
        "vehicleCounts": vehicle_counts,
        "signalStatus": signal_status,
        "emptyLanes": empty_lanes,
        "signalSkip": signal_skip,
        "preferredLane": preferred_lane
    }
    
    # Print in a format that can be parsed by the Node.js server
    print(f"SIMULATION_DATA: {json.dumps(output_data)}")
    sys.stdout.flush()

class Button:
    def __init__(self, x, y, width, height, text, color, hover_color):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.color = color
        self.hover_color = hover_color
        self.current_color = color
        self.font = pygame.font.Font(None, 24)
        self.text_surf = self.font.render(text, True, (255, 255, 255))
        self.text_rect = self.text_surf.get_rect(center=self.rect.center)
        
    def draw(self, screen):
        pygame.draw.rect(screen, self.current_color, self.rect, border_radius=5)
        screen.blit(self.text_surf, self.text_rect)
        
    def is_hovered(self, pos):
        if self.rect.collidepoint(pos):
            self.current_color = self.hover_color
            return True
        self.current_color = self.color
        return False
        
    def is_clicked(self, pos, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(pos):
                return True
        return False

class TrafficSignal:
    def __init__(self, red, yellow, green, minimum, maximum):
        self.red = red
        self.yellow = yellow
        self.green = green
        self.minimum = minimum
        self.maximum = maximum
        self.signalText = "30"
        self.totalGreenTime = 0

class Vehicle(pygame.sprite.Sprite):
    def __init__(self, lane, vehicleClass, direction_number, direction, will_turn):
        pygame.sprite.Sprite.__init__(self)
        self.lane = lane
        self.vehicleClass = vehicleClass
        self.speed = speeds[vehicleClass]
        self.direction_number = direction_number
        self.direction = direction
        self.x = x[direction][lane]
        self.y = y[direction][lane]
        self.crossed = 0
        self.willTurn = will_turn
        self.turned = 0
        self.rotateAngle = 0
        vehicles[direction][lane].append(self)
        self.index = len(vehicles[direction][lane]) - 1
        path = "image/" + direction + "/" + vehicleClass + ".png"
        self.originalImage = pygame.image.load(path)
        self.currentImage = pygame.image.load(path)

        if(direction=='right'):
            if(len(vehicles[direction][lane])>1 and vehicles[direction][lane][self.index-1].crossed==0):
                self.stop = vehicles[direction][lane][self.index-1].stop - vehicles[direction][lane][self.index-1].currentImage.get_rect().width - gap
            else:
                self.stop = defaultStop[direction]
            temp = self.currentImage.get_rect().width + gap    
            x[direction][lane] -= temp
            stops[direction][lane] -= temp
        elif(direction=='left'):
            if(len(vehicles[direction][lane])>1 and vehicles[direction][lane][self.index-1].crossed==0):
                self.stop = vehicles[direction][lane][self.index-1].stop + vehicles[direction][lane][self.index-1].currentImage.get_rect().width + gap
            else:
                self.stop = defaultStop[direction]
            temp = self.currentImage.get_rect().width + gap
            x[direction][lane] += temp
            stops[direction][lane] += temp
        elif(direction=='down'):
            if(len(vehicles[direction][lane])>1 and vehicles[direction][lane][self.index-1].crossed==0):
                self.stop = vehicles[direction][lane][self.index-1].stop - vehicles[direction][lane][self.index-1].currentImage.get_rect().height - gap
            else:
                self.stop = defaultStop[direction]
            temp = self.currentImage.get_rect().height + gap
            y[direction][lane] -= temp
            stops[direction][lane] -= temp
        elif(direction=='up'):
            if(len(vehicles[direction][lane])>1 and vehicles[direction][lane][self.index-1].crossed==0):
                self.stop = vehicles[direction][lane][self.index-1].stop + vehicles[direction][lane][self.index-1].currentImage.get_rect().height + gap
            else:
                self.stop = defaultStop[direction]
            temp = self.currentImage.get_rect().height + gap
            y[direction][lane] += temp
            stops[direction][lane] += temp
        simulation.add(self)

    def render(self, screen):
        screen.blit(self.currentImage, (self.x, self.y))

    def move(self):
        if(self.direction=='right'):
            if(self.crossed==0 and self.x+self.currentImage.get_rect().width>stopLines[self.direction]):
                self.crossed = 1
                vehicles[self.direction]['crossed'] += 1
            if(self.willTurn==1):
                if(self.crossed==0 or self.x+self.currentImage.get_rect().width<mid[self.direction]['x']):
                    if((self.x+self.currentImage.get_rect().width<=self.stop or (currentGreen==0 and currentYellow==0) or self.crossed==1) and (self.index==0 or self.x+self.currentImage.get_rect().width<(vehicles[self.direction][self.lane][self.index-1].x - gap2) or vehicles[self.direction][self.lane][self.index-1].turned==1)):                
                        self.x += self.speed
                else:   
                    if(self.turned==0):
                        self.rotateAngle += rotationAngle
                        self.currentImage = pygame.transform.rotate(self.originalImage, -self.rotateAngle)
                        self.x += 2
                        self.y += 1.8
                        if(self.rotateAngle==90):
                            self.turned = 1
                    else:
                        if(self.index==0 or self.y+self.currentImage.get_rect().height<(vehicles[self.direction][self.lane][self.index-1].y - gap2) or self.x+self.currentImage.get_rect().width<(vehicles[self.direction][self.lane][self.index-1].x - gap2)):
                            self.y += self.speed
            else: 
                if((self.x+self.currentImage.get_rect().width<=self.stop or self.crossed == 1 or (currentGreen==0 and currentYellow==0)) and (self.index==0 or self.x+self.currentImage.get_rect().width<(vehicles[self.direction][self.lane][self.index-1].x - gap2) or (vehicles[self.direction][self.lane][self.index-1].turned==1))):                
                    self.x += self.speed

        elif(self.direction=='down'):
            if(self.crossed==0 and self.y+self.currentImage.get_rect().height>stopLines[self.direction]):
                self.crossed = 1
                vehicles[self.direction]['crossed'] += 1
            if(self.willTurn==1):
                if(self.crossed==0 or self.y+self.currentImage.get_rect().height<mid[self.direction]['y']):
                    if((self.y+self.currentImage.get_rect().height<=self.stop or (currentGreen==1 and currentYellow==0) or self.crossed==1) and (self.index==0 or self.y+self.currentImage.get_rect().height<(vehicles[self.direction][self.lane][self.index-1].y - gap2) or vehicles[self.direction][self.lane][self.index-1].turned==1)):                
                        self.y += self.speed
                else:   
                    if(self.turned==0):
                        self.rotateAngle += rotationAngle
                        self.currentImage = pygame.transform.rotate(self.originalImage, -self.rotateAngle)
                        self.x -= 2.5
                        self.y += 2
                        if(self.rotateAngle==90):
                            self.turned = 1
                    else:
                        if(self.index==0 or self.x>(vehicles[self.direction][self.lane][self.index-1].x + vehicles[self.direction][self.lane][self.index-1].currentImage.get_rect().width + gap2) or self.y<(vehicles[self.direction][self.lane][self.index-1].y - gap2)):
                            self.x -= self.speed
            else: 
                if((self.y+self.currentImage.get_rect().height<=self.stop or self.crossed == 1 or (currentGreen==1 and currentYellow==0)) and (self.index==0 or self.y+self.currentImage.get_rect().height<(vehicles[self.direction][self.lane][self.index-1].y - gap2) or (vehicles[self.direction][self.lane][self.index-1].turned==1))):                
                    self.y += self.speed
            
        elif(self.direction=='left'):
            if(self.crossed==0 and self.x<stopLines[self.direction]):
                self.crossed = 1
                vehicles[self.direction]['crossed'] += 1
            if(self.willTurn==1):
                if(self.crossed==0 or self.x>mid[self.direction]['x']):
                    if((self.x>=self.stop or (currentGreen==2 and currentYellow==0) or self.crossed==1) and (self.index==0 or self.x>(vehicles[self.direction][self.lane][self.index-1].x + vehicles[self.direction][self.lane][self.index-1].currentImage.get_rect().width + gap2) or vehicles[self.direction][self.lane][self.index-1].turned==1)):                
                        self.x -= self.speed
                else: 
                    if(self.turned==0):
                        self.rotateAngle += rotationAngle
                        self.currentImage = pygame.transform.rotate(self.originalImage, -self.rotateAngle)
                        self.x -= 1.8
                        self.y -= 2.5
                        if(self.rotateAngle==90):
                            self.turned = 1
                    else:
                        if(self.index==0 or self.y>(vehicles[self.direction][self.lane][self.index-1].y + vehicles[self.direction][self.lane][self.index-1].currentImage.get_rect().height +  gap2) or self.x>(vehicles[self.direction][self.lane][self.index-1].x + gap2)):
                            self.y -= self.speed
            else: 
                if((self.x>=self.stop or self.crossed == 1 or (currentGreen==2 and currentYellow==0)) and (self.index==0 or self.x>(vehicles[self.direction][self.lane][self.index-1].x + vehicles[self.direction][self.lane][self.index-1].currentImage.get_rect().width + gap2) or (vehicles[self.direction][self.lane][self.index-1].turned==1))):                
                    self.x -= self.speed
                    
        elif(self.direction=='up'):
            if(self.crossed==0 and self.y<stopLines[self.direction]):
                self.crossed = 1
                vehicles[self.direction]['crossed'] += 1
            if(self.willTurn==1):
                if(self.crossed==0 or self.y>mid[self.direction]['y']):
                    if((self.y>=self.stop or (currentGreen==3 and currentYellow==0) or self.crossed == 1) and (self.index==0 or self.y>(vehicles[self.direction][self.lane][self.index-1].y + vehicles[self.direction][self.lane][self.index-1].currentImage.get_rect().height +  gap2) or vehicles[self.direction][self.lane][self.index-1].turned==1)):
                        self.y -= self.speed
                else:   
                    if(self.turned==0):
                        self.rotateAngle += rotationAngle
                        self.currentImage = pygame.transform.rotate(self.originalImage, -self.rotateAngle)
                        self.x += 1
                        self.y -= 1
                        if(self.rotateAngle==90):
                            self.turned = 1
                    else:
                        if(self.index==0 or self.x<(vehicles[self.direction][self.lane][self.index-1].x - vehicles[self.direction][self.lane][self.index-1].currentImage.get_rect().width - gap2) or self.y>(vehicles[self.direction][self.lane][self.index-1].y + gap2)):
                            self.x += self.speed
            else: 
                if((self.y>=self.stop or self.crossed == 1 or (currentGreen==3 and currentYellow==0)) and (self.index==0 or self.y>(vehicles[self.direction][self.lane][self.index-1].y + vehicles[self.direction][self.lane][self.index-1].currentImage.get_rect().height + gap2) or (vehicles[self.direction][self.lane][self.index-1].turned==1))):                
                    self.y -= self.speed

# Reset simulation state
def reset_simulation():
    global signals, vehicles, currentGreen, nextGreen, currentYellow, timeElapsed, simulation, empty_lanes
    
    # Clear signals
    signals = []
    
    # Reset vehicles
    for direction in vehicles:
        for lane in range(3):
            vehicles[direction][lane] = []
        vehicles[direction]['crossed'] = 0
    
    # Reset simulation sprites
    simulation.empty()
    
    # Reset signal states
    currentGreen = 0
    nextGreen = (currentGreen + 1) % noOfSignals
    currentYellow = 0
    
    # Reset time
    timeElapsed = 0
    
    # Initialize empty lanes
    update_empty_lanes()
    
    # Initialize signals
    initialize()

# Update which lanes are empty - now properly checks for vehicles
def update_empty_lanes():
    global empty_lanes, vehicle_counts
    
    # Count vehicles in each lane
    count_vehicles_in_lanes()
    count_stopped_vehicles()
    
    # Clear current empty lanes
    empty_lanes = []
    
    # Only consider a lane empty if it has no vehicles at all
    for direction in directionNumbers.values():
        if vehicle_counts[direction] == 0 and stopped_vehicles[direction] == 0:
            empty_lanes.append(direction)
    
    # If we have more empty lanes than allowed, randomly select some
    if len(empty_lanes) > max_empty_lanes:
        empty_lanes = random.sample(empty_lanes, max_empty_lanes)
    
    print(f"Updated empty lanes: {empty_lanes}")

# Count vehicles in each lane
def count_vehicles_in_lanes():
    global vehicle_counts
    
    # Reset counts
    vehicle_counts = {key: 0 for key in vehicle_counts}
    
    # Count vehicles that haven't crossed yet
    for direction in directionNumbers.values():
        for lane in range(3):
            for vehicle in vehicles[direction][lane]:
                if vehicle.crossed == 0:
                    vehicle_counts[direction] += 1

# Count vehicles stopped at each signal
def count_stopped_vehicles():
    global stopped_vehicles
    stopped_vehicles = {key: 0 for key in stopped_vehicles}
    
    for direction in directionNumbers.values():
        for lane in range(3):
            for vehicle in vehicles[direction][lane]:
                if vehicle.crossed == 0:
                    stopped_vehicles[direction] += 1

# Thread to periodically change which lanes are empty
def empty_lane_changer():
    while True:
        try:
            time.sleep(empty_lane_change_interval)
            update_empty_lanes()
        except Exception as e:
            print(f"Error in empty_lane_changer: {e}")
            time.sleep(5)  # Wait a bit before trying again

# Initialization of signals with default values
def initialize():
    global signals
    signals = []  # Clear existing signals
    
    ts1 = TrafficSignal(0, defaultYellow, defaultGreen, defaultMinimum, defaultMaximum)
    signals.append(ts1)
    ts2 = TrafficSignal(ts1.red+ts1.yellow+ts1.green, defaultYellow, defaultGreen, defaultMinimum, defaultMaximum)
    signals.append(ts2)
    ts3 = TrafficSignal(defaultRed, defaultYellow, defaultGreen, defaultMinimum, defaultMaximum)
    signals.append(ts3)
    ts4 = TrafficSignal(defaultRed, defaultYellow, defaultGreen, defaultMinimum, defaultMaximum)
    signals.append(ts4)

# Find the lane with the most vehicles
def find_lane_with_most_vehicles():
    global vehicle_counts, empty_lanes
    
    # Update vehicle counts
    count_vehicles_in_lanes()
    
    max_count = -1
    max_lane = None
    
    for direction, count in vehicle_counts.items():
        # Skip empty lanes
        if direction in empty_lanes:
            continue
        
        if count > max_count:
            max_count = count
            max_lane = direction
    
    return max_lane, max_count

# Set time according to formula and vehicle count
def setTime():
    global noOfCars, noOfBikes, noOfBuses, noOfTrucks, noOfRickshaws, noOfLanes
    global carTime, busTime, truckTime, rickshawTime, bikeTime
    global nextGreen, currentGreen, empty_lanes, preferred_lane, preference_reason
    
    # Get direction for the next green signal
    next_direction = directionNumbers[nextGreen]
    
    # If the next lane is an empty lane and we're skipping signals for empty lanes
    if next_direction in empty_lanes and signal_skip:
        # Set green time to minimum for empty lanes
        signals[nextGreen].green = defaultMinimum
        print(f'Green Time for {next_direction}: {defaultMinimum} (Empty Lane)')
        return
    
    # Count actual vehicles in the lane
    noOfCars = noOfBuses = noOfTrucks = noOfRickshaws = noOfBikes = 0
    
    # Count vehicles in each lane
    for lane in range(3):
        for vehicle in vehicles[next_direction][lane]:
            if vehicle.crossed == 0:  # Only count vehicles that haven't crossed yet
                if vehicle.vehicleClass == 'car':
                    noOfCars += 1
                elif vehicle.vehicleClass == 'bus':
                    noOfBuses += 1
                elif vehicle.vehicleClass == 'truck':
                    noOfTrucks += 1
                elif vehicle.vehicleClass == 'rickshaw':
                    noOfRickshaws += 1
                elif vehicle.vehicleClass == 'bike':
                    noOfBikes += 1
    
    # Calculate green time based on vehicle count
    greenTime = math.ceil(((noOfCars*carTime) + (noOfRickshaws*rickshawTime) + (noOfBuses*busTime) + (noOfTrucks*truckTime)+ (noOfBikes*bikeTime))/(noOfLanes+1))
    
    # Adjust green time based on vehicle count
    total_vehicles = noOfCars + noOfBuses + noOfTrucks + noOfRickshaws + noOfBikes
    
    # Scale green time based on vehicle count (more vehicles = more time)
    if total_vehicles > 0:
        # Add bonus time for lanes with more vehicles
        bonus_time = min(20, total_vehicles * 2)  # Cap bonus time at 20 seconds
        greenTime += bonus_time
        
        # Set this as the preferred lane if it has the most vehicles
        most_vehicles_lane, most_count = find_lane_with_most_vehicles()
        if next_direction == most_vehicles_lane:
            preferred_lane = next_direction
            preference_reason = f"Most vehicles: {total_vehicles}"
    
    if(greenTime < defaultMinimum):
        greenTime = defaultMinimum
    elif(greenTime > defaultMaximum):
        greenTime = defaultMaximum
    
    signals[nextGreen].green = greenTime

def repeat():
    global currentGreen, currentYellow, nextGreen, empty_lanes, signal_skip, preferred_lane, preference_reason
    
    # Check if current lane is an empty lane
    current_direction = directionNumbers[currentGreen]
    if current_direction in empty_lanes and signal_skip:
        # Skip or reduce green time for empty lanes
        signals[currentGreen].green = min(signals[currentGreen].green, defaultMinimum)
        preferred_lane = None  # No preference for empty lanes
    else:
        # Check if this lane should get preference (has most vehicles)
        most_vehicles_lane, most_count = find_lane_with_most_vehicles()
        if current_direction == most_vehicles_lane:
            # This lane has the most vehicles, give it preference
            preferred_lane = current_direction
            preference_reason = f"Most vehicles: {most_count}"
            
            # Increase green time for preferred lane
            if signals[currentGreen].green < defaultMaximum:
                bonus_time = min(10, most_count)  # Add up to 10 seconds based on vehicle count
                signals[currentGreen].green += bonus_time
        else:
            preferred_lane = None
    
    while(signals[currentGreen].green > 0):
        updateValues()
        if(signals[(currentGreen+1)%(noOfSignals)].red == detectionTime):
            thread = threading.Thread(name="detection", target=setTime, args=())
            thread.daemon = True
            thread.start()
        time.sleep(1)
        
        # If current lane is empty and we're skipping signals, end green time early
        if current_direction in empty_lanes and signal_skip and signals[currentGreen].green > defaultMinimum:
            signals[currentGreen].green = 0
            break
    
    currentYellow = 1
    vehicleCountTexts[currentGreen] = "0"
    for i in range(0,3):
        stops[directionNumbers[currentGreen]][i] = defaultStop[directionNumbers[currentGreen]]
        for vehicle in vehicles[directionNumbers[currentGreen]][i]:
            vehicle.stop = defaultStop[directionNumbers[currentGreen]]
    
    # Check if current lane is empty
    if current_direction in empty_lanes and signal_skip:
        # Skip yellow time for empty lanes
        signals[currentGreen].yellow = 0
    
    while(signals[currentGreen].yellow > 0):
        updateValues()
        time.sleep(1)
    
    currentYellow = 0
    
    signals[currentGreen].green = defaultGreen
    signals[currentGreen].yellow = defaultYellow
    signals[currentGreen].red = defaultRed
    
    # Find next lane with preference for lanes with vehicles
    if signal_skip:
        # Update vehicle counts
        count_vehicles_in_lanes()
        
        # Find lane with most vehicles
        most_vehicles_lane, most_count = find_lane_with_most_vehicles()
        
        if most_vehicles_lane and most_count > 0:
            # Give preference to the lane with most vehicles
            for i, direction in enumerate(directionNumbers.values()):
                if direction == most_vehicles_lane:
                    nextGreen = i
                    preferred_lane = most_vehicles_lane
                    preference_reason = f"Most vehicles: {most_count}"
                    break
        else:
            # If no lane has vehicles or all lanes are empty, use normal rotation
            nextGreen = (currentGreen + 1) % noOfSignals
    else:
        # Normal rotation
        nextGreen = (currentGreen + 1) % noOfSignals
    
    currentGreen = nextGreen
    nextGreen = (currentGreen + 1) % noOfSignals
    signals[nextGreen].red = signals[currentGreen].yellow + signals[currentGreen].green
    repeat()

# Update values of the signal timers after every second
def updateValues():
    for i in range(0, noOfSignals):
        if(i==currentGreen):
            if(currentYellow==0):
                signals[i].green-=1
                signals[i].totalGreenTime+=1
            else:
                signals[i].yellow-=1
        else:
            signals[i].red-=1

# Generating vehicles in the simulation
def generateVehicles():
    global last_vehicle_generated, vehicle_generation_interval
    
    while(True):
        try:
            current_time = timeElapsed
            
            # Check each direction
            for direction_name in directionNumbers.values():
                # Skip vehicle generation for empty lanes
                if direction_name in empty_lanes:
                    continue
                    
                # Check if it's time to generate a vehicle for this direction
                if current_time - last_vehicle_generated[direction_name] >= vehicle_generation_interval[direction_name]:
                    # Map direction name to direction number
                    direction_number = {"right": 0, "down": 1, "left": 2, "up": 3}.get(direction_name, 0)
                    
                    # Generate random vehicles with different probabilities
                    vehicle_types = ['car', 'bus', 'truck', 'bike']
                    vehicle_probabilities = [0.5, 0.15, 0.15, 0.2]  # Higher probability for cars
                    
                    # Select a random vehicle type based on probabilities
                    vehicle_type = random.choices(vehicle_types, vehicle_probabilities)[0]
                    
                    # Determine lane number (bikes in lane 0, others in lanes 1-2)
                    lane_number = 0 if vehicle_type == "bike" else random.randint(1, 2)
                    
                    # Determine if vehicle will turn (only for lane 2)
                    will_turn = 0
                    if lane_number == 2:
                        will_turn = 1 if random.random() < 0.3 else 0  # 30% chance to turn
                    
                    # Create the vehicle in the simulation
                    Vehicle(lane_number, vehicle_type, direction_number, direction_name, will_turn)
                    
                    # Update last generation time and set new interval
                    last_vehicle_generated[direction_name] = current_time
                    # Randomize next generation interval to create natural traffic patterns
                    vehicle_generation_interval[direction_name] = random.randint(2, 5)  # Slower generation (2-5 seconds)
            
            # Sleep to avoid creating too many vehicles at once
            time.sleep(1.0)  # Increased sleep time for slower generation
        except Exception as e:
            print(f"Error in generateVehicles: {e}")
            time.sleep(1.0)

def simulationTime():
    global timeElapsed, simTime
    while(True):
        timeElapsed += 1
        time.sleep(1)
        if(timeElapsed==simTime):
            totalVehicles = 0
            print('Lane-wise Vehicle Counts')
            for i in range(noOfSignals):
                print('Lane',i+1,':',vehicles[directionNumbers[i]]['crossed'])
                totalVehicles += vehicles[directionNumbers[i]]['crossed']
            print('Total vehicles passed: ',totalVehicles)
            print('Total time passed: ',timeElapsed)
            print('No. of vehicles passed per unit time: ',(float(totalVehicles)/float(timeElapsed)))
            os._exit(1)

def displayStoppedVehicles(screen, font):
    """Function to display the count of vehicles stopped at each signal"""
    for i, direction in enumerate(directionNumbers.values()):
        # For empty lane, show "EMPTY LANE"
        if direction in empty_lanes:
            stoppedText = font.render("EMPTY LANE", True, (255, 0, 0), (0, 0, 0))
        else:
            stoppedText = font.render(f"Stopped: {stopped_vehicles[direction]}", True, (255, 255, 255), (0, 0, 0))
        screen.blit(stoppedText, (vehicleCountCoods[i][0], vehicleCountCoods[i][1] + 30))

def main():
    global empty_lanes, signal_skip, preferred_lane, timeElapsed
    
    # Initialize empty lanes
    update_empty_lanes()
    
    # Colours 
    black = (0, 0, 0)
    white = (255, 255, 255)
    green = (0, 200, 0)
    yellow = (200, 200, 0)
    red = (200, 0, 0)
    blue = (0, 0, 200)

    # Screensize 
    screenWidth = 1280
    screenHeight = 800
    screenSize = (screenWidth, screenHeight)

    # Setting background image i.e. image of intersection
    try:
        background = pygame.image.load('image/background.jpg')
    except pygame.error as e:
        print(f"Error loading background image: {e}")
        # Create a fallback background
        background = pygame.Surface(screenSize)
        background.fill((200, 200, 200))

    screen = pygame.display.set_mode(screenSize)
    pygame.display.set_caption("TRAFFIC SIMULATION")

    # Loading signal images and font with error handling
    try:
        redSignal = pygame.image.load('image/signals/red.png')
        yellowSignal = pygame.image.load('image/signals/yellow.png')
        greenSignal = pygame.image.load('image/signals/green.png')
    except pygame.error as e:
        print(f"Error loading signal images: {e}")
        # Create fallback signal images
        redSignal = pygame.Surface((30, 30))
        redSignal.fill(red)
        yellowSignal = pygame.Surface((30, 30))
        yellowSignal.fill(yellow)
        greenSignal = pygame.Surface((30, 30))
        greenSignal.fill(green)
    
    font = pygame.font.Font(None, 30)
    
    # Create toggle button for simulation mode
    #toggle_button = Button(900, 200, 150, 30, "Toggle Signal Skip", (50, 50, 200), (100, 100, 250))
    
    # State variables
    simulation_running = False
    simulation_threads = []
    
    try:
        # Reset simulation state
        reset_simulation()
        simulation_running = True
        
        # Start simulation threads
        thread_sim_time = threading.Thread(name="simulationTime", target=simulationTime, args=())
        thread_sim_time.daemon = True
        thread_sim_time.start()
        simulation_threads.append(thread_sim_time)
        
        thread_init = threading.Thread(name="initialization", target=repeat, args=())
        thread_init.daemon = True
        thread_init.start()
        simulation_threads.append(thread_init)
        
        thread_gen_vehicles = threading.Thread(name="generateVehicles", target=generateVehicles, args=())
        thread_gen_vehicles.daemon = True
        thread_gen_vehicles.start()
        simulation_threads.append(thread_gen_vehicles)
        
        # Thread to change empty lanes periodically
        thread_empty_lane = threading.Thread(name="emptyLaneChanger", target=empty_lane_changer, args=())
        thread_empty_lane.daemon = True
        thread_empty_lane.start()
        simulation_threads.append(thread_empty_lane)
        
        while True:
            try:
                mouse_pos = pygame.mouse.get_pos()
                
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        # Clean up any running threads
                        pygame.quit()
                        sys.exit()
                    
                    # Handle button click
               #     if toggle_button.is_clicked(mouse_pos, event):
                  #      signal_skip = not signal_skip
                   #     print(f"Signal skip toggled to: {signal_skip}")

                # Check if button is hovered
             #   toggle_button.is_hovered(mouse_pos)

                screen.blit(background, (0, 0))
                
                # Draw the toggle button
             #   toggle_button.draw(screen)
                
                # Update empty lanes based on actual vehicle presence
                if timeElapsed % 5 == 0:  # Check every 5 seconds
                    update_empty_lanes()
                
                # Display signals
                for i in range(0, noOfSignals):
                    direction = directionNumbers[i]
                    
                    # Highlight preferred lane with a black border
                    if direction == preferred_lane:
                        # Draw a highlight rectangle around the signal
                        highlight_rect = pygame.Rect(signalCoods[i][0]-5, signalCoods[i][1]-5, 40, 40)
                        pygame.draw.rect(screen, black, highlight_rect, 3)
                    
                    # For empty lane, always show red if signal_skip is True
                    if direction in empty_lanes and signal_skip:
                        signals[i].signalText = "EMPTY"
                        screen.blit(redSignal, signalCoods[i])
                    else:
                        if(i==currentGreen):
                            if(currentYellow==1):
                                if(signals[i].yellow==0):
                                    signals[i].signalText = "STOP"
                                else:
                                    signals[i].signalText = signals[i].yellow
                                screen.blit(yellowSignal, signalCoods[i])
                            else:
                                if(signals[i].green==0):
                                    signals[i].signalText = "SLOW"
                                else:
                                    signals[i].signalText = signals[i].green
                                screen.blit(greenSignal, signalCoods[i])
                        else:
                            if(signals[i].red<=10):
                                if(signals[i].red==0):
                                    signals[i].signalText = "GO"
                                else:
                                    signals[i].signalText = signals[i].red
                            else:
                                signals[i].signalText = "---"
                            screen.blit(redSignal, signalCoods[i])
                
                signalTexts = ["", "", "", ""]

                # Display signal timer and vehicle count
                for i in range(0, noOfSignals):
                    direction = directionNumbers[i]
                    
                    # Use different colors for preferred lane
                    text_color = black if direction == preferred_lane else white
                    
                    signalTexts[i] = font.render(str(signals[i].signalText), True, text_color, black)
                    screen.blit(signalTexts[i], signalTimerCoods[i])
                    
                    # Display vehicle count
                    displayText = vehicles[direction]['crossed']
                    vehicleCountTexts[i] = font.render(str(displayText), True, black, white)
                    screen.blit(vehicleCountTexts[i], vehicleCountCoods[i])
                    
                    # Display current vehicle count in each lane
                    count_vehicles_in_lanes()
                    currentCountText = font.render(f"Current: {vehicle_counts[direction]}", True, 
                                                black if direction == preferred_lane else blue, 
                                                white)
                    screen.blit(currentCountText, (vehicleCountCoods[i][0], vehicleCountCoods[i][1] + 60))

                # Display time elapsed
                timeElapsedText = font.render(("Time Elapsed: " + str(timeElapsed)), True, black, white)
                screen.blit(timeElapsedText, (900, 50))
                
                # Display simulation mode
                modeText = font.render("Traffic Simulation", True, black, white)
                screen.blit(modeText, (900, 80))
                
                # Display empty lane information
                emptyLaneText = font.render(f"Empty Lanes: {', '.join(empty_lanes) if empty_lanes else 'None'}", True, black, white)
                screen.blit(emptyLaneText, (900, 110))
                
                signalSkipText = font.render(f"Signal Skip: {'Yes' if signal_skip else 'No'}", True, black, white)
                screen.blit(signalSkipText, (900, 140))
                
                # Display preferred lane information
                if preferred_lane:
                    preferredLaneText = font.render(f"Preferred Lane: {preferred_lane}", True, black, white)
                    screen.blit(preferredLaneText, (900, 170))

                # Count and display stopped vehicles
                count_stopped_vehicles()
                displayStoppedVehicles(screen, font)

                # Display vehicles
                for vehicle in simulation:
                    screen.blit(vehicle.currentImage, [vehicle.x, vehicle.y])
                    vehicle.move()
                
                pygame.display.update()
                
            except Exception as e:
                print(f"Error in main loop: {e}")
                time.sleep(0.1)  # Prevent CPU overload in case of error
                
    except Exception as e:
        print(f"Critical error in main function: {e}")
    finally:
        # Clean up
        print("Simulation ended")
        pygame.quit()

if __name__ == "__main__":
    main()
