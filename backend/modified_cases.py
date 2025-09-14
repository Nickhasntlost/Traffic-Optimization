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
NORMAL_MODE = 0
EMPTY_LANES_MODE = 1
simulation_mode = NORMAL_MODE

# Empty lanes will be dynamically determined based on vehicle count
empty_lanes = []  # This will be updated dynamically

pygame.init()
simulation = pygame.sprite.Group()

# Function to get vehicle counts from the API
def get_vehicle_counts():
    try:
        response = requests.get('http://localhost:8000/vehicle_counts')
        if response.status_code == 200:
            return response.json().get('counts', {})
        else:
            print(f"Error fetching vehicle counts: {response.status_code}")
            return {}
    except Exception as e:
        print(f"Exception fetching vehicle counts: {e}")
        return {}

# Add this function to output simulation data in JSON format
def output_simulation_data():
    global timeElapsed, currentGreen, currentYellow, signals, vehicles, stopped_vehicles
    
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
        "signalStatus": signal_status
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
    
    # Reset empty lanes
    empty_lanes = []
    
    # Initialize signals
    initialize()

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

# Determine which lanes are empty and update the empty_lanes list
def update_empty_lanes():
    global empty_lanes
    empty_lanes = []
    
    # Count vehicles in each direction
    for direction in directionNumbers.values():
        total_vehicles = 0
        for lane in range(3):
            total_vehicles += sum(1 for vehicle in vehicles[direction][lane] if vehicle.crossed == 0)
        
        # If no vehicles in this direction, add to empty_lanes
        if total_vehicles == 0:
            empty_lanes.append(direction)
    
    return empty_lanes

# Find the lane with the highest vehicle count
def find_highest_priority_lane():
    lane_counts = {}
    
    # Count vehicles in each direction
    for i, direction in enumerate(directionNumbers.values()):
        if direction in empty_lanes:
            lane_counts[i] = 0
            continue
            
        total_vehicles = 0
        for lane in range(3):
            total_vehicles += sum(1 for vehicle in vehicles[direction][lane] if vehicle.crossed == 0)
        
        lane_counts[i] = total_vehicles
    
    # Find the lane with the highest count
    max_count = -1
    max_lane = 0
    
    for lane, count in lane_counts.items():
        if count > max_count:
            max_count = count
            max_lane = lane
    
    return max_lane

# Set time according to formula and vehicle count
def setTime():
    global noOfCars, noOfBikes, noOfBuses, noOfTrucks, noOfRickshaws, noOfLanes
    global carTime, busTime, truckTime, rickshawTime, bikeTime
    global nextGreen, currentGreen
    
    # Update which lanes are empty
    update_empty_lanes()
    
    # If in EMPTY_LANES_MODE, prioritize lanes with vehicles
    if simulation_mode == EMPTY_LANES_MODE:
        # Find the lane with the highest priority (most vehicles)
        highest_priority = find_highest_priority_lane()
        
        # If the next lane is empty, change to the highest priority lane
        next_direction = directionNumbers[nextGreen]
        if next_direction in empty_lanes and highest_priority != nextGreen:
            nextGreen = highest_priority
            print(f"Changing priority to lane {nextGreen} ({directionNumbers[nextGreen]}) with highest vehicle count")
    
    # Check if the next lane is empty
    next_direction = directionNumbers[nextGreen]
    if next_direction in empty_lanes:
        # Set green time to minimum for empty lanes
        signals[nextGreen].green = defaultMinimum
        print(f'Green Time for {next_direction}: {defaultMinimum} (Empty Lane)')
        return
    
    # Get vehicle counts from API
    vehicle_counts = get_vehicle_counts()
    direction_counts = vehicle_counts.get(next_direction, {})
    
    # Reset counts
    noOfCars = direction_counts.get('car', 0)
    noOfBuses = direction_counts.get('bus', 0)
    noOfTrucks = direction_counts.get('truck', 0)
    noOfRickshaws = direction_counts.get('rickshaw', 0)
    noOfBikes = direction_counts.get('bike', 0)
    
    print(f"Detection counts for {directionNumbers[nextGreen]}: Cars={noOfCars}, Buses={noOfBuses}, Trucks={noOfTrucks}, Rickshaws={noOfRickshaws}, Bikes={noOfBikes}")

    # Calculate green time based on vehicle count
    greenTime = math.ceil(((noOfCars*carTime) + (noOfRickshaws*rickshawTime) + (noOfBuses*busTime) + (noOfTrucks*truckTime)+ (noOfBikes*bikeTime))/(noOfLanes+1))
    
    # Adjust green time based on vehicle count
    total_vehicles = noOfCars + noOfBuses + noOfTrucks + noOfRickshaws + noOfBikes
    
    # Scale green time based on vehicle count (more vehicles = more time)
    if total_vehicles > 0:
        # Add bonus time for lanes with more vehicles
        bonus_time = min(20, total_vehicles * 2)  # Cap bonus time at 20 seconds
        greenTime += bonus_time
    
    print(f'Green Time for {next_direction}: {greenTime} (Vehicles: {total_vehicles})')
    
    if(greenTime < defaultMinimum):
        greenTime = defaultMinimum
    elif(greenTime > defaultMaximum):
        greenTime = defaultMaximum
    
    signals[nextGreen].green = greenTime

def repeat():
    global currentGreen, currentYellow, nextGreen
    
    # Update which lanes are empty
    update_empty_lanes()
    
    # If in EMPTY_LANES_MODE, check if we need to change the next green signal
    if simulation_mode == EMPTY_LANES_MODE:
        # Find the lane with the highest priority (most vehicles)
        highest_priority = find_highest_priority_lane()
        
        # If the current lane is empty and there's a lane with vehicles, change to it
        current_direction = directionNumbers[currentGreen]
        if current_direction in empty_lanes and highest_priority != currentGreen:
            # Skip to the highest priority lane
            signals[currentGreen].green = 0
            print(f"Skipping empty lane {currentGreen} ({current_direction})")
    
    # Check if current lane is empty
    current_direction = directionNumbers[currentGreen]
    if current_direction in empty_lanes:
        # Reduce green time for empty lanes
        signals[currentGreen].green = min(signals[currentGreen].green, defaultMinimum)
        print(f"Reducing green time for empty lane {current_direction}")
    
    while(signals[currentGreen].green > 0):
        printStatus()
        updateValues()
        if(signals[(currentGreen+1)%(noOfSignals)].red == detectionTime):
            thread = threading.Thread(name="detection", target=setTime, args=())
            thread.daemon = True
            thread.start()
        time.sleep(1)
        
        # If in EMPTY_LANES_MODE, continuously check if we need to update priorities
        if simulation_mode == EMPTY_LANES_MODE and signals[currentGreen].green % 5 == 0:  # Check every 5 seconds
            update_empty_lanes()
            highest_priority = find_highest_priority_lane()
            
            # If current lane is empty and there's a lane with vehicles, end green time early
            if current_direction in empty_lanes and highest_priority != currentGreen:
                signals[currentGreen].green = 0
                print(f"Ending green time early for empty lane {current_direction}")
                break
    
    currentYellow = 1
    vehicleCountTexts[currentGreen] = "0"
    for i in range(0,3):
        stops[directionNumbers[currentGreen]][i] = defaultStop[directionNumbers[currentGreen]]
        for vehicle in vehicles[directionNumbers[currentGreen]][i]:
            vehicle.stop = defaultStop[directionNumbers[currentGreen]]
    
    # Check if current lane is empty
    if current_direction in empty_lanes:
        # Skip yellow time for empty lanes
        signals[currentGreen].yellow = 0
    
    while(signals[currentGreen].yellow > 0):
        printStatus()
        updateValues()
        time.sleep(1)
    
    currentYellow = 0
    
    signals[currentGreen].green = defaultGreen
    signals[currentGreen].yellow = defaultYellow
    signals[currentGreen].red = defaultRed
    
    # Update which lanes are empty before selecting next green
    update_empty_lanes()
    
    # If in EMPTY_LANES_MODE, select the lane with highest priority as next green
    if simulation_mode == EMPTY_LANES_MODE:
        highest_priority = find_highest_priority_lane()
        nextGreen = highest_priority
        print(f"Next green set to lane {nextGreen} ({directionNumbers[nextGreen]}) with highest vehicle count")
    else:
        # Normal rotation
        nextGreen = (currentGreen + 1) % noOfSignals
    
    currentGreen = nextGreen
    nextGreen = (currentGreen + 1) % noOfSignals
    signals[nextGreen].red = signals[currentGreen].yellow + signals[currentGreen].green
    repeat()

# Print the signal timers on cmd
def printStatus():                                                                                           
    for i in range(0, noOfSignals):
        if(i==currentGreen):
            if(currentYellow==0):
                print(" GREEN TS",i+1,"-> r:",signals[i].red," y:",signals[i].yellow," g:",signals[i].green)
            else:
                print("YELLOW TS",i+1,"-> r:",signals[i].red," y:",signals[i].yellow," g:",signals[i].green)
        else:
            print("   RED TS",i+1,"-> r:",signals[i].red," y:",signals[i].yellow," g:",signals[i].green)
    print()

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
    
    # Output simulation data every second
    output_simulation_data()

# Count vehicles stopped at each signal
def countStoppedVehicles():
    global stopped_vehicles
    stopped_vehicles = {key: 0 for key in stopped_vehicles}
    
    for direction in directionNumbers.values():
        for lane in range(3):
            for vehicle in vehicles[direction][lane]:
                if vehicle.crossed == 0:
                    stopped_vehicles[direction] += 1

# Generating vehicles in the simulation
def generateVehicles():
    while(True):
        # Get the latest vehicle counts from API
        vehicle_data = get_vehicle_counts()
        
        # Process each direction
        for direction_name, vehicle_counts in vehicle_data.items():
            # Map direction name to direction number
            direction_number = {"right": 0, "down": 1, "left": 2, "up": 3}.get(direction_name, 0)
            
            # Process each vehicle type
            for vehicle_type, count in vehicle_counts.items():
                # Map vehicle type to vehicle class index
                vehicle_class = vehicle_type  # Use the vehicle type directly
                
                # Create vehicles based on detection count (limit to avoid flooding)
                for _ in range(min(count, 3)):  # Limit to 3 vehicles per type per direction
                    # Determine lane number (bikes in lane 0, others in lanes 1-2)
                    lane_number = 0 if vehicle_type == "bike" else random.randint(1, 2)
                    
                    # Determine if vehicle will turn (only for lane 2)
                    will_turn = 0
                    if lane_number == 2:
                        will_turn = 1 if random.randint(0, 4) <= 2 else 0
                    
                    # Create the vehicle in the simulation
                    Vehicle(lane_number, vehicle_class, direction_number, direction_name, will_turn)
        
        # Sleep to avoid creating too many vehicles at once
        time.sleep(2)

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
        # For empty lanes in EMPTY_LANES_MODE, show "EMPTY" if truly empty
        if direction in empty_lanes:
            stoppedText = font.render("EMPTY LANE", True, (255, 0, 0), (0, 0, 0))
        else:
            stoppedText = font.render(f"Stopped: {stopped_vehicles[direction]}", True, (255, 255, 255), (0, 0, 0))
        screen.blit(stoppedText, (vehicleCountCoods[i][0], vehicleCountCoods[i][1] + 30))

def main():
    global simulation_mode
    
    # Create buttons
    normal_button = Button(50, 50, 200, 40, "Normal Simulation", (0, 128, 0), (0, 200, 0))
    empty_lanes_button = Button(50, 100, 200, 40, "Dynamic Traffic Simulation", (128, 0, 0), (200, 0, 0))
    
    # Colours 
    black = (0, 0, 0)
    white = (255, 255, 255)

    # Screensize 
    screenWidth = 1280
    screenHeight = 800
    screenSize = (screenWidth, screenHeight)

    # Setting background image i.e. image of intersection
    background = pygame.image.load('image/background.jpg')

    screen = pygame.display.set_mode(screenSize)
    pygame.display.set_caption("TRAFFIC SIMULATION WITH REAL DETECTION")

    # Loading signal image and font
    redSignal = pygame.image.load('image/signals/red.png')
    yellowSignal = pygame.image.load('image/signals/yellow.png')
    greenSignal = pygame.image.load('image/signals/green.png')
    font = pygame.font.Font(None, 30)
    
    # State variables
    simulation_running = False
    simulation_threads = []
    
    try:
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    # Clean up any running threads
                    for thread in simulation_threads:
                        if thread.is_alive():
                            thread.join(0)  # Non-blocking join
                    sys.exit()
                    
                # Handle button clicks
                mouse_pos = pygame.mouse.get_pos()
                
                if not simulation_running:
                    if normal_button.is_clicked(mouse_pos, event):
                        simulation_mode = NORMAL_MODE
                        simulation_running = True
                        
                        # Reset simulation state
                        reset_simulation()
                        
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
                        
                    elif empty_lanes_button.is_clicked(mouse_pos, event):
                        simulation_mode = EMPTY_LANES_MODE
                        simulation_running = True
                        
                        # Reset simulation state
                        reset_simulation()
                        
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

            screen.blit(background, (0, 0))
            
            if not simulation_running:
                # Draw buttons when simulation is not running
                normal_button.is_hovered(pygame.mouse.get_pos())
                empty_lanes_button.is_hovered(pygame.mouse.get_pos())
                normal_button.draw(screen)
                empty_lanes_button.draw(screen)
                
                # Draw instructions
                instructions = font.render("Select a simulation mode to begin", True, white, black)
                screen.blit(instructions, (500, 75))
            else:
                # Update empty lanes status
                if simulation_mode == EMPTY_LANES_MODE and timeElapsed % 5 == 0:  # Update every 5 seconds
                    update_empty_lanes()
                
                # Display signals
                for i in range(0, noOfSignals):
                    # For empty lanes in EMPTY_LANES_MODE, show special display
                    if simulation_mode == EMPTY_LANES_MODE and directionNumbers[i] in empty_lanes:
                        signals[i].signalText = "0"
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
                    signalTexts[i] = font.render(str(signals[i].signalText), True, white, black)
                    screen.blit(signalTexts[i], signalTimerCoods[i])
                    
                    # Display vehicle count
                    displayText = vehicles[directionNumbers[i]]['crossed']
                    vehicleCountTexts[i] = font.render(str(displayText), True, black, white)
                    screen.blit(vehicleCountTexts[i], vehicleCountCoods[i])

                # Display time elapsed
                timeElapsedText = font.render(("Time Elapsed: " + str(timeElapsed)), True, black, white)
                screen.blit(timeElapsedText, (900, 50))
                
                # Display simulation mode
                mode_text = "Normal Mode" if simulation_mode == NORMAL_MODE else "Dynamic Traffic Mode"
                modeText = font.render(("Mode: " + mode_text), True, black, white)
                screen.blit(modeText, (900, 80))

                # Count and display stopped vehicles
                countStoppedVehicles()
                displayStoppedVehicles(screen, font)
                
                # Display priority information in EMPTY_LANES_MODE
                if simulation_mode == EMPTY_LANES_MODE:
                    priority_lane = find_highest_priority_lane()
                    priority_text = font.render(f"Priority Lane: {priority_lane+1} ({directionNumbers[priority_lane]})", True, black, white)
                    screen.blit(priority_text, (900, 110))
                    
                    empty_text = font.render(f"Empty Lanes: {', '.join(empty_lanes) if empty_lanes else 'None'}", True, black, white)
                    screen.blit(empty_text, (900, 140))

                # Display vehicles
                for vehicle in simulation:
                    screen.blit(vehicle.currentImage, [vehicle.x, vehicle.y])
                    vehicle.move()
                
            pygame.display.update()
    finally:
        # Clean up
        print("Simulation ended")

if __name__ == "__main__":
    main()
