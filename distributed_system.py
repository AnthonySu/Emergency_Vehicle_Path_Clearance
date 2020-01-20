import numpy as np
import matplotlib.pyplot as plt

# basic parameters:
arrival_rate = 0.8
background_speed = 12 # 25mph = 11.176 m/s
cruising_speed = 0.25 * background_speed
segment_length = 1000
vehicle_length = 4.5 # a typical vehicle length is 4.5 meters
pull_over_dist = 3 # The artifical pull over distance is 3 m



def generate_random_seed(arrival_rate, segment_length):

	# Generate possion arrival for left land and right lane as the inter-arrival time is following exponential distribution
	beta = 1 / arrival_rate

	left_interarr_times = np.random.exponential(scale = beta, size = 1000)
	right_interarr_times = np.random.exponential(scale = beta, size = 1000)

	# So the inter-vehicle distances could be calculated as time * background_speed
	left_interarr_dists = left_interarr_times * background_speed
	right_interarr_dists = right_interarr_times * background_speed

	# The physical stamp is the distance between of the fronts of two vehicle
	# To make sure that there is at least a vehicle, we need to filter those distance < 1.1 * vehicle_length and replace with 1.1 * vehicle_length
	left_interarr_dists_modified = np.clip(left_interarr_dists, 1.1 * vehicle_length, np.inf)
	right_interarr_dists_modified = np.clip(right_interarr_dists, 1.1 * vehicle_length, np.inf)

	# Get the coordinates from the intersection of the front of each vehicles, which is the cumulative of distances between vehicles:
	left_vehicle_coordinates = np.cumsum(left_interarr_dists_modified, dtype = float)
	left_vehicle_coordinates = left_vehicle_coordinates[left_vehicle_coordinates < segment_length]
	right_vehicle_coordinates = np.cumsum(right_interarr_dists_modified, dtype = float)
	right_vehicle_coordinates = right_vehicle_coordinates[right_vehicle_coordinates < segment_length]

	return left_vehicle_coordinates, right_vehicle_coordinates



# required length for pull over slot:
required_length = 2 * vehicle_length

# A special rule regarding the distributed system is that all vehicles will try to pull over into their nearest
# slot. However, with respect to time, we might not know which vehicle would pull in first.
# So, we need to calculate that and pick the smallest one to be timestamp of interest. Through the process, vehicles
# to be cleared and empty slots are changing per timestamp of interest




# Utility functions
# We then have to calculate the exact coordinates of those slots, starting from 0:
def get_empty_slots(right_vehicle_coordinates):
    empty_slots = []
    current_position = 0
    for i in range(len(right_vehicle_coordinates)):
        head = current_position
        tail = right_vehicle_coordinates[i]
        empty_slots.append([head, tail])
        current_position = tail + vehicle_length
    if (current_position - vehicle_length) < segment_length :
        empty_slots.append([current_position, segment_length])
    return empty_slots

# Pull over rules for distributed systems:
# 1. a vehicle could only pull over to a spot that is physically ahead of him, which means the head of this slot
# should be smaller than the front of the vehicle;
# 2. the empty slot should at least be 2 * vehicle length long
# 3. Since there is a pull over distance, the back of the vehicle - pull over distance should be smaller than 
# the tail of the slot
def can_pull_over(vehicle_coordinate, slot):
    head = slot[0]
    tail = slot[1]
    vehicle_back = vehicle_coordinate + vehicle_length
    if (tail - head) < required_length:
        return False
    elif vehicle_coordinate - pull_over_dist < head:
        return False
    return True


# First of all, we need to know which slot each vehicle is going to pull over into at timestamp 0:
def slot_to_park_distributed(vehicle_coordinate, empty_slots):
    # default is exiting at the intersection
    res = 0
    
    for slot_element in empty_slots:
        if can_pull_over(vehicle_coordinate, slot_element):
            res = slot_element     
    return res

# Compute the smallest pull over time at certain timestamp:
def next_timestamp(vehicle_coordinates, empty_slots):
    pull_over_times = []
    for vehicle in vehicle_coordinates:
        
        vehicle_back = vehicle + vehicle_length
        tail = slot_to_park_distributed(vehicle, empty_slots)
        if tail != 0:
            tail = tail[1]
            
        
        if tail <= vehicle_back - pull_over_dist:
            pull_over_time = (vehicle_back - tail) / cruising_speed
        else:
            pull_over_time = pull_over_dist / cruising_speed
        pull_over_times.append(pull_over_time)
    
    min_timestamp = np.min(pull_over_times)
    min_positions = list([i for i, x in enumerate(pull_over_times) if x == min_timestamp])
    return min_timestamp, min_positions

# Update the available slots
def update(vehicle_coordinates, min_positions, timestamp, right_vehicle_coordinates):
    # Get the coordinates of the vehicle to pull over
    vehicles_to_pull_over = vehicle_coordinates[min_positions]
    # picture the new coordinates of these vehicles including the old ones by timestamp
    new_coordinates = vehicles_to_pull_over - timestamp * cruising_speed
    # print(new_coordinates)
    # print(right_vehicle_coordinates)
    right_vehicle_coordinates = np.append(right_vehicle_coordinates, new_coordinates)
    right_vehicle_coordinates = np.sort(right_vehicle_coordinates)
    right_vehicle_coordinates = np.clip(right_vehicle_coordinates, 0, np.inf)
    
    # Compute the new available empty slots
    new_empty_slots = get_empty_slots(right_vehicle_coordinates)
    # update the vehicle coordinates: decrement the position and get rid of the pulled over:
    new_vehicle_coordinates = vehicle_coordinates - timestamp * cruising_speed
    new_vehicle_coordinates = np.delete(new_vehicle_coordinates, min_positions)
    new_vehicle_coordinates = np.clip(new_vehicle_coordinates, 0, np.inf)
    
    return new_vehicle_coordinates, new_empty_slots, right_vehicle_coordinates


def timing_process(left_vehicle_coordinates, right_vehicle_coordinates):
    time_counter = 0
    while left_vehicle_coordinates != []:
        # Compute the empty slots on the right lane:
        current_empty_slots = get_empty_slots(right_vehicle_coordinates)
        
        # Calculate the timestamp of the next event:
        min_timestamp, min_positions = next_timestamp(left_vehicle_coordinates, current_empty_slots)
        
        time_counter += min_timestamp
        
        # update:
        left_vehicle_coordinates, empty_slots, right_vehicle_coordinates = update(left_vehicle_coordinates, min_positions, min_timestamp, right_vehicle_coordinates)
        
    return time_counter


time_records = []
arrival_rates = [1.0, 0.8, 0.6, 0.4, 0.2, 0.1, 0.05, 0.02, 0.01]
segment_lengths = [100, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 3000]
for arrival_rate in arrival_rates:
	for i in range(100):
		util = []
		left_vehicle_coordinates, right_vehicle_coordinates = generate_random_seed(arrival_rate, segment_length)
		util.append(timing_process(left_vehicle_coordinates, right_vehicle_coordinates))
	time_records.append(np.average(util))

# for segment_length in segment_lengths:
# 	for i in range(100):
# 		util = []
# 		left_vehicle_coordinates, right_vehicle_coordinates = generate_random_seed(0.8, segment_length)
# 		util.append(timing_process(left_vehicle_coordinates, right_vehicle_coordinates))
# 	time_records.append(np.average(util))

plt.plot(arrival_rates, time_records)
plt.show()
plt.savefig("arrival_rates.png")
