import math

drone_lat = 38.9210761
drone_lon = 14.2857082
drone_alt = 348.58  

angle_to_target = 35  # Örnek açı

height = drone_alt  
distance = height / math.tan(math.radians(angle_to_target))

R = 6371e3  

delta_lat = (distance / R) * (180 / math.pi)
delta_lon = (distance / R) * (180 / math.pi) / math.cos(math.radians(drone_lat))

target_lat = drone_lat + delta_lat
target_lon = drone_lon + delta_lon

print(f"Hedefin tahmini konumu: Enlem: {target_lat}, Boylam: {target_lon}")
