import numpy as np
from pyproj import Proj, transform
import folium
import cv2
import numpy as np
import matplotlib.pyplot as plt
from util import transformation
import math
from matplotlib.pyplot import cm
import csv
import folium

def on_click(event, x, y, p1, p2):
    if event == cv2.EVENT_LBUTTONDOWN:
        print("x: " + str(x) + ", y:" + str(y) + ", p1: " + str(p1) + ", p2: " + str(p2))

# Example usage
csv_file_path = './test/coords_test2.csv'
img = cv2.imread('./test/test2.jpeg')
img = cv2.resize(img, (0,0), fx=0.5, fy=0.5) 

h, w, c = img.shape

# Points x, y, lat, lon, alt, ECEF_x, ECEF_y, ECEF, z
coordinates = []
with open(csv_file_path, mode='r') as file:
    reader = csv.reader(file)
    for row in reader:
        # Convert each row to the appropriate data types and append to coordinates
        coordinates.append([int(row[0]), int(row[1]), float(row[2]), float(row[3]), int(row[4]), float(row[5]), float(row[6]), float(row[7])])

# Define the color for the dots (red in BGR format)
#color = (0, 0, 255)  # Red
color = cm.jet(np.linspace(0, 1, len(coordinates)))
radius = 5
thickness = -1  # Solid circle

# Plot each point as a red dot
for coord, c in zip(coordinates, color):    
    x, y = coord[0], coord[1]
    cv2.circle(img, (x, y), radius, tuple([int(i * 255) for i in c[:3]]), thickness)
    
cv2.imshow("image", img)
cv2.namedWindow('image')
cv2.setMouseCallback('image', on_click)
cv2.waitKey(0)

#Convert and print each coordinate
for coord in coordinates:
    lat, lon, alt = coord[2:5]
    x, y, z = transformation.wgs84_to_ecef(lat, lon, alt)
    print(f"WGS84: ({lat}, {lon}, {alt}) -> ECEF: ({x}, {y}, {z})")

# Convert to numpy arrays for OpenCV
image_points = np.array([[coord[0], coord[1]] for coord in coordinates], dtype=np.float32)
world_points = np.array([[coord[5], coord[6], coord[7]] for coord in coordinates], dtype=np.float32)

dist_coeffs = np.array([-0.02400951, -0.12507687, -0.00467869, -0.00041547, 0.40573053])

initial_camera_matrix = np.array([[1.21569840e+03, 0.00000000e+00, 8.06295583e+02],
                                  [0.00000000e+00, 1.22496651e+03, 5.74704067e+02],
                                  [0.00000000e+00, 0.00000000e+00, 1.0]])

# Convert to numpy arrays for OpenCV
image_points = np.array([[coord[0], coord[1]] for coord in coordinates], dtype=np.float32)
world_points = np.array([[coord[5], coord[6], coord[7]] for coord in coordinates], dtype=np.float32)

# SolvePnP to find rotation and translation vectors
success, rvec, tvec = cv2.solvePnP(world_points, image_points, initial_camera_matrix, dist_coeffs)


# Convert the rotation vector to a rotation matrix
R, _ = cv2.Rodrigues(rvec)

# The camera position in the world coordinate system (ECEF) is the inverse of the translation
# vector and rotation matrix. This means we need to transform the origin of the camera
# coordinate system (which is at [0, 0, 0]) to the world coordinate system.
camera_position_ecef = -np.dot(R.T, tvec)

print("Camera Position (ECEF coordinates):\n", camera_position_ecef.flatten())

[lat, lon, alt] = transformation.ecef_to_wgs84(camera_position_ecef.flatten()[0], camera_position_ecef.flatten()[1], camera_position_ecef.flatten()[2])
print(str(math.degrees(lat)))
print(str(math.degrees(lon)))
print(str(alt))

avg_lat = sum([coord[2] for coord in coordinates]) / len(coordinates)
avg_lon = sum([coord[3] for coord in coordinates]) / len(coordinates)
map = folium.Map(location=[avg_lat, avg_lon], zoom_start=20)

# tile = folium.TileLayer(
#         tiles = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
#         attr = 'Esri',
#         name = 'Esri Satellite',
#         overlay = False,
#         control = True
#        ).add_to(map)

# Add markers to the map
for coord in coordinates:    
    folium.Marker(location=[coord[2], coord[3]]).add_to(map)

# folium.Marker(location=[math.degrees(lat), math.degrees(lon)],  fill_color='red').add_to(map)
folium.CircleMarker(location=[math.degrees(lat), math.degrees(lon)],
                        radius=5,
                        weight=2,
                        color='red',
                        fill=True,
                        fill_color='red').add_to(map)

# Save the map to an HTML file
map.save('./test/map_with_markers.html')