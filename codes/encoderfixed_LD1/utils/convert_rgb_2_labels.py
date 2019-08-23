"""
Utils script to convert from RGB labels image to a specific class (r,g,b) -> (k,k,k).
Can be used to map labels from different dataset to the same classes
"""

import os

from PIL import Image

input_folder = '../../../data/CamVid/val_labels_rgb/'
output_folder = '../../../data/CamVid/val_labels/'

# Check if the output folder exists otherwise create it
try:
    os.makedirs(output_folder)
except os.error:
    pass

'''
CamVid RGB legend
64 128 64	Animal
192 0 128	Archway
0 128 192	Bicyclist
0 128 64	Bridge
128 0 0		Building
64 0 128	Car
64 0 192	CartLuggagePram
192 128 64	Child
192 192 128	Column_Pole
64 64 128	Fence
128 0 192	LaneMkgsDriv
192 0 64	LaneMkgsNonDriv
128 128 64	Misc_Text
192 0 192	MotorcycleScooter
128 64 64	OtherMoving
64 192 128	ParkingBlock
64 64 0		Pedestrian
128 64 128	Road
128 128 192	RoadShoulder
0 0 192		Sidewalk
192 128 128	SignSymbol
128 128 128	Sky
64 128 192	SUVPickupTruck
0 0 64		TrafficCone
0 64 64		TrafficLight
192 64 128	Train
128 128 0	Tree
192 128 192	Truck_Bus
64 0 64		Tunnel
192 192 0	VegetationMisc
0 0 0		Void
64 192 0	Wall
'''

'''
VOC2012 ground truth map:
0 = background -> [0, 0, 0]
1 = aeroplane -> [128, 0, 0]      
2 = bicycle -> [0, 128, 0]
3 = bird ->[128, 128, 0]
4 = boat ->[0, 0, 128]
5 = bottle ->[128, 0, 128]
6 = bus ->[0, 128, 128]
7 = car ->[128, 128, 128]
8 = cat ->[64, 0, 0]
9 = chair ->[192, 0, 0]
10 = cow ->[64, 128, 0]
11 = dining table ->[192, 128, 0]
12 = dog ->[64, 0, 128]
13 = horse ->[192, 0, 128]
14 = motorbike ->[64, 128, 128]
15 = person ->[192, 128, 128]
16 = potted plant ->[0, 64, 0]
17 = sheep ->[128, 64, 0]
18 = sofa ->[0, 192, 0]
19 = train ->[128, 192, 0]
20 = tv/monitor ->[0, 64, 128]
'''

# Create the desired map. All the value that are not contained in it will be mapped to (255, 255, 255)
# Format (r,g,b):(k,k,k) where k is the desired label corresponding to the class
rgb_map = {
    (128, 64, 64): (2, 2, 2),  # OtherMoving -> Bycicle
    (64, 0, 128): (7, 7, 7),  # Car -> Car
    (64, 128, 192): (7, 7, 7),  # SUVPickupTruck -> Car
    (192, 0, 192): (14, 14, 14),  # MotorcycleScooter -> Motorbike
    (192, 128, 64): (15, 15, 15),  # child -> person
    (64, 64, 0): (15, 15, 15),  # pedestrian -> person
    (192, 64, 128): (19, 19, 19),  # Train -> Train
}

k = 0
files = os.listdir(input_folder)
for file in files:
    img = Image.open(input_folder + file)
    pixels = img.load()
    k += 1

    # PIL accesses images in Cartesian co-ordinates, so it is Image[columns, rows]
    out_img = Image.new('RGB', img.size, "black")  # create a new black image
    out_pixels = out_img.load()  # create the pixel map

    for i in range(img.size[0]):  # for every col:
        for j in range(img.size[1]):  # For every row
            # print(pixels[i, j])
            new_pixel = rgb_map.get(pixels[i, j])
            if new_pixel is not None:
                out_pixels[i, j] = new_pixel  # set the colour accordingly to the map
            else:
                out_pixels[i, j] = (255, 255, 255)

    out_img.save(output_folder + file, 'PNG')
    print('elaborated file {}/{}'.format(k, len(files)))
