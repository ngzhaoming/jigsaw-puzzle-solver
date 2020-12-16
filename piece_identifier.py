import os
from os.path import join
import cv2
import numpy as np
import matplotlib.pyplot as plt
from side_extractor.py import process_piece
from functools import partial
import traceback # For error tracing

filenames = os.listdir('underwater')
filenames.sort()
filenames = filenames[1:] # Remove .DS_Store file

label_tuples = [('A', 2)]

def create_label(label_tuple):
    letter, max_num = label_tuple
    for i in range(1, max_num + 1):
        label = letter + str(i) if i >= 10 else letter + '0' + str(i)
        yield label
        
labels = []

for label_tuple in label_tuples:
    for label in create_label(label_tuple):
        labels.append(label)

# Possible change
postprocess = partial(cv2.blur, ksize=(3, 3)) # Blurring image with kernel size 3
results = []
error_labels = [] # List of all pieces with unsuccessful detection

for filename, label in zip(filenames, labels):
    img = cv2.imread(join('underwater/', filename))

    # Check side_extractor.py for default params
    out_dict = process_piece(img, after_segmentation_func=postprocess, scale_factor=0.4, 
                             harris_block_size=5, harris_ksize=5,
                             corner_score_threshold=0.2, corner_minmax_threshold=100)

    plt.figure(figsize=(6, 6))
    plt.title("{0} - {1}".format(filename, label)) # "filename - label"
    plt.imshow(out_dict['extracted'], cmap='gray') # Colormap set to grayscale
    plt.scatter(out_dict['xy'][:, 0], out_dict['xy'][:, 1], color='red') # Draw all corners as red dots
    
    plt.show()

    if 'error' in out_dict:
        print label, ':', out_dict['error']
        error_labels.append(label)
        traceback.print_exc()
        continue

    else: # Highlight the 4 edges of each piece
        plt.figure(figsize=(6, 6))
        plt.imshow(out_dict['class_image'])
        plt.show()

to_ignore = [] # Currently there are no pieces to ignore

for el in error_labels:
    labels.remove(el) # Pieces with error have no results

for label, result in zip(labels, results):
    if label in to_ignore:
        continue

    for i, (side_image, io) in enumerate(zip(result['side_images'], result['inout']), start=1):
        out_io = 'int' if io == 'in' else 'out' # out_io is the answer to the piece
        side_image = side_image * 255 # Increase the image by size 255

        out_filename = "{0}_{1}_{2}.jpg".format(label, i, out_io)
        
        # NOTE: Create the folder first so that the cv2 is able to save images into it
        out_path = join('sides/', out_filename)

        cv2.imwrite(out_path, side_image)


"""
Additional comments:

1) plot_grid function removed since it is not used
2) label_tubels is used to classify each piece base on their shapes
3) Partial creates higher order functions - Not all arguments have to be passed as a whole


NOTE: Types of keys

    1) extracted: Bordered binarized image
    2) edges: Canny edge detection
    3) side_images: Consist of 4 arrays representing the sides of each piece
    4) harris: Harris corner detection (coordinates)
    5) segmented: Cropped out binarized image
    6) xy: x and y-coordinates of the corners
    7) class_image**: Highlights the 4 sides of the piece
    8) inout: Determine whether edge is in or out

out_dict = 
{
    'extracted': array([[0, 0, 0, ..., 0, 0, 0],
        [0, 0, 0, ..., 0, 0, 0],
        [0, 0, 0, ..., 0, 0, 0]], dtype=uint8), 

    'edges': array([[0, 0, 0, ..., 0, 0, 0],
        [0, 0, 0, ..., 0, 0, 0],
        [0, 0, 0, ..., 0, 0, 0]], dtype=uint8), 

    'side_images': [array([[0, 1],
        [0, 1],
        [1, 1]], dtype=uint8), 
        array([[0, 0, 0, ..., 0, 0, 0],
        [0, 0, 0, ..., 0, 0, 0],
        [1, 1, 1, ..., 1, 1, 1]], dtype=uint8), 
        array([[1, 1, 1, ..., 1, 1, 0],
        [1, 1, 1, ..., 1, 1, 0],
        [0, 0, 0, ..., 0, 0, 0]], dtype=uint8), 
        array([[1],
        [1],
        [1]], dtype=uint8)], 

    'harris': array([[0., 0., 0., ..., 0., 0., 0.],
        [0., 0., 0., ..., 0., 0., 0.],
        [0., 0., 0., ..., 0., 0., 0.]], dtype=float32), 

    'segmented': array([[0, 0, 0, ..., 0, 0, 0],
        [0, 0, 0, ..., 0, 0, 0],
        [0, 0, 0, ..., 0, 0, 0]], dtype=uint8), 

    'xy': array([[ 268,  428],
        [1538,  428],
        [ 840,  700],
        [ 932,  708],
        [1080,  718],
        [ 708,  732],
        [1065,  880],
        [1078,  882],
        [ 728,  890],
        [1065,  990],
        [ 732,  998],
        [1085, 1140],
        [ 720, 1148],
        [ 942, 1172],
        [ 268, 1378],
        [1538, 1378]]), 

    'class_image': array([[0, 0, 0, ..., 0, 0, 0],
        [0, 0, 0, ..., 0, 0, 0],
        [0, 0, 0, ..., 0, 0, 0]], dtype=uint8), 

    'inout': ['out', 'out', 'out', 'out']
}

"""