import cv2
import numpy as np
import scipy.spatial
import scipy.stats
import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters
import matplotlib.pyplot as plt
from functools import partial

####################################################################################################################

# Self-Defined functions

_corner_indexes = [(0, 1), (1, 3), (3, 2), (0, 2)]

#################
# Segment Piece #
#################

# Set bin_threshold to 128 by default
# Higher threshold leads to darker images (Less pixel set to white)
# Apply segmentation of the image by simple binarization
# cv2.threshold returns ret, thresh: 2nd output is the binarized image
def segment_piece(image, bin_threshold=128):
    return cv2.threshold(image, bin_threshold, 255, cv2.THRESH_BINARY)[1]

#################
# Extract Piece #
#################

# Given the binarized image, compute min and max x and y-coordinates
def compute_minmax_xy(thresh):
    idx_shape = np.where(thresh == 0)
    return [np.array([coords.min(), coords.max()]) for coords in idx_shape]

# Takes in the binarized (thresh) img as params
# Build a square image centered on the blob (piece of the puzzle).
# The image is constructed large enough to allow for piece rotations. 
def extract_piece(thresh):
    minmax_y, minmax_x = compute_minmax_xy(thresh)

    ly, lx = minmax_y[1] - minmax_y[0], minmax_x[1] - minmax_x[0]
    size = int(max(ly, lx) * np.sqrt(2))
    
    # Further crop img to only the piece, get the height and width using shape function
    x_extract = thresh[minmax_y[0]:minmax_y[1] + 1, minmax_x[0]:minmax_x[1] + 1]
    ly, lx = x_extract.shape

    x_copy = np.full((size, size), 255, dtype='uint8')
    sy, sx = size // 2 - ly // 2, size // 2 - lx // 2

    x_copy[sy: sy + ly, sx: sx + lx] = x_extract
    thresh = x_copy
    thresh = 255 - thresh
    
    return thresh

###################
# Extract Corners #
###################
    
# Input the Harris img, identify and extract discrete corners
# A1: Harris img, A2: No. of neighboring pixels to consider
def get_corners(dst, neighborhood_size=5, score_threshold=0.3, minmax_threshold=100):
    data = dst.copy()
    data[data < score_threshold * dst.max()] = 0. # Filter out all non-discrete corners

    # Calculate multi-dimensional min/max filter
    data_max = filters.maximum_filter(data, neighborhood_size)
    maxima = (data == data_max)
    data_min = filters.minimum_filter(data, neighborhood_size)
    diff = ((data_max - data_min) > minmax_threshold) # Checks whether diff is above threshold
    maxima[diff == 0] = 0

    # Label the features in an array
    # R1: nD-array where each unique feature in input has a unique label in returned array
    # R2: No. of objects found
    labeled, num_objects = ndimage.label(maxima)
    slices = ndimage.find_objects(labeled)
    yx = np.array(ndimage.center_of_mass(data, labeled, range(1, num_objects+1)))

    return yx[:, ::-1] # Return with the height and width tuple reversed

###################
# Locate best rec #
###################

"""
Since we expect the 4 puzzle corners to be the corners of a rectangle, here we take
all detected Harris corners and we find the best corresponding rectangle.
We perform a recursive search with max depth = 2:
- At depth 0 we take one of the input point as the first corner of the rectangle
- At depth 1 we select another input point (with distance from the first point greater
    then d_threshold) as the second point
- At depth 2 and 3 we take the other points. However, the lines 01-12 and 12-23 should be
    as perpendicular as possible. If the angle formed by these lines is too much far from the
    right angle, we discard the choice.
- At depth 3, if a valid candidate (4 points that form an almost perpendicular rectangle) is found,
    we add it to the list of candidates.
    
Given a list of candidate rectangles, we then select the best one by taking the candidate that maximizes
the function: area * Gaussian(rectangularness)
- area: it is the area of the candidate shape. We expect that the puzzle corners will form the maximum area
- rectangularness: it is the mse of the candidate shape's angles compared to a 90 degree angles. The smaller
                    this value, the most the shape is similar toa rectangle.
"""
def get_best_fitting_rect_coords(xy, d_threshold=30, perp_angle_thresh=20):
    N = len(xy)

    distances = scipy.spatial.distance.cdist(xy, xy) # Return nD-array dist matrix
    distances[distances < d_threshold] = 0
    
    #===== Func 1: Compute angles based on perm of 4 corner points =====
    def compute_angles(xy):
        angles = np.zeros((N, N))

        for i in range(N):
            for j in range(i + 1, N):

                point_i, point_j = xy[i], xy[j]
                if point_i[0] == point_j[0]:
                    angle = 90
                else:
                    angle = np.arctan2(point_j[1] - point_i[1], point_j[0] - point_i[0]) * 180 / np.pi

                angles[i, j] = angle
                angles[j, i] = angle

        return angles

    angles = compute_angles(xy)

    #===== Func 2: Find possible rectangle based on perm of 4 corner points =====
    possible_rectangles = []

    # Recursive function to find all possible rectangles
    def search_for_possible_rectangle(idx, prev_points=[]):
        curr_point = xy[idx]
        depth = len(prev_points)

        if depth == 0:
            right_points_idx = np.nonzero(np.logical_and(xy[:, 0] > curr_point[0], distances[idx] > 0))[0]
            
            for right_point_idx in right_points_idx:
                search_for_possible_rectangle(right_point_idx, [idx])
            
            return

        last_angle = angles[idx, prev_points[-1]]
        perp_angle = last_angle - 90
        if perp_angle < 0:
            perp_angle += 180

        if depth in (1, 2):
            diff0 = np.abs(angles[idx] - perp_angle) <= perp_angle_thresh
            diff180_0 = np.abs(angles[idx] - (perp_angle + 180)) <= perp_angle_thresh
            diff180_1 = np.abs(angles[idx] - (perp_angle - 180)) <= perp_angle_thresh
            all_diffs = np.logical_or(diff0, np.logical_or(diff180_0, diff180_1))
            
            diff_to_explore = np.nonzero(np.logical_and(all_diffs, distances[idx] > 0))[0]

            for dte_idx in diff_to_explore:
                if dte_idx not in prev_points: # unlickly to happen but just to be certain
                    next_points = prev_points[::]
                    next_points.append(idx)

                    search_for_possible_rectangle(dte_idx, next_points)

        if depth == 3:
            angle41 = angles[idx, prev_points[0]]

            diff0 = np.abs(angle41 - perp_angle) <= perp_angle_thresh
            diff180_0 = np.abs(angle41 - (perp_angle + 180)) <= perp_angle_thresh
            diff180_1 = np.abs(angle41 - (perp_angle - 180)) <= perp_angle_thresh
            dist = distances[idx, prev_points[0]] > 0

            if dist and (diff0 or diff180_0 or diff180_1):
                rect_points = prev_points[::]
                rect_points.append(idx)

                already_present = False
                for possible_rectangle in possible_rectangles:
                    if set(possible_rectangle) == set(rect_points):
                        already_present = True
                        break

                if not already_present:
                    possible_rectangles.append(rect_points)

    for i in range(N):
        search_for_possible_rectangle(i)

    if len(possible_rectangles) == 0:
        return None

    #===== Func 3: Compute the area for all possible rec and find the best fit =====
    def PolyArea(x,y):
        return 0.5 * np.abs(np.dot(x,np.roll(y,1)) - np.dot(y,np.roll(x,1)))

    areas = []
    rectangularness = []
    diff_angles = []

    for r in possible_rectangles:
        points = xy[r]
        areas.append(PolyArea(points[:, 0], points[:, 1]))

        mse = 0
        da = []
        for i1, i2, i3 in [(0, 1, 2), (1, 2, 3), (2, 3, 0), (3, 0, 1)]:
            diff_angle = abs(angles[r[i1], r[i2]] - angles[r[i2], r[i3]])
            da.append(abs(diff_angle - 90))
            mse += (diff_angle - 90) ** 2

        diff_angles.append(da)
        rectangularness.append(mse)

    areas = np.array(areas)
    rectangularness = np.array(rectangularness)

    # Continuous normal variable with probability density function
    scores = areas * scipy.stats.norm(0, 150).pdf(rectangularness)
    best_fitting_idxs = possible_rectangles[np.argmax(scores)]
    return xy[best_fitting_idxs]

#################
# Rotate Pieces #
#################

# Rotate an image by the amount specified in degrees
# Returns the rotated image and the rotation matrix (M)
def rotate(image, degrees):
    if len(image.shape) == 3: # Additional channel variable
        rows,cols, _ = image.shape
    else:
        rows, cols = image.shape
        
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), degrees, 1)
    
    return cv2.warpAffine(image,M,(cols,rows)), M

###################
# Finding Corners #
###################

# Given the segmented puzzle piece, compute its barycentre.
def compute_barycentre(thresh, value=0):
    idx_shape = np.where(thresh == value)
    return [int(np.round(coords.mean())) for coords in idx_shape]

# Find corners by taking the highest distant point from a 45 degrees inclined line
# inside a squared ROI centerd on the previously found intersection point.
# Inclination of the line depends on which corner we are looking for, and is
# computed based on the position of the barycenter of the piece.
def corner_detection(edges, intersections, barycenter, rect_size=50, show=False):
    corners = []
    xb, yb = barycenter[0], barycenter[1]

    for idx, intersection in enumerate(intersections):
        xi, yi = intersection

        m = -1 if (yb - yi) * (xb - xi) > 0 else 1
        y0 = 0 if yb < yi else 2*rect_size
        x0 = 0 if xb < xi else 2*rect_size

        a, b, c = m, -1, -m * x0 + y0

        rect = edges[yi - rect_size: yi + rect_size, xi - rect_size: xi + rect_size].copy()
        edge_idx = np.nonzero(rect)
        if len(edge_idx[0]) > 0:
            distances = [(a*edge_x + b*edge_y + c)**2 for edge_y, edge_x in zip(*edge_idx)]
            corner_idx = np.argmax(distances)

            rect_corner = np.array((edge_idx[1][corner_idx], edge_idx[0][corner_idx]))
            offset_corner = rect_corner - rect_size
            real_corner = intersection + offset_corner

            corners.append(real_corner)
        else:
            # If the window is completely black I can make no assumption: I keep the same corner
            corners.append(intersection)

        if show:
            plt.subplot(220 + idx + 1)
            cv2.circle(rect, tuple(rect_corner), 5, 128)
            
            plt.title("{0} | {1}".format(intersection, (x0, y0)))
            plt.imshow(rect)

    if show:
        plt.show()

    return corners

def order_corners(corners):
    corners.sort(key=lambda k: k[0] + k[1])
    antidiag_corners = sorted(corners[1:3], key=lambda k: k[1])
    corners[1:3] = antidiag_corners
    return corners

# Given two points p0 (x0, y0) and p1 (x1, y1),
# compute the coefficients (a, b, c) of the line 
# that passes through both points.
def get_line_through_points(p0, p1):
    x0, y0 = p0
    x1, y1 = p1
    
    return y1 - y0, x0 - x1, x1 * y0 - x0 * y1

def compute_line_params(corners):
    return [get_line_through_points(corners[i1], corners[i2]) for i1, i2 in _corner_indexes]

######################
# Generate Class Img #
######################

# Computes the squared distance of a 2D point (x0, y0) from a line ax + by + c = 0
def distance_point_line_squared(eqn, coor):
    a, b, c = eqn[0], eqn[1], eqn[2]
    x0, y0 = coor[0], coor[1]

    return (a * x0 + b * y0 + c)**2 / (a**2 + b**2)

def shape_classification(edges, line_params, d_threshold=500, n_hs=10):
    # First part: we take all edge points and classify them only if their distance to one of the 4 piece
    # lines is smaller than a certain threshold. If that happens, we can be certain that the point belongs
    # to that side of the piece. If each one of the four distances is higher than the threshold, the point
    # will be classified during the second phase.

    y_nonzero, x_nonzero = np.nonzero(edges)
    distances = []

    class_image = np.zeros(edges.shape, dtype='uint8') # Create a black canvas first
    non_classified_points = [] # Classified during the second phase

    for x_edge, y_edge in zip(x_nonzero, y_nonzero):
        d = [distance_point_line_squared(line_param, (x_edge, y_edge)) for line_param in line_params]
        if np.min(d) < d_threshold:
            class_image[y_edge, x_edge] = np.argmin(d) + 1
        else:
            non_classified_points.append((x_edge, y_edge))

    non_classified_points = np.array(non_classified_points)
    
    # Second part: hysteresis classification
    # Edge points that have not been classified because they are too far from all lines
    # will be classified based on their neighborood: if the neighborhood of a point contains
    # an already classified point, it will be classified with the same class.
    # It's very unlikely that the neighborhood of a non classified point will contain two different
    # classes, so we just take the first non-zero value that we find inside the neighborhood
    # The process is repeated and at each iteration the newly classified points are removed from the ones
    # that still need to be classified. The process is interrupted when no new point has been classified
    # or when a maximum number of iterations has been reached (in case of a noisy points that has no neighbours).

    map_iteration = 0
    max_map_iterations = 50

    while map_iteration < max_map_iterations:
        map_iteration += 1
        classified_points_at_current_iteration = []

        for idx, (x_edge, y_edge) in enumerate(non_classified_points):
            neighborhood = class_image[y_edge - n_hs: y_edge + n_hs + 1, x_edge - n_hs: x_edge + n_hs + 1]
            n_mapped = np.nonzero(neighborhood)
            if len(n_mapped[0]) > 0:
                ny, nx = n_mapped[0][0] - n_hs, n_mapped[1][0] - n_hs
                class_image[y_edge, x_edge] = class_image[y_edge + ny, x_edge + nx]
                classified_points_at_current_iteration.append(idx)

        if len(non_classified_points) > 0:
            non_classified_points = np.delete(non_classified_points, classified_points_at_current_iteration, axis=0)
        else:
            break
    
    return class_image

# Computes the signed distance of a 2D point (x0, y0) from a line ax + by + c = 0
def distance_point_line_signed(eqn, coor):
    a, b, c = eqn[0], eqn[1], eqn[2]
    x0, y0 = coor[0], coor[1]

    return (a * x0 + b * y0 + c) / np.sqrt(a**2 + b**2)

# Given the full class image, the line parameters and the coordinates of the barycenter,
# compute for each side if the curve of the piece goes inside (in) or outside (out).
# This is done by computing the mean coordinates for each class and see if the signed distance
# from the corners' line has the same sign of the signed distance of the barycenter. If that
# true, the two points lie on the same side and we have a in; otherwise we have a out.
# To let the points of the curve to contribute more to the mean point calculation, only the
# signed distances that are greater than a threshold are used.
def compute_inout(class_image, line_params, barycenter, d_threshold=10):
    inout = []
    xb, yb = barycenter[0], barycenter[1]

    for line_param, cl in zip(line_params, (1, 2, 3, 4)):
        coords = np.array([zip(*np.where(class_image == cl))])[0]

        distances = np.array([distance_point_line_signed(line_param, (x0, y0)) for y0, x0 in coords])    
        distances = distances[np.abs(distances) > d_threshold]
        m_dist = np.mean(distances)

        b_dist = distance_point_line_signed(line_param, (xb, yb))

        if b_dist * m_dist > 0:
            inout.append('in')
        else:
            inout.append('out')
            
    return inout

#####################
# Generate Side Img #
#####################

def create_side_images(class_image, inout, corners):
    how_to_rotate = [(90, -90), (180, 0), (-90, 90), (0, 180)]
    side_images = []

    for cl in (1, 2, 3, 4):
        side_image = np.zeros(class_image.shape, dtype='uint8')
        side_image[class_image == cl] = cl

        io = inout[cl - 1]
        htw = how_to_rotate[cl - 1]
        side_corners_idx = _corner_indexes[cl - 1]

        htw = htw[0] if io == 'in' else htw[1]
        side_image_rot, M = rotate(side_image, htw)

        side_corners = np.array(np.round([M.dot((corners[corner_idx][0], corners[corner_idx][1], 1)) for corner_idx in side_corners_idx])).astype(np.int)

        # Order the corners from higher (smaller y coordinate)
        if side_corners[0, 1] > side_corners[1, 1]:
            side_corners = side_corners[::-1]

        # Correct the angle on each side separately
        if side_corners[0, 0] != side_corners[1, 0]:
            m = float(side_corners[1, 1] - side_corners[0, 1]) / (side_corners[1, 0] - side_corners[0, 0])
            corners_angle = np.arctan(m) * 180 / np.pi
            correction_angle = - (corners_angle / abs(corners_angle) * 90 - corners_angle)

            side_image_rot, M = rotate(side_image_rot, correction_angle)

        side_image_rot[side_image_rot <= 0.5] = 0
        side_image_rot[side_image_rot > 0.5] = 1

        nz = np.nonzero(side_image_rot)
        min_y, max_y, min_x, max_x = np.min(nz[0]), np.max(nz[0]), np.min(nz[1]), np.max(nz[1])
        side_image_rot = side_image_rot[min_y:max_y+1, min_x:max_x+1]

        side_images.append(side_image_rot)
            
    return side_images

def plot_side_images(side_images, inout):
    for cl, (side_image, io) in enumerate(zip(side_images, inout), start=1):

        plt.subplot(220 + cl)
        plt.title("{0} {1}".format(cl, io))
        plt.imshow(cv2.dilate(side_image, (3,3)))

####################################################################################################################

# Most important functions

def get_default_params():
    # before_segmentation_func: Apply medianBlur to img before segmentation (Cropped out binarize image)
    # bin_threshold: Binarization threshold 130 <= x <= 255 pixels are set to white
    # after_segmentation_func: Apply additional blur to the segmented img
    # scale_factor: Change the scale of the img
    # negate_img: Boolean value to negate binarized image

    # Self-Defined functions:
    # corner_*: Parameters used to extract distinct corners
    # shape_classification_*: Parameters used to generate class_image NOTE: VERY IMPORTANT!!
    # inout_*: Parameters used to determine whehter the edge of that piece goes in or out

    # OpenCV functions:
    #   1) cv2.cornerHarris(img, blockSize, ksize, k)
    #       blockSize - Size of neighborhood considered for corner detection
    #       ksize - Aperture parameter of the Sobel derivative used
    #
    #   2) cv2.erode() - Used for thinning the edges
    #       kernel - Odd sized kernel
    #       iteration - Number of times edges are thinned   

    side_extractor_default_values = {
        'before_segmentation_func': partial(cv2.medianBlur, ksize=5),
        'bin_threshold': 130,
        'after_segmentation_func': None,
        'scale_factor': 0.5,
        'negate_img': True,
        'harris_blocksize': 5,
        'harris_ksize': 5,
        'corner_nsize': 5,
        'corner_score_threshold': 0.2,
        'corner_minmax_threshold': 100,
        'corner_refine_rect_size': 5,
        'edge_erode_size': 3,
        'shape_classification_distance_threshold': 100,
        'shape_classification_nhs': 5,
        'inout_distance_threshold': 5
    }
    
    return side_extractor_default_values.copy()


def process_piece(image, **kwargs):

    out_dict = {} # Output with all the array information
    params = get_default_params()

    # Change any default params if needed
    for key in kwargs:
        params[key] = kwargs[key]

    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        #=== Step 1: Segmentation ===

        before_segmentation_func = params['before_segmentation_func']
        after_segmentation_func = params['after_segmentation_func']
        bin_threshold = params['bin_threshold']

        # Apply some blur by default, helps to remove noise
        if before_segmentation_func is not None:
            gray = before_segmentation_func(gray)

        gray = segment_piece(gray, bin_threshold)

        if params['negate_img']:
            gray = cv2.bitwise_not(gray) # Negate the image

        out_dict['segmented'] = gray.copy() # Keep a copy of the binarized img

        # Further apply blur to the binarized image if needed
        if after_segmentation_func is not None:
            gray = after_segmentation_func(gray)

        #=== Step 2: Extracting pieces ===

        gray = extract_piece(gray)
        ret, labels = cv2.connectedComponents(gray) # Connected pixel == 1 else 0 in labels array
        connected_areas = [np.count_nonzero(labels == l) for l in range(1, ret)]
        max_area_idx = np.argmax(connected_areas) + 1
        gray[labels != max_area_idx] = 0
        gray = 255 - gray
        gray = extract_piece(gray)

        out_dict['extracted'] = gray.copy() # Keep a copy of the cropped binarized img

        #=== Step 3: Harris Corner detection ===

        scaled_size = int(gray.shape[0] * params['scale_factor']), int(gray.shape[1] * params['scale_factor'])
        gray_harris = cv2.resize(gray, scaled_size) # Resize the img shape based on scale factor
        
        harris = cv2.cornerHarris(gray_harris, params['harris_blocksize'], params['harris_ksize'], 0.04)
        harris = harris * gray_harris
        out_dict['harris'] = harris # Img with just the corner points

        xy = get_corners(harris, params['corner_nsize'], params['corner_score_threshold'], params['corner_minmax_threshold'])
        xy = np.round(xy / params['scale_factor']).astype(np.int) # Revert back to original scale
        out_dict['xy'] = xy # x and y-coordinates of the rectangles

        if len(xy) < 4:
            raise RuntimeError('Not enough corners')

        #=== Step 4: Derive the intersections and rectangle===

        intersections = get_best_fitting_rect_coords(xy, perp_angle_thresh=30)
        
        if intersections is None:
            raise RuntimeError('No rectangle found')

        # Find the angle of rotation to rotate the rectangle to the correct orientation
        rotation_angle = 90

        if intersections[1, 0] == intersections[0, 0]:
            rotation_angle = np.arctan2(intersections[1, 1] - intersections[0, 1], intersections[1, 0] - intersections[0, 0]) * 180 / np.pi 

        #=== Step 5: Thin the borders using erode ===

        edges = gray - cv2.erode(gray, np.ones((params['edge_erode_size'], params['edge_erode_size'])))
        
        # Rotate all images
        edges, M = rotate(edges, rotation_angle)
        out_dict['edges'] = edges # Rotated eroded image

        #=== Step 6: Separate the 4 sides of the piece ===
        intersections = np.array(np.round([M.dot((point[0], point[1], 1)) for point in intersections])).astype(np.int)
        yb, xb = compute_barycentre(gray)

        corners = corner_detection(edges, intersections, (xb, yb), params['corner_refine_rect_size'])
        corners = order_corners(corners)
        line_params = compute_line_params(corners)
        class_image = shape_classification(edges, line_params, params['shape_classification_distance_threshold'], params['shape_classification_nhs'])
        
        out_dict['class_image'] = class_image

        inout = compute_inout(class_image, line_params, (xb, yb), params['inout_distance_threshold'])
        out_dict['inout'] = inout

        # Cropped out version of all 4 sides
        side_images = create_side_images(class_image, inout, corners)
        out_dict['side_images'] = side_images
        
    except Exception as e:
        # Record all error to remove those pieces from the results
        out_dict['error'] = e

    finally:
        return out_dict


####################################################################################################################

# Stub testing

postprocess = partial(cv2.blur, ksize=(3, 3))

img = cv2.imread('underwater/cropped.jpg')
process_piece(img, after_segmentation_func=postprocess)

# img = cv2.imread('underwater/underwater_piece.jpg')
# process_piece(img, after_segmentation_func=postprocess, negate_img=True)

cv2.waitKey(0)