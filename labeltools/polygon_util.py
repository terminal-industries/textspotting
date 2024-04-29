from scipy.spatial import distance as dist
import numpy as np
import math
 
 
def cos_dist(a, b):
    if len(a) != len(b):
        return None
    part_up = 0.0
    a_sq = 0.0
    b_sq = 0.0
    #print a, b
    #print zip(a, b)
    for a1, b1 in zip(a, b):
        part_up += a1*b1
        a_sq += a1**2
        b_sq += b1**2
    part_down = math.sqrt(a_sq*b_sq)
    if part_down == 0.0:
        return None
    else:
        return part_up / part_down
 
 
# this function is confined to rectangle
def order_points(pts):
    # sort the points based on their x-coordinates
    xSorted = pts[np.argsort(pts[:, 0]), :]
 
    # grab the left-most and right-most points from the sorted
    # x-roodinate points
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]
 
    # now, sort the left-most coordinates according to their
    # y-coordinates so we can grab the top-left and bottom-left
    # points, respectively
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost
 
    # now that we have the top-left coordinate, use it as an
    # anchor to calculate the Euclidean distance between the
    # top-left and right-most points; by the Pythagorean
    # theorem, the point with the largest distance will be
    # our bottom-right point
    D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
    (br, tr) = rightMost[np.argsort(D)[::-1], :]
 
    # return the coordinates in top-left, top-right,
    # bottom-right, and bottom-left order
    return np.array([tl, tr, br, bl], dtype="float32")
 
 
def order_points_quadrangle(pts):
    # sort the points based on their x-coordinates
    xSorted = pts[np.argsort(pts[:, 0]), :]
 
    # grab the left-most and right-most points from the sorted
    # x-roodinate points
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]
 
    # now, sort the left-most coordinates according to their
    # y-coordinates so we can grab the top-left and bottom-left
    # points, respectively
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost
 
    # now that we have the top-left and bottom-left coordinate, use it as an
    # base vector to calculate the angles between the other two vectors
 
    vector_0 = np.array(bl-tl)
    vector_1 = np.array(rightMost[0]-tl)
    vector_2 = np.array(rightMost[1]-tl)
 
    angle = [np.arccos(cos_dist(vector_0, vector_1)), np.arccos(cos_dist(vector_0, vector_2))]
    (br, tr) = rightMost[np.argsort(angle), :]
 
    # return the coordinates in top-left, top-right,
    # bottom-right, and bottom-left order
    return np.array([tl, tr, br, bl], dtype="float32")




def sort_remaining_points(center, remaining_points):
    # Calculate the angle from center to each point, negative for counter-clockwise sort
    angles = np.arctan2(remaining_points[:, 1] - center[1], remaining_points[:, 0] - center[0])
    # Sort points by angle in counter-clockwise direction (negative to sort correctly)
    sorted_indices = np.argsort(angles)
    return remaining_points[sorted_indices]    

def sort_vertical_points(points):

    # sort the points based on their x-coordinates
    xSorted = points[np.argsort(points[:, 0]), :]
 
    # grab the left-most and right-most points from the sorted
    # x-roodinate points
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]
 
    # now, sort the left-most coordinates according to their
    # y-coordinates so we can grab the top-left and bottom-left
    # points, respectively
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost
    top_left = (tl, bl)
    
    # Remove the top left point from the list
    remaining_points = np.array([p for p in points if not np.all(p == top_left)])
    
    # Compute the centroid of the remaining points
    center = np.mean(remaining_points, axis=0)
    
    # Sort remaining points in counter-clockwise order
    sorted_points = sort_remaining_points(center, remaining_points)
    
    sorted_points[[1, 3]] = sorted_points[[3, 1]]
    
    return sorted_points