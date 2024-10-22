import cv2
import numpy as np
import constants
from constants import DOUBLE_LINE_WIDTH, HALF_COURT_LINE_HEIGHT, DOUBLE_ALLY_DIFFERENCE, \
    SINGLE_LINE_WIDTH, NO_MANS_LAND_HEIGHT


def convert_pixel_distance_to_meters(pixel_distance, reference_height_in_meters, reference_height_in_pixels):
    return (pixel_distance * reference_height_in_meters) / reference_height_in_pixels

def convert_meters_to_pixel_distance(meters, reference_height_in_meters, reference_height_in_pixels):
    return (meters * reference_height_in_pixels) / reference_height_in_meters

def perspective_transform_detections(court_keypoints, detections):
    source_court_keypoints_pixels = [[court_keypoints[i], court_keypoints[i+1]] for i in range(0, len(court_keypoints), 2)]
    court_keypoints_meters = [[0, 0],
                            [DOUBLE_LINE_WIDTH, 0],
                            [0, 2*HALF_COURT_LINE_HEIGHT],
                            [DOUBLE_LINE_WIDTH, 2*HALF_COURT_LINE_HEIGHT],
                            [DOUBLE_ALLY_DIFFERENCE, 0],
                            [DOUBLE_ALLY_DIFFERENCE, 2*HALF_COURT_LINE_HEIGHT],
                            [DOUBLE_ALLY_DIFFERENCE + SINGLE_LINE_WIDTH, 0],
                            [DOUBLE_ALLY_DIFFERENCE + SINGLE_LINE_WIDTH, 2*HALF_COURT_LINE_HEIGHT],
                            [DOUBLE_ALLY_DIFFERENCE, NO_MANS_LAND_HEIGHT],
                            [DOUBLE_ALLY_DIFFERENCE + SINGLE_LINE_WIDTH, NO_MANS_LAND_HEIGHT],
                            [DOUBLE_ALLY_DIFFERENCE, 2*HALF_COURT_LINE_HEIGHT- NO_MANS_LAND_HEIGHT],
                            [DOUBLE_ALLY_DIFFERENCE + SINGLE_LINE_WIDTH, 2*HALF_COURT_LINE_HEIGHT- NO_MANS_LAND_HEIGHT],
                            [DOUBLE_ALLY_DIFFERENCE + 0.5 * SINGLE_LINE_WIDTH, NO_MANS_LAND_HEIGHT],
                            [DOUBLE_ALLY_DIFFERENCE + 0.5 * SINGLE_LINE_WIDTH, 2*HALF_COURT_LINE_HEIGHT- NO_MANS_LAND_HEIGHT]
                            ]
    source_court_keypoints_pixels = source_court_keypoints_pixels[:4]
    court_keypoints_meters = court_keypoints_meters[:4]

    source_court_keypoints_pixels = np.array(source_court_keypoints_pixels, dtype=np.float32)
    court_keypoints_meters = np.array(court_keypoints_meters, dtype = np.float32)

    matrix = cv2.getPerspectiveTransform(source_court_keypoints_pixels, court_keypoints_meters)
    result_detections = detections.copy()
    for frame_num in range(len(detections)):
        for key in detections[frame_num].keys():
            array = detections[frame_num][key]
            formatted_array = [[array[i], array[i+1]] for i in range(0, len(array), 2)]
            points = np.array(formatted_array, dtype = np.float32)[None, :, :]
            print(f'points : {points}')
            result = cv2.perspectiveTransform(points, matrix)
            result = result.squeeze().tolist()
            result = [x for xs in result for x in xs]
            print(f'result  : {result}')
            result_detections[frame_num][key] = result
    return result_detections