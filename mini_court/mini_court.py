import cv2
import numpy as np
import sys
sys.path.append("../")
import constants
from utils import (
    convert_pixel_distance_to_meters,
    convert_meters_to_pixel_distance
)

class MiniCourt:
    def __init__(self, first_frame):
        self.drawing_rectangle_width = 250
        self.drawing_rectangle_height = 500
        self.buffer = 50
        self.padding_court = 20

        self.set_canvas_background_box_position(first_frame)
        self.set_mini_court_position()
        self.set_court_drawing_key_points()
        self.set_court_lines()



    def convert_meters_to_pixels(self, meters):
        return convert_meters_to_pixel_distance(meters,
                                                constants.DOUBLE_LINE_WIDTH,
                                                self.court_drawing_width
                                                )

    def set_court_drawing_key_points(self):
        drawing_key_points = [0]*28

        # point 0
        drawing_key_points[0], drawing_key_points[1] = int(self.court_start_x), int(
            self.court_start_y)
        # point 1
        drawing_key_points[2], drawing_key_points[3] = int(self.court_end_x), int(
            self.court_start_y)
        # point 2
        drawing_key_points[4] = int(self.court_start_x)
        drawing_key_points[5] = self.court_start_y + self.convert_meters_to_pixels(
            constants.HALF_COURT_LINE_HEIGHT * 2)
        # point 3
        drawing_key_points[6] = drawing_key_points[0] + self.court_drawing_width
        drawing_key_points[7] = drawing_key_points[5]
        # #point 4
        drawing_key_points[8] = drawing_key_points[0] + self.convert_meters_to_pixels(
            constants.DOUBLE_ALLY_DIFFERENCE)
        drawing_key_points[9] = drawing_key_points[1]
        # #point 5
        drawing_key_points[10] = drawing_key_points[4] + self.convert_meters_to_pixels(
            constants.DOUBLE_ALLY_DIFFERENCE)
        drawing_key_points[11] = drawing_key_points[5]
        # #point 6
        drawing_key_points[12] = drawing_key_points[2] - self.convert_meters_to_pixels(
            constants.DOUBLE_ALLY_DIFFERENCE)
        drawing_key_points[13] = drawing_key_points[3]
        # #point 7
        drawing_key_points[14] = drawing_key_points[6] - self.convert_meters_to_pixels(
            constants.DOUBLE_ALLY_DIFFERENCE)
        drawing_key_points[15] = drawing_key_points[7]
        # #point 8
        drawing_key_points[16] = drawing_key_points[8]
        drawing_key_points[17] = drawing_key_points[9] + self.convert_meters_to_pixels(
            constants.NO_MANS_LAND_HEIGHT)
        # # #point 9
        drawing_key_points[18] = drawing_key_points[16] + self.convert_meters_to_pixels(
            constants.SINGLE_LINE_WIDTH)
        drawing_key_points[19] = drawing_key_points[17]
        # #point 10
        drawing_key_points[20] = drawing_key_points[10]
        drawing_key_points[21] = drawing_key_points[11] - self.convert_meters_to_pixels(
            constants.NO_MANS_LAND_HEIGHT)
        # # #point 11
        drawing_key_points[22] = drawing_key_points[20] + self.convert_meters_to_pixels(
            constants.SINGLE_LINE_WIDTH)
        drawing_key_points[23] = drawing_key_points[21]
        # # #point 12
        drawing_key_points[24] = int((drawing_key_points[16] + drawing_key_points[18]) / 2)
        drawing_key_points[25] = drawing_key_points[17]
        # # #point 13
        drawing_key_points[26] = int((drawing_key_points[20] + drawing_key_points[22]) / 2)
        drawing_key_points[27] = drawing_key_points[21]

        self.drawing_keypoints = drawing_key_points

    def set_court_lines(self):
        self.lines = [
            (0, 2),
            (4, 5),
            (6, 7),
            (1, 3),

            (0, 1),
            (8, 9),
            (10, 11),
            (10, 11),
            (2, 3)
        ]



    def set_mini_court_position(self):
        self.court_start_x = self.start_x + self.padding_court
        self.court_start_y = self.start_y + self.padding_court
        self.court_end_x = self.end_x - self.padding_court
        self.court_end_y = self.end_y - self.padding_court
        self.court_drawing_width = self.court_end_x - self.court_start_x


    def set_canvas_background_box_position(self, frame):
        
        frame = frame.copy()
        self.end_x = frame.shape[1] - self.buffer
        self.end_y = self.buffer + self.drawing_rectangle_height
        self.start_x = self.end_x - self.drawing_rectangle_width
        self.start_y = self.end_y - self.drawing_rectangle_height

    def draw_court(self,frame):
        # draws kps
        for i in range(0, len(self.drawing_keypoints), 2):
            x = int(self.drawing_keypoints[i])
            y = int(self.drawing_keypoints[i + 1])
            cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)

        # draw lines
        for line in self.lines:
            start_point = (int(self.drawing_keypoints[line[0]*2]), int(self.drawing_keypoints[line[0]*2+1]))
            end_point = (int(self.drawing_keypoints[line[1]*2]), int(self.drawing_keypoints[line[1]*2+1]))
            cv2.line(frame, start_point, end_point, (0, 0, 0), 2)



        return frame

    def draw_background_rectangle(self, frame):

        shapes = np.zeros_like(frame,np.uint8)
        # Draw the rectangle
        cv2.rectangle(shapes, (self.start_x, self.start_y), (self.end_x, self.end_y), (255,255,255), -1)
        out = frame.copy()
        alpha = 0.5
        mask = shapes.astype(bool)
        out[mask] = cv2.addWeighted(frame, alpha, shapes, 1-alpha, 0)[mask]

        return out

    def draw_mini_court(self, frames):
        output_frames = []
        for frame in frames:
            frame = self.draw_background_rectangle(frame)
            frame = self.draw_court(frame)

            output_frames.append(frame)
        return output_frames

