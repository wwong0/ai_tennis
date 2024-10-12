from ultralytics import YOLO
import cv2
import pickle

class PlayerTracker:
    def __init__(self,model_path):
        self.model = YOLO(model_path)

    def detect_frames(self,frames, read_from_stub = False, stub_path=None):
        player_detections = []

        #loads detections from stub and returns if read_from_stub
        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as f:
                player_detections = pickle.load(f)
            return player_detections

        #if no stub, appends player detections to the player_dections list
        for frame in frames:
            player_dict = self.detect_frame(frame)
            player_detections.append(player_dict)

        #writes a stub
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(player_detections, f)

        return player_detections

    def detect_frame(self, frame):
        #runs a frame through the model to track objects
        results = self.model.track(frame, persist=True)[0]
        #class ids to names mapping
        id_name_dict = results.names

        player_dict = {}
        for box in results.boxes:
            track_id = int(box.id.tolist()[0])
            result = box.xyxy.tolist()[0] #bbox cords
            object_cls_id = box.cls.tolist()[0] #class id
            object_cls_name = id_name_dict[object_cls_id] #class name
            if object_cls_name == "person":
                #add tracking id of people to, track ids persist between the same obj
                player_dict[track_id] = result

        return player_dict

    def draw_bboxes(self, video_frames, player_detections):
        output_video_frames = []
        for frame, player_dict in zip(video_frames, player_detections):
            # Draw Bounding Boxes
            for track_id, bbox in player_dict.items():
                x1, y1, x2, y2 = bbox
                cv2.putText(frame, f"Player Id: {track_id}", (int(bbox[0]), int(bbox[1] - 10 )), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            output_video_frames.append(frame)

        return output_video_frames