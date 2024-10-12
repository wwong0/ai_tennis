import torch
import torchvision.transforms as transforms
import torchvision.models as models
import cv2

class CourtLineDetector:
    def __init__(self, model_path):
        self.model = models.resnet50(pretrained=True)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 14*2)
        self.model.load_state_dict(torch.load(model_path, map_location='cuda'))

        self.transforms = transforms.Compose([
            # converts from numpy array to PIL
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            # converts to a tensor for use with Pytorch
            transforms.ToTensor(),
            # normalizes image for use with imagenet models
            transforms.Normalize([0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
        ])

    def predict(self, image):
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_tensor = self.transforms(img_rgb)
        #unsqeezes the batch size of the tensor, puts the img in a len 1 list, only need to predict 1 set of kps
        image_tensor = image_tensor.unsqueeze(0)

        with torch.no_grad():
            outputs = self.model(image_tensor)

        #unsqueezes it from a list
        keypoints = outputs.squeeze().cpu().numpy()
        original_h, original_w = image.shape[:2]

        keypoints[::2] *= original_w/224.0
        keypoints[1::2] *= original_h/224.0

        return keypoints

    def draw_keypoints(self, image, keypoints):
        for i in range(0, len(keypoints), 2):
            x = int(keypoints[i])
            y = int(keypoints[i+1])

            cv2.putText(image, str(i//2), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.circle(image, (x, y), 5, (0, 0, 255), -1)

        return image

    def draw_keypoints_on_video(self, video_frames, keypoints):
        output_video_frames = []
        for frame in video_frames:
            frame = self.draw_keypoints(frame, keypoints)
            output_video_frames.append(frame)
        return output_video_frames