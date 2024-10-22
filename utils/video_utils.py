import cv2
import subprocess

def read_video(video_path):
    #video capture object allows reading of video frame by frame
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        #read is if the frame was sucessfully read if not then its the end of the video
        #framedata is read as a numpy array
        read, frame = cap.read()
        if not read:
            break
        frames.append(frame)
    cap.release()
    return frames

def save_video(output_video_frames, output_video_path):
    fourcc = cv2.VideoWriter.fourcc(*'mp4v')
    #creates a videowriter obj to write the video
    out = cv2.VideoWriter(output_video_path, fourcc, 24, (output_video_frames[0].shape[1], output_video_frames[0].shape[0]))
    for frame in output_video_frames:
        out.write(frame)
    out.release()

def convert_to_mp4(input_path, output_path):
    subprocess.run([
        'ffmpeg', '-y', '-i', input_path, '-vcodec', 'libx264', '-acodec', 'aac', output_path
    ], check = True)