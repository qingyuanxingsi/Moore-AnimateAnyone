import cv2
import random

def extract_random_frame(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    random_frame_index = random.randint(0, total_frames - 1)

    cap.set(cv2.CAP_PROP_POS_FRAMES, random_frame_index)
    ret, frame = cap.read()

    if ret:
        cv2.imwrite(output_path, frame)
        print("Random frame extracted and saved successfully.")
    else:
        print("Failed to extract random frame from the video.")

    cap.release()

video_path = 'datasets/talk/seg_filter_no_sub_filter/3-Scene-071.mp4'
output_path = 'configs/inference/ref_images/anyone-12.png'

extract_random_frame(video_path, output_path)