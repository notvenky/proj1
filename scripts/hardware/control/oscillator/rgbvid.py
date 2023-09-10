import cv2
import numpy as np
import csv
import subprocess

# Function to compress video using FFmpeg
def compress_video(input_file, output_file):
    command = f"ffmpeg -i {input_file} -vcodec libx264 -crf 23 {output_file}"
    subprocess.call(command, shell=True)

def save_rgb_pixel_details(video_file, output_file):
    cap = cv2.VideoCapture(video_file)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # Write header for the CSV file
        header = ['Frame', 'X', 'Y', 'R', 'G', 'B']
        writer.writerow(header)

        frame_num = 0
        while frame_num < frames:
            ret, frame = cap.read()
            if not ret:
                break

            # Resize the frame to a smaller size for compression
            compressed_frame = cv2.resize(frame, (width // 4, height // 4))

            # Extract RGB pixel details from the compressed frame
            for y in range(compressed_frame.shape[0]):
                for x in range(compressed_frame.shape[1]):
                    r, g, b = compressed_frame[y, x]
                    # Write RGB values to the CSV file
                    writer.writerow([frame_num, x, y, r, g, b])

            frame_num += 1

    cap.release()

if __name__ == "__main__":
    input_file = 'input_video.mp4'
    output_file = 'compressed_video.mp4'

    # Compress the video first to a smaller size
    compress_video(input_file, output_file)

    # Save RGB pixel details from the compressed video
    save_rgb_pixel_details(output_file, 'rgb_pixel_details.csv')
