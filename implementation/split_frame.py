import cv2
import os

"""
CARA PAKAI
cek nama input, output, 
cek lama video - pastikan tidak terlalu lama
"""

# Input video file
video_file = './video/streamMeet1_output.mp4'

# Output folder for frames
output_folder = 'frames'

# Frame rate (FPS) for extraction
fps = 24

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Open the video file
cap = cv2.VideoCapture(video_file)

# Initialize variables
frame_count = 0
success = True

while success:
    # Set the frame position based on the frame rate
    frame_id = int(frame_count * (cap.get(5) / fps))

    # Set the frame position
    cap.set(1, frame_id)

    # Read the next frame
    success, frame = cap.read()

    if success:
        # Save the frame as an image
        frame_filename = os.path.join(output_folder, f'frame_test_{frame_count:04d}.jpg')
        cv2.imwrite(frame_filename, frame)
        
        print(f' {frame_count} processed in {frame_filename}')
        frame_count += 1

# Release the video capture object
cap.release()

print(f"Frames extracted: {frame_count}")
print(f"Frames saved in '{output_folder}' folder.")
