import os
from moviepy.editor import ImageSequenceClip

# Input folder containing images
image_folder = 'detected_eff0005/'

# Output video file
output_video = 'video/efficientNetB0_0005.mp4'

# Set the frames per second (FPS)
fps = 24

# Sort the image files in the folder
image_files = sorted([img for img in os.listdir(image_folder) if img.endswith(".jpg")])

# Create a list of image paths
image_paths = [os.path.join(image_folder, img) for img in image_files]

# Create a video clip from the images
clip = ImageSequenceClip(image_paths, fps=fps)

# Write the video to a file
clip.write_videofile(output_video)

print(f"Video created: {output_video}")
