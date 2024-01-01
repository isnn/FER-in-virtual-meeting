from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip

# Input video file
input_video_path = './video/streamMeet1.mp4'

# Output video file
output_video_path = './video/streamMeet1_test.mp4'

# Start times (in minutes and seconds)
start_minutes = 27
start_seconds = 25

# End times (in minutes and seconds)
end_minutes = 27
end_seconds = 30

# Convert start and end times to seconds
start_time = start_minutes * 60 + start_seconds
end_time = end_minutes * 60 + end_seconds

# Cut the video
ffmpeg_extract_subclip(input_video_path, start_time, end_time, targetname=output_video_path)

print(f"Video cut from {start_time} to {end_time} seconds saved as '{output_video_path}'")
