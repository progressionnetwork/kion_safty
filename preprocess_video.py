import datetime
import subprocess
import sys
import cv2
import json
from datetime import timedelta
import numpy as np
import os
from datetime import datetime, timedelta
from utils import getcodec_audio


BRIGHTNESS_CHANGE_THRESHOLD = 40


def detect_fragments_with_flashes1(video_path):
    # Load the video file
    cap = cv2.VideoCapture(video_path)

    # Initialize variables
    fragments_with_flashes = []
    fragments = []
    current_fragment_start = None
    current_fragment_end = None
    previous_frame = None
    previous_brightness = None
    frame_count = 0

    # Loop through the video frames
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Calculate the brightness of the frame
        brightness = cv2.mean(gray_frame)[0]

        # Check if the frame has a white flash
        if previous_frame is not None:
            frame_diff = cv2.absdiff(gray_frame, previous_frame)
            diff_mean = cv2.mean(frame_diff)[0]
            if diff_mean > BRIGHTNESS_CHANGE_THRESHOLD and brightness > previous_brightness:
                if current_fragment_start is None:
                    current_fragment_start = frame_count - 1
                current_fragment_end = frame_count

        # Check if the brightness of the frame has changed too fast
        if previous_brightness is not None:
            brightness_diff = abs(brightness - previous_brightness)
            if brightness_diff > BRIGHTNESS_CHANGE_THRESHOLD:
                if current_fragment_start is None:
                    current_fragment_start = frame_count - 1
                current_fragment_end = frame_count

        # If a fragment has ended, add it to the appropriate list
        if current_fragment_start is not None and current_fragment_end is not None:
            if previous_frame is not None:
                fragments_with_flashes.append((current_fragment_start, current_fragment_end))
            fragments.append((current_fragment_start, current_fragment_end))
            current_fragment_start = None
            current_fragment_end = None

        # Update the previous frame and brightness
        previous_frame = gray_frame
        previous_brightness = brightness

        # Increment the frame count
        frame_count += 1

    # Release the video file and destroy the window
    cap.release()
    cv2.destroyAllWindows()

    # If a fragment is currently being tracked, add its end timestamp to the list of fragments
    if current_fragment_start is not None and current_fragment_end is not None:
        fragments.append((current_fragment_start, current_fragment_end))

    return fragments_with_flashes, fragments


def detect_flash(video_file):
    cap = cv2.VideoCapture(video_file)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Threshold for detecting a flash
    flash_threshold = 150

    # Minimum duration (in seconds) for a fragment to be considered a flash
    min_flash_duration = 0.5

    fragments_with_flashes = []
    all_fragments = []

    previous_frame = None
    previous_brightness = None
    fragment_start_time = None
    fragment_end_time = None

    for i in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to grayscale and compute its brightness
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = cv2.mean(gray)[0]

        # Check for a flash
        if previous_frame is not None:
            frame_diff = cv2.absdiff(previous_frame, frame)
            diff_mean = cv2.mean(frame_diff)[0]
            if diff_mean > flash_threshold or abs(brightness - previous_brightness) > 100:
                if fragment_start_time is None:
                    fragment_start_time = i / fps
                fragment_end_time = i / fps
        previous_frame = frame.copy()
        previous_brightness = brightness

        # Check if the current fragment has ended
        if fragment_end_time is not None and fragment_end_time - fragment_start_time >= min_flash_duration:
            if fragment_end_time - fragment_start_time >= min_flash_duration:
                fragments_with_flashes.append((int(fragment_start_time * 1000), int(fragment_end_time * 1000)))
            else:
                all_fragments.append((int(fragment_start_time * 1000), int(fragment_end_time * 1000)))
            fragment_start_time = None
            fragment_end_time = None

    # Check if the last fragment has ended
    if fragment_end_time is not None:
        if fragment_end_time - fragment_start_time >= min_flash_duration:
            fragments_with_flashes.append((int(fragment_start_time * 1000), int(fragment_end_time * 1000)))
        else:
            all_fragments.append((int(fragment_start_time * 1000), int(fragment_end_time * 1000)))

    cap.release()

    # Convert to SRT format
    srt_output = ''
    fragment_counter = 1
    for fragment_start, fragment_end in fragments_with_flashes:
        srt_output += str(fragment_counter) + '\n'
        srt_output += '{0} --> {1}\n'.format(timedelta(milliseconds=fragment_start), timedelta(milliseconds=fragment_end))
        srt_output += ' Frame {0} - Frame {1}\n\n'.format(int(fragment_start / 1000 * fps), int(fragment_end / 1000 * fps))
        fragment_counter += 1

    for fragment_start, fragment_end in all_fragments:
        srt_output += str(fragment_counter) + '\n'
        srt_output += '{0} --> {1}\n'.format(timedelta(milliseconds=fragment_start), timedelta(milliseconds=fragment_end))
        srt_output += ' None\n\n'
        fragment_counter += 1

    return srt_output


def remove_flashes(input_video, output_video, brightness_threshold, brightness_drop):
    cap = cv2.VideoCapture(input_video)
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = cv2.VideoWriter(output_video, fourcc, fps, frame_size)

    previous_frame = None
    previous_brightness = None
    is_flashing = False
    flash_start_frame = None

    for frame_idx in range(total_frames):
        ret, frame = cap.read()

        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)

        if previous_brightness is not None and abs(brightness - previous_brightness) > brightness_drop:
            is_flashing = True
            flash_start_frame = frame_idx

        if is_flashing and brightness < brightness_threshold:
            is_flashing = False
            for i in range(flash_start_frame, frame_idx):
                writer.write(previous_frame)

        elif not is_flashing:
            writer.write(frame)

        previous_frame = frame
        previous_brightness = brightness

    cap.release()
    writer.release()
    cv2.destroyAllWindows()



#inputFile = 'D:\Projects\kion_safty\demos\Free Epilepsy Test [2017] (possible seizures).mp4'
# inputFile = 'D:\Projects\kion_safty\demos\Landshapes  Rosemary Official Video Photosensitive Seizure Warning_1080p.mp4'
inputFile = 'D:\Projects\kion_safty\demos\SCREAMER.mp4'
outputFile = 'D:\Projects\kion_safty\demos\LCREAMER.mp4'
codecName = "avc1"

video_path = inputFile
srt_lines = detect_flash(video_path)
print('Fragments with flashes:', srt_lines)
remove_flashes(video_path, outputFile, brightness_threshold=30, brightness_drop=5)
sys.exit()

