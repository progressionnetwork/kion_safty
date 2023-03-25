import subprocess
import sys

import cv2
import json

from utils import getcodec_audio

SENSITIVITY = 3  # A number between 0 and 10, for intensity of flashes
BUFFER_SIZE = 3  # The number of frames to hold in history, minimum of 2


def capture_size(capture):
    width = capture.get(3)  # frame width
    height = capture.get(4)  # frame height

    print("Width: ", width)
    print("Height: ", height)

    return (int(width), int(height))


def codec(codecName):
    print("Using codec:", codecName)
    return cv2.VideoWriter_fourcc(*codecName)


def setup_stream(inputFile, outputFile, codecName="avc1"):
    inputVideo = cv2.VideoCapture(inputFile)

    videoSize = capture_size(inputVideo)
    videoCodec = codec(codecName)
    frameRate = 20.0

    outputVideo = cv2.VideoWriter(
        outputFile, videoCodec, frameRate, videoSize, True)

    return (inputVideo, outputVideo)


# Gets a numeric value, for average brightness of a given frame
def get_brightness(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    value = cv2.split(frame)[2]
    average = cv2.mean(value)
    return average[0]


# Determins if a given frame is a camera flash, or just slight change
def detect_if_flash(frameBrightness):
    if len(frameBrightness) < BUFFER_SIZE:
        return False
    didFlashHappen = True
    for b in range(BUFFER_SIZE - 1):
        if frameBrightness[b] + SENSITIVITY > frameBrightness[BUFFER_SIZE - 1]:
            didFlashHappen = False
            break
    return didFlashHappen


# Takes a frame which has white flash, brings brightness down to match before frame
def fix_flash():
    return True


def analyse_frames(inputFile, outputFile, codecName):
    inputVideo, outputVideo = setup_stream(inputFile, outputFile, codecName)

    count = 0  # Iterating through number of frames
    buffer = []
    previous_frame = None
    previous_brightness = None

    while inputVideo.isOpened():
        ret, frame = inputVideo.read()

        if ret is True:
            # Calculate brightness of current frame
            frameBrightness = get_brightness(frame)

            # Maintain a buffer of the past 3 frames
            buffer.append(frameBrightness)
            if len(buffer) > BUFFER_SIZE:
                buffer.pop(0)

            # Detect if flash
            if detect_if_flash(buffer):
                # print("Flash detected at frame %d" % count)
                print("Flash detected at frame : " + str(count) + "   timestamp is: ",
                      str(round(inputVideo.get(cv2.CAP_PROP_POS_MSEC), 3)))

                # Convert the frame to grayscale
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Calculate the brightness of the current frame
                brightness = cv2.mean(gray_frame)[0]

                # If the previous frame exists and its brightness is greater than the current frame's brightness
                if previous_frame is not None and previous_brightness > brightness:
                    # Reduce the brightness of the current frame to match the previous frame's brightness
                    alpha = previous_brightness / brightness
                    beta = 0
                    adjusted_frame = cv2.addWeighted(gray_frame, alpha, gray_frame, beta, 0)

                    frame = adjusted_frame
                # If the previous frame exists and its brightness is less than the current frame's brightness
                else:
                    pass

            count += 1

            outputVideo.write(frame)

        else:
            print("stream failed to read")
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("stream ended")
            break

    # Close everything
    outputVideo.release()
    inputVideo.release()
    cv2.destroyAllWindows()


inputFile = 'D:\Projects\kion_safty\demos\Free Epilepsy Test [2017] (possible seizures).mp4'
# inputFile = 'D:\Projects\kion_safty\demos\Landshapes  Rosemary Official Video Photosensitive Seizure Warning_1080p.mp4'
# inputFile = 'D:\Projects\kion_safty\demos\SCREAMER.mp4'
outputFile = 'D:\Projects\kion_safty\demos\LCREAMER.mp4'
codecName = "avc1"


def probe_file(filename):
    command_line = ['ffmpeg/bin/ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format', '-show_streams',
                    f"{filename}"]
    p = subprocess.Popen(command_line, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print(filename)  # * Sanity check *
    out, err = p.communicate()
    if len(out) > 0:  # ** if result okay
        print("==========output==========")
        result = json.loads(out)
    else:
        result = {}
    if err:
        print("========= error ========")
        print(err)
    return result


jstream = probe_file(inputFile)
for streams in jstream['streams']:
    codecName = streams['codec_name']
    if codecName:
        print(codecName)
        break

print("inputFile: ", inputFile)
print("outputFile: ", outputFile)
analyse_frames(inputFile, outputFile, codecName)
