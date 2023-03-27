import subprocess
import json


def probe_file(filename):
    codecName = ''
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

    jstream = result
    for streams in jstream['streams']:
        codecName = streams['codec_name']
        if codecName:
            print(codecName)
            break

    return codecName


def extract_wav(video_file, wav_file):
    # ./ffmpeg -i GMT20220517-140045_Recording.mp4 -acodec pcm_s16le -ac 1 -ar 16000 recout.wav -hide_banner
    # -loglevel error
    command = ['ffmpeg/bin/ffmpeg', '-i', video_file, '-acodec', 'pcm_s16le', '-ac', '1', '-ar', '16000',
               wav_file, '-hide_banner', '-loglevel', 'error']
    print('ffmpeg command:', command)
    subprocess.run(command, stdout=subprocess.PIPE, stdin=subprocess.PIPE)
    return wav_file


def norm_wav(video_file, wav_file):
    # ./ffmpeg -i input.wav -filter:a "dynaudnorm=p=0.9:s=5" output.wav -hide_banner
    # -loglevel error
    command = ['ffmpeg/bin/ffmpeg', '-i', video_file, '-af', 'volumedetect', '-vn', '-sn', '-dn',
               wav_file, '-hide_banner', '-loglevel', 'error']
    print('ffmpeg command:', command)
    subprocess.run(command, stdout=subprocess.PIPE, stdin=subprocess.PIPE)
    return wav_file


def replace_audio(video_filename, audio_wav, out_file):
    # ./ffmpeg -i v.mp4 -i a.wav -c:v copy -map 0:v:0 -map 1:a:0 new.mp4 -hide_banner -loglevel error
    command = ['ffmpeg/bin/ffmpeg', '-i', video_filename, '-i', audio_wav, '-c:v', 'copy', '-map', '0:v:0', '-map',
               '1:a:0', out_file, '-hide_banner', '-loglevel', 'error']
    print('ffmpeg command:', command)
    subprocess.run(command, stdout=subprocess.PIPE, stdin=subprocess.PIPE)
    return out_file


def getcodec_audio(video_filename):
    # ffprobe -v error -select_streams v:0 -show_entries stream=codec_name -of default=noprint_wrappers=1:nokey=1
    # video.mkv
    command = ['ffmpeg/bin/ffprobe', '-v', 'quiet', '-print_format', 'json', '-select_streams', 'v:0', '-show_entries', 'stream=codec_name',
               '-of', 'default=noprint_wrappers=1:nokey=1', video_filename]
    print('ffprobe command:', command)
    subprocess.run(command, stdout=subprocess.PIPE, stdin=subprocess.PIPE)
