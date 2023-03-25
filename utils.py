import subprocess


def extract_wav(video_file, wav_file):
    # ./ffmpeg -i GMT20220517-140045_Recording.mp4 -acodec pcm_s16le -ac 1 -ar 16000 recout.wav -hide_banner
    # -loglevel error
    command = ['ffmpeg/bin/ffmpeg', '-i', video_file, '-acodec', 'pcm_s16le', '-ac', '1', '-ar', '16000',
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
