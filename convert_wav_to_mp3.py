from pydub import AudioSegment
import glob
import os


dir = "target_wav"
output_dir = "target_wav_mp3"
os.makedirs(output_dir, exist_ok=True)

files = glob.glob(os.path.join(dir, "*.wav"))
for file in files:
    print(file)
    file_name = os.path.splitext(os.path.basename(file))[0]
    print(os.path.join(output_dir, file_name + ".mp3"))
    AudioSegment.from_wav(file).export(os.path.join(output_dir, file_name + ".mp3"), format="mp3")