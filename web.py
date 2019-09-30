import sys
from flask import Flask, render_template, redirect, url_for, request, send_from_directory
import numpy as np
import os
import torch
from scipy.io.wavfile import write
import uuid
import time
import random
from synthesizer.inference import Synthesizer
from encoder import inference as encoder
from vocoder import inference as vocoder
from pathlib import Path
import librosa
import argparse
from pydub import AudioSegment

synthesizer = None
args = None
def init_model():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-e", "--enc_model_fpath", type=str,
                        default="encoder/saved_models/pretrained.pt",
                        help="Path to a saved encoder")
    parser.add_argument("-s", "--syn_model_dir", type=str,
                        default="synthesizer/saved_models/logs-pretrained/",
                        help="Directory containing the synthesizer model")
    parser.add_argument("-v", "--voc_model_fpath", type=str,
                        default="vocoder/saved_models/pretrained/pretrained.pt",
                        help="Path to a saved vocoder")
    parser.add_argument("-t", "--text_list", type=str,
                        default="그는 괜찮은 척하려고 애쓰는것 같았다.|지난해 삼월 김전장관의 동료인 장동련 홍익대 교수가 민간 자문단장으로 위촉되었습니다.|설빙 슈퍼브랜드데이 딸기치즈메론 시즌한정|그래도 권위있는 아이비리그 명문대학의 발표이니 믿어야 할까요?|안녕하시렵니까? 저는 인공지능이예요.|상처가 있는 부위 등에는 사용을 자제해 주세요.|안녕하세요. 저는지금 성대모사중이예요.",
                        help="Path to a saved vocoder")
    parser.add_argument("--low_mem", action="store_true", help= \
        "If True, the memory used by the synthesizer will be freed after each use. Adds large "
        "overhead but allows to save some GPU memory for lower-end GPUs.")
    global args
    args = parser.parse_args()

    encoder.load_model(args.enc_model_fpath)
    syn_path = Path(args.syn_model_dir)
    global synthesizer
    synthesizer = Synthesizer(syn_path.joinpath("taco_pretrained"), low_mem=args.low_mem)
    vocoder.load_model(args.voc_model_fpath)

def create_app():
    app = Flask(__name__)
    def run_on_start():
        init_model()
    run_on_start()
    return app

tmp_dir = "static/tmp"
os.makedirs(tmp_dir, exist_ok=True)
wav_result_dir = "static/result"
os.makedirs(wav_result_dir, exist_ok=True)
app = create_app()
# app = Flask(__name__)

@app.route("/index")
def index():
    return render_template("recorder.html")

@app.route("/record", methods=['POST'])
def record():
    f = request.files['audio_data']
    filename = str(uuid.uuid4())
    f.save(os.path.join(tmp_dir, filename + ".webm"))

    import subprocess

    command = "ffmpeg -i filename.webm -ab 160k -ac 2 -ar 16000 -vn %s.wav" % os.path.join(tmp_dir, filename + ".wav")

    subprocess.call(command, shell=True)
    target_wav_path = os.path.join(tmp_dir, filename + ".wav")
    filename_list = []

    text_list = args.text_list.split("|")
    for j, text in enumerate(text_list):

        ## Load the models one by one.
        print("Preparing the encoder, the synthesizer and the vocoder...")

        ## Computing the embedding
        # First, we load the wav using the function that the speaker encoder provides. This is
        # important: there is preprocessing that must be applied.

        # The following two methods are equivalent:
        # - Directly load from the filepath:
        # preprocessed_wav = encoder.preprocess_wav(target_wav_path)
        # - If the wav is already loaded:
        original_wav, sampling_rate = librosa.load(target_wav_path)
        preprocessed_wav = encoder.preprocess_wav(original_wav, sampling_rate)
        print("Loaded file succesfully")

        embed = encoder.embed_utterance(preprocessed_wav)
        print("Created the embedding")

        # The synthesizer works in batch, so you need to put your data in a list or numpy array
        texts = [text]
        embeds = [embed]
        # If you know what the attention layer alignments are, you can retrieve them here by
        # passing return_alignments=True
        specs = synthesizer.synthesize_spectrograms(texts, embeds)
        spec = specs[0]
        print("Created the mel spectrogram")

        ## Generating the waveform
        print("Synthesizing the waveform:")
        # Synthesizing the waveform is fairly straightforward. Remember that the longer the
        # spectrogram, the more time-efficient the vocoder.
        generated_wav = vocoder.infer_waveform(spec)

        ## Post-generation
        # There's a bug with sounddevice that makes the audio cut one second earlier, so we
        # pad it.
        generated_wav = np.pad(generated_wav, (0, synthesizer.sample_rate), mode="constant")

        # Save it on the disk
        filename = str(uuid.uuid4()) + '.wav'
        audio_path = os.path.join(wav_result_dir, filename)
        librosa.output.write_wav(audio_path, generated_wav.astype(np.float32),
                                 synthesizer.sample_rate)
        filename_list.append(filename)

    return render_template("synth.html", filename=filename_list)

@app.after_request
def add_header(r):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r


@app.route('/download/<path:filename>', methods=['GET', 'POST'])
def download(filename):
    return send_from_directory(directory=wav_result_dir, filename=filename, as_attachment=True)

@app.route('/download_mp3/<path:filename>', methods=['GET', 'POST'])
def download_mp3(filename):
    new_filename = os.path.splitext(filename)[0] + ".mp3"
    AudioSegment.from_wav(os.path.join(wav_result_dir, filename)).export(os.path.join(wav_result_dir, new_filename), format="mp3")
    return send_from_directory(directory=wav_result_dir, filename=new_filename, as_attachment=True)

if __name__ == "__main__":

    app.run(debug=True, threaded=False)
