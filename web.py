# -*- coding: utf-8 -*-
import sys
from flask import Flask, render_template, redirect, url_for, request, send_from_directory
import numpy as np
import os
import uuid
# from synthesizer.inference import Synthesizer
# from encoder import inference as encoder
# from vocoder import inference as vocoder
from pathlib import Path
import librosa
from pydub import AudioSegment
import subprocess

synthesizer = None


def init_model():
    enc_model_fpath = "/home/ilseo/source/Real-Time-Voice-Cloning/encoder/saved_models/batch64_10_e256_val_12_epoch.pt"
    syn_model_dir = "/home/ilseo/source/Real-Time-Voice-Cloning/synthesizer/saved_models/logs-synth_kr_epoch12_508k_steps/"
    voc_model_fpath = "/home/ilseo/source/Real-Time-Voice-Cloning/vocoder/saved_models/dim256_gta_bs256_epoch_12_508k_20190926/dim256_gta_bs256_epoch_12_508k_20190926.pt"

    enc_path = Path(enc_model_fpath)
    syn_path = Path(syn_model_dir)
    voc_path = Path(voc_model_fpath)
    encoder.load_model(enc_path)
    global synthesizer
    synthesizer = Synthesizer(syn_path.joinpath("taco_pretrained"))
    vocoder.load_model(voc_path)


def create_app():
    app = Flask(__name__)

    def run_on_start():
        init_model()

    # run_on_start()
    return app


tmp_dir = "/home/ilseo/source/Real-Time-Voice-Cloning/static/tmp"
os.makedirs(tmp_dir, exist_ok=True)
wav_result_dir = "/home/ilseo/source/Real-Time-Voice-Cloning/static/result"
os.makedirs(wav_result_dir, exist_ok=True)
app = create_app()


# app = Flask(__name__)

@app.route("/upload")
def upload():
    return render_template("upload.html")

@app.route("/index")
def index():
    return render_template("recorder.html")


@app.route("/record", methods=['POST', 'GET'])
def record():
    f = request.files['audio']
    filename = str(uuid.uuid4())
    ext = os.path.splitext(request.files['audio'].filename)[1]
    text_list = request.form.get('text_list', default="텍스트를 입력하라 말이여.")
    text_list = text_list.split(os.linesep)
    print(text_list)
    f.save(os.path.join(tmp_dir, filename + ext))

    if ext.lower() in [".mp3", ".m4a"]:
        command = "avconv -i %s %s" % (os.path.join(tmp_dir, filename + ext), os.path.join(tmp_dir, filename + ".wav"))

        subprocess.call(command, shell=True)
    target_wav_path = os.path.join(tmp_dir, filename + ".wav")
    filename_list = []

    for j, text in enumerate(text_list):
        text = text.strip()
        if len(text) == 0:
            continue
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
        filename = "%s_%d.wav" % (filename, j)
        audio_path = os.path.join(wav_result_dir, filename)
        librosa.output.write_wav(audio_path, generated_wav.astype(np.float32),
                                 synthesizer.sample_rate)
        filename_list.append(filename)

    return render_template("synth.html", file_list=filename_list)


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
    AudioSegment.from_wav(os.path.join(wav_result_dir, filename)).export(os.path.join(wav_result_dir, new_filename),
                                                                         format="mp3")
    return send_from_directory(directory=wav_result_dir, filename=new_filename, as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True, threaded=False)
