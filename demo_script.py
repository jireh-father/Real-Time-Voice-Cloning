from encoder.params_model import model_embedding_size as speaker_embedding_size
from utils.argutils import print_args
from synthesizer.inference import Synthesizer
from encoder import inference as encoder
from vocoder import inference as vocoder
import numpy as np
import librosa
import argparse
import torch
import sys
import os

if __name__ == '__main__':
    ## Info & args
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
                        default="그는 괜찮은 척하려고 애쓰는것 같았다.|지난해 삼월 김전장관의 동료인 장동련 홍익대 교수가 민간 자문단장으로 위촉되면서 본격적인 공모와 개발 작업에 들어갔다.|[설빙](슈퍼브랜드데이) 딸기치즈메론(시즌한정)",
                        help="Path to a saved vocoder")
    parser.add_argument("-w", "--wav_list", type=str,
                        default="vocoder/saved_models/pretrained/pretrained.pt",
                        help="Path to a saved vocoder")
    parser.add_argument("-n", "--name_list", type=str,
                        default="vocoder/saved_models/pretrained/pretrained.pt",
                        help="Path to a saved vocoder")
    parser.add_argument("-o", "--output_dir", type=str,
                        default="./result",
                        help="Path to a saved vocoder")
    parser.add_argument("--low_mem", action="store_true", help= \
        "If True, the memory used by the synthesizer will be freed after each use. Adds large "
        "overhead but allows to save some GPU memory for lower-end GPUs.")
    parser.add_argument("--no_sound", action="store_true", help= \
        "If True, audio won't be played.")
    args = parser.parse_args()
    print_args(args, parser)
    if not args.no_sound:
        import sounddevice as sd

    ## Print some environment information (for debugging purposes)
    print("Running a test of your configuration...\n")
    if not torch.cuda.is_available():
        print("Your PyTorch installation is not configured to use CUDA. If you have a GPU ready "
              "for deep learning, ensure that the drivers are properly installed, and that your "
              "CUDA version matches your PyTorch installation. CPU-only inference is currently "
              "not supported.", file=sys.stderr)
        quit(-1)
    device_id = torch.cuda.current_device()
    gpu_properties = torch.cuda.get_device_properties(device_id)
    print("Found %d GPUs available. Using GPU %d (%s) of compute capability %d.%d with "
          "%.1fGb total memory.\n" %
          (torch.cuda.device_count(),
           device_id,
           gpu_properties.name,
           gpu_properties.major,
           gpu_properties.minor,
           gpu_properties.total_memory / 1e9))

    enc_list = args.enc_model_fpath.split(",")
    syn_list = args.syn_model_dir.split(",")
    voc_list = args.voc_model_fpath.split(",")
    wav_list = args.wav_list.split(",")
    name_list = args.name_list.split(",")
    text_list = args.text_list.split("|")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    total_cnt = len(enc_list) * len(wav_list) * len(text_list)
    created_cnt = 0
    for i in range(len(enc_list)):
        enc_path = enc_list[i]
        syn_path = syn_list[i]
        voc_path = voc_list[i]
        for target_wav_path in wav_list:
            target_name = os.path.splitext(os.path.basename(target_wav_path))[0]
            for j, text in enumerate(text_list):

                ## Load the models one by one.
                print("Preparing the encoder, the synthesizer and the vocoder...")
                encoder.load_model(enc_path)
                synthesizer = Synthesizer(os.path.join(syn_path, "taco_pretrained"), low_mem=args.low_mem)
                vocoder.load_model(voc_path)

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

                # Play the audio (non-blocking)
                if not args.no_sound:
                    sd.stop()
                    sd.play(generated_wav, synthesizer.sample_rate)

                # Save it on the disk
                fpath = os.path.join(args.output_dir, "%s_%s_%02d.wav" % (name_list[i], target_name, j))
                print(generated_wav.dtype)
                librosa.output.write_wav(fpath, generated_wav.astype(np.float32),
                                         synthesizer.sample_rate)
                created_cnt += 1
                print("\n[%d/%d] Saved output as %s\n\n" % (created_cnt, total_cnt, fpath))
