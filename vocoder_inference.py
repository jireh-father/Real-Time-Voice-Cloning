from utils.argutils import print_args
from vocoder import inference as vocoder
import numpy as np
import librosa
import argparse
import torch
import traceback

if __name__ == '__main__':
    ## Info & args
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-v", "--voc_model_fpath", type=str,
                        default="vocoder/saved_models/pretrained/pretrained.pt",
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

    ## Load the models one by one.
    print("Preparing the encoder, the synthesizer and the vocoder...")
    vocoder.load_model(args.voc_model_fpath)

    ## Run a test
    print("Testing your configuration with small inputs.")
    # Forward an audio waveform of zeroes that lasts 1 second. Notice how we can get the encoder's
    # sampling rate, which may differ.
    # If you're unfamiliar with digital audio, know that it is encoded as an array of floats
    # (or sometimes integers, but mostly floats in this projects) ranging from -1 to 1.
    # The sampling rate is the number of values (samples) recorded per second, it is set to
    # 16000 for the encoder. Creating an array of length <sampling_rate> will always correspond
    # to an audio of 1 second.
    print("\tTesting the encoder...")
    num_generated = 0
    while True:
        try:
            # Get the reference audio filepath
            message = "Reference voice: enter an audio filepath of a voice to be cloned (mp3, " \
                      "wav, m4a, flac, ...):\n"
            mel_path = input(message)

            ## Computing the embedding
            # First, we load the wav using the function that the speaker encoder provides. This is
            # important: there is preprocessing that must be applied.

            # The following two methods are equivalent:
            # - Directly load from the filepath:
            melspec = torch.load(mel_path)
            # Synthesizing the waveform is fairly straightforward. Remember that the longer the
            # spectrogram, the more time-efficient the vocoder.
            generated_wav = vocoder.infer_waveform(melspec.unsqueeze(0))

            ## Post-generation
            # There's a bug with sounddevice that makes the audio cut one second earlier, so we
            # pad it.
            generated_wav = np.pad(generated_wav, (0, 22050), mode="constant")

            # Save it on the disk
            fpath = "vocoder_output_%02d.wav" % num_generated
            print(generated_wav.dtype)
            librosa.output.write_wav(fpath, generated_wav.astype(np.float32), 22050)
            num_generated += 1
            print("\nSaved output as %s\n\n" % fpath)


        except Exception as e:
            print(traceback.print_exc())
            print("Caught exception: %s" % repr(e))
            print("Restarting\n")
