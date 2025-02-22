from synthesizer.preprocess import create_embeddings, create_embeddings_custom_dataset
from utils.argutils import print_args
from pathlib import Path
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Creates embeddings for the synthesizer from the LibriSpeech utterances.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-e", "--encoder_model_fpath", type=Path,
                        default="encoder/saved_models/pretrained.pt", help=\
        "Path your trained encoder model.")
    parser.add_argument("-o", "--output_dir", type=str,
                        default="embedding")
    parser.add_argument("-t", "--target_files", type=str,
                        default="filelists_kss_bak/ljs_audio_text_train_filelist.txt,filelists_kss_bak/ljs_audio_text_val_filelist.txt")
    parser.add_argument("-n", "--n_processes", type=int, default=4, help= \
        "Number of parallel processes. An encoder is created for each, so you may need to lower "
        "this value on GPUs with low memory. Set it to 1 if CUDA is unhappy.")
    args = parser.parse_args()
    
    # Preprocess the dataset
    print_args(args, parser)
    create_embeddings_custom_dataset(**vars(args))
