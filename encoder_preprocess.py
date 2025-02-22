from encoder.preprocess import preprocess_librispeech, preprocess_voxceleb1, preprocess_voxceleb2, preprocess_zeroth, \
    preprocess_speech_ko, preprocess_etri_8channel, preprocess_librispeech_clean, preprocess_datatang, \
    preprocess_etri_etc
from utils.argutils import print_args
from pathlib import Path
import argparse

if __name__ == "__main__":
    class MyFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
        pass


    parser = argparse.ArgumentParser(
        description="Preprocesses audio files from datasets, encodes them as mel spectrograms and "
                    "writes them to the disk. This will allow you to train the encoder. The "
                    "datasets required are at least one of VoxCeleb1, VoxCeleb2 and LibriSpeech. "
                    "Ideally, you should have all three. You should extract them as they are "
                    "after having downloaded them and put them in a same directory, e.g.:\n"
                    "-[datasets_root]\n"
                    "  -LibriSpeech\n"
                    "    -train-other-500\n"
                    "  -VoxCeleb1\n"
                    "    -wav\n"
                    "    -vox1_meta.csv\n"
                    "  -VoxCeleb2\n"
                    "    -dev",
        formatter_class=MyFormatter
    )
    parser.add_argument("datasets_root", type=Path, help= \
        "Path to the directory containing your LibriSpeech/TTS and VoxCeleb datasets.")
    parser.add_argument("-o", "--out_dir", type=Path, default=argparse.SUPPRESS, help= \
        "Path to the output directory that will contain the mel spectrograms. If left out, "
        "defaults to <datasets_root>/SV2TTS/encoder/")
    parser.add_argument("-d", "--datasets", type=str,
                        default="librispeech_other,voxceleb1,voxceleb2,librispeech_clean,zeroth,speech_ko,etri_8channel,datatang,etri_etc",
                        # default="speech_ko,etri_8channel,datatang,etri_etc",
                        help= \
                            "Comma-separated list of the name of the datasets you want to preprocess. Only the train "
                            "set of these datasets will be used. Possible names: librispeech_other, voxceleb1, "
                            "voxceleb2.")
    parser.add_argument("-s", "--skip_existing", action="store_true", help= \
        "Whether to skip existing output files with the same name. Useful if this script was "
        "interrupted.")
    parser.add_argument("-p", "--num_processes", type=int, default=8)
    parser.add_argument("-u", "--use_short_data", type=bool, default=True)
    args = parser.parse_args()
    print("use_short_data", args.use_short_data)

    # Process the arguments
    args.datasets = args.datasets.split(",")
    if not hasattr(args, "out_dir"):
        args.out_dir = args.datasets_root.joinpath("SV2TTS", "encoder")
    assert args.datasets_root.exists()
    args.out_dir.mkdir(exist_ok=True, parents=True)

    # Preprocess the datasets
    print_args(args, parser)
    preprocess_func = {
        "librispeech_other": preprocess_librispeech,
        "librispeech_clean": preprocess_librispeech_clean,
        "voxceleb1": preprocess_voxceleb1,
        "voxceleb2": preprocess_voxceleb2,
        "zeroth": preprocess_zeroth,
        "speech_ko": preprocess_speech_ko,
        "etri_8channel": preprocess_etri_8channel,
        "datatang": preprocess_datatang,
        "etri_etc": preprocess_etri_etc
    }
    args = vars(args)
    for dataset in args.pop("datasets"):
        print("Preprocessing %s" % dataset)
        preprocess_func[dataset](**args)
