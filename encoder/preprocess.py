from multiprocess.pool import Pool  # ThreadPool
from encoder.params_data import *
from encoder.config import librispeech_datasets, anglophone_nationalites
from datetime import datetime
from encoder import audio
from pathlib import Path
from tqdm import tqdm
import numpy as np
from scipy.io.wavfile import write
import os


class DatasetLog:
    """
    Registers metadata about the dataset in a text file.
    """

    def __init__(self, root, name):
        self.text_file = open(Path(root, "Log_%s.txt" % name.replace("/", "_")), "w")
        self.sample_data = dict()

        start_time = str(datetime.now().strftime("%A %d %B %Y at %H:%M"))
        self.write_line("Creating dataset %s on %s" % (name, start_time))
        self.write_line("-----")
        self._log_params()

    def _log_params(self):
        from encoder import params_data
        self.write_line("Parameter values:")
        for param_name in (p for p in dir(params_data) if not p.startswith("__")):
            value = getattr(params_data, param_name)
            self.write_line("\t%s: %s" % (param_name, value))
        self.write_line("-----")

    def write_line(self, line):
        self.text_file.write("%s\n" % line)

    def add_sample(self, **kwargs):
        for param_name, value in kwargs.items():
            if not param_name in self.sample_data:
                self.sample_data[param_name] = []
            self.sample_data[param_name].append(value)

    def finalize(self):
        self.write_line("Statistics:")
        for param_name, values in self.sample_data.items():
            self.write_line("\t%s:" % param_name)
            self.write_line("\t\tmin %.3f, max %.3f" % (np.min(values), np.max(values)))
            self.write_line("\t\tmean %.3f, median %.3f" % (np.mean(values), np.median(values)))
        self.write_line("-----")
        end_time = str(datetime.now().strftime("%A %d %B %Y at %H:%M"))
        self.write_line("Finished on %s" % end_time)
        self.text_file.close()


def _init_preprocess_dataset(dataset_name, datasets_root, out_dir) -> (Path, DatasetLog):
    dataset_root = datasets_root.joinpath(dataset_name)
    if not dataset_root.exists():
        print("Couldn\'t find %s, skipping this dataset." % dataset_root)
        return None, None
    return dataset_root, DatasetLog(out_dir, dataset_name)


def _preprocess_speaker_dirs(speaker_dirs, dataset_name, datasets_root, out_dir, extension,
                             skip_existing, logger, num_processes=8, speaker_dir_2_depth=True, use_short_data=False):
    print("%s: Preprocessing data for %d speakers." % (dataset_name, len(speaker_dirs)))

    # Function to preprocess utterances for one speaker
    def preprocess_speaker(speaker_dir: Path):
        # Give a name to the speaker that includes its dataset
        speaker_name = "_".join(speaker_dir.relative_to(datasets_root).parts)

        # Create an output directory with that name, as well as a txt file containing a 
        # reference to each source file.
        speaker_out_dir = out_dir.joinpath(speaker_name)
        speaker_out_dir.mkdir(exist_ok=True)
        sources_fpath = speaker_out_dir.joinpath("_sources.txt")

        # There's a possibility that the preprocessing was interrupted earlier, check if 
        # there already is a sources file.
        if sources_fpath.exists():
            try:
                with sources_fpath.open("r") as sources_file:
                    existing_fnames = {line.split(",")[0] for line in sources_file}
            except:
                existing_fnames = {}
        else:
            existing_fnames = {}

        # Gather all audio files for that speaker recursively
        sources_file = sources_fpath.open("a" if skip_existing else "w")
        if speaker_dir_2_depth:
            in_fpath_list = speaker_dir.glob("**/*.%s" % extension)
        else:
            in_fpath_list = speaker_dir.glob("*.%s" % extension)
        for in_fpath in in_fpath_list:
            # Check if the target output file already exists
            out_fname = "_".join(in_fpath.relative_to(speaker_dir).parts)
            out_fname = out_fname.replace(".%s" % extension, ".npy")
            if skip_existing and out_fname in existing_fnames:
                continue

            # Load and preprocess the waveform
            wav = audio.preprocess_wav(in_fpath, extension=extension)
            if len(wav) == 0:
                continue

            # Create the mel spectrogram, discard those that are too short
            frames = audio.wav_to_mel_spectrogram(wav)
            if not use_short_data and len(frames) < partials_n_frames:
                continue

            out_fpath = speaker_out_dir.joinpath(out_fname)
            np.save(out_fpath, frames)
            logger.add_sample(duration=len(wav) / sampling_rate)
            sources_file.write("%s,%s\n" % (out_fname, in_fpath))

        sources_file.close()

    # Process the utterances for each speaker
    with Pool(num_processes) as pool:  # ThreadPool(8) as pool:
        # list(tqdm(pool.imap(preprocess_speaker, speaker_dirs), dataset_name, len(speaker_dirs),
        list(tqdm(pool.map(preprocess_speaker, speaker_dirs), dataset_name, len(speaker_dirs),
                  unit="speakers"))
    logger.finalize()
    print("Done preprocessing %s.\n" % dataset_name)


def preprocess_librispeech(datasets_root: Path, out_dir: Path, skip_existing=False, num_processes=8,
                           use_short_data=False):
    for dataset_name in librispeech_datasets["train"]["other"]:
        # Initialize the preprocessing
        dataset_root, logger = _init_preprocess_dataset(dataset_name, datasets_root, out_dir)
        if not dataset_root:
            return

            # Preprocess all speakers
        speaker_dirs = list(dataset_root.glob("*"))
        _preprocess_speaker_dirs(speaker_dirs, dataset_name, datasets_root, out_dir, "flac",
                                 skip_existing, logger, num_processes=num_processes, use_short_data=use_short_data)


def preprocess_librispeech_clean(datasets_root: Path, out_dir: Path, skip_existing=False, num_processes=8,
                                 use_short_data=False):
    for dataset_name in librispeech_datasets["train"]["clean"]:
        # Initialize the preprocessing
        dataset_root, logger = _init_preprocess_dataset(dataset_name, datasets_root, out_dir)
        if not dataset_root:
            return

            # Preprocess all speakers
        speaker_dirs = list(dataset_root.glob("*"))
        _preprocess_speaker_dirs(speaker_dirs, dataset_name, datasets_root, out_dir, "flac",
                                 skip_existing, logger, num_processes=num_processes, use_short_data=use_short_data)


def preprocess_voxceleb1(datasets_root: Path, out_dir: Path, skip_existing=False, num_processes=8,
                         use_short_data=False):
    # Initialize the preprocessing
    dataset_name = "VoxCeleb1"
    dataset_root, logger = _init_preprocess_dataset(dataset_name, datasets_root, out_dir)
    if not dataset_root:
        return

    # Get the contents of the meta file
    with dataset_root.joinpath("vox1_meta.csv").open("r") as metafile:
        metadata = [line.split("\t") for line in metafile][1:]

    # Select the ID and the nationality, filter out non-anglophone speakers
    nationalities = {line[0]: line[3] for line in metadata}
    keep_speaker_ids = [speaker_id for speaker_id, nationality in nationalities.items() if
                        nationality.lower() in anglophone_nationalites]
    print("VoxCeleb1: using samples from %d (presumed anglophone) speakers out of %d." %
          (len(keep_speaker_ids), len(nationalities)))

    # Get the speaker directories for anglophone speakers only
    speaker_dirs = dataset_root.joinpath("wav").glob("*")
    speaker_dirs = [speaker_dir for speaker_dir in speaker_dirs if
                    speaker_dir.name in keep_speaker_ids]
    print("VoxCeleb1: found %d anglophone speakers on the disk, %d missing (this is normal)." %
          (len(speaker_dirs), len(keep_speaker_ids) - len(speaker_dirs)))

    # Preprocess all speakers
    _preprocess_speaker_dirs(speaker_dirs, dataset_name, datasets_root, out_dir, "wav",
                             skip_existing, logger, num_processes=num_processes, use_short_data=use_short_data)


def preprocess_voxceleb2(datasets_root: Path, out_dir: Path, skip_existing=False, num_processes=8,
                         use_short_data=False):
    # Initialize the preprocessing
    dataset_name = "VoxCeleb2"
    dataset_root, logger = _init_preprocess_dataset(dataset_name, datasets_root, out_dir)
    if not dataset_root:
        return

    # Get the speaker directories
    # Preprocess all speakers
    speaker_dirs = list(dataset_root.joinpath("dev", "aac").glob("*"))
    _preprocess_speaker_dirs(speaker_dirs, dataset_name, datasets_root, out_dir, "m4a",
                             skip_existing, logger, num_processes=num_processes, use_short_data=use_short_data)


def preprocess_zeroth(datasets_root: Path, out_dir: Path, skip_existing=False, num_processes=8, use_short_data=False):
    dataset_name = "zeroth-korean"
    dataset_root, logger = _init_preprocess_dataset(dataset_name, datasets_root, out_dir)
    if not dataset_root:
        return

    # Get the speaker directories
    # Preprocess all speakers
    speaker_dirs = list(dataset_root.joinpath("train_data_01", "003").glob("*"))
    speaker_dirs += list(dataset_root.joinpath("test_data_01", "003").glob("*"))
    _preprocess_speaker_dirs(speaker_dirs, dataset_name, datasets_root, out_dir, "flac",
                             skip_existing, logger, num_processes=num_processes, speaker_dir_2_depth=False,
                             use_short_data=use_short_data)


def preprocess_speech_ko(datasets_root: Path, out_dir: Path, skip_existing=False, num_processes=8,
                         use_short_data=False):
    dataset_name = "speech_ko"
    dataset_root, logger = _init_preprocess_dataset(dataset_name, datasets_root, out_dir)
    if not dataset_root:
        return

    # Get the speaker directories
    # Preprocess all speakers
    speaker_dirs = [d for d in list(dataset_root.glob("*/")) if os.path.isdir(d)]
    _preprocess_speaker_dirs(speaker_dirs, dataset_name, datasets_root, out_dir, "wav",
                             skip_existing, logger, num_processes=num_processes, speaker_dir_2_depth=False,
                             use_short_data=use_short_data)


def preprocess_etri_8channel(datasets_root: Path, out_dir: Path, skip_existing=False, num_processes=8,
                             use_short_data=False):
    dataset_name = "etri_voice_dataset"
    dataset_root, logger = _init_preprocess_dataset(dataset_name, datasets_root, out_dir)
    if not dataset_root:
        return

    # Get the speaker directories
    # Preprocess all speakers
    tmp_speaker_dirs = list(dataset_root.joinpath("8channel").glob("*/"))
    new_path = dataset_root.joinpath("8channel_by_speaker")
    speaker_dirs = []
    if not new_path.is_dir():
        new_path.mkdir()
    for tmp_dir in tmp_speaker_dirs:
        if not os.path.isdir(tmp_dir):
            continue
        speaker_name = os.path.basename(tmp_dir)[5:9]
        new_speaker_path = new_path.joinpath(speaker_name)
        if not new_speaker_path.is_dir():
            new_speaker_path.mkdir()
            speaker_dirs.append(new_speaker_path)
        pcms = tmp_dir.glob("*.RAW")
        for pcm_path in pcms:
            write(os.path.join(new_speaker_path, os.path.splitext(os.path.basename(pcm_path))[0]) + ".wav", 16000,
                  np.memmap(pcm_path, dtype='h', mode='r'))

    _preprocess_speaker_dirs(speaker_dirs, dataset_name, datasets_root, out_dir, "wav",
                             skip_existing, logger, num_processes=num_processes, speaker_dir_2_depth=False,
                             use_short_data=use_short_data)


def preprocess_datatang(datasets_root: Path, out_dir: Path, skip_existing=False, num_processes=8, use_short_data=False):
    dataset_name = "datatang"
    dataset_root, logger = _init_preprocess_dataset(dataset_name, datasets_root, out_dir)
    if not dataset_root:
        return

    # Get the speaker directories
    # Preprocess all speakers
    speaker_dirs = [d for d in list(dataset_root.glob("*/")) if os.path.isdir(d)]
    _preprocess_speaker_dirs(speaker_dirs, dataset_name, datasets_root, out_dir, "wav",
                             skip_existing, logger, num_processes=num_processes, speaker_dir_2_depth=False,
                             use_short_data=use_short_data)


def preprocess_etri_etc(datasets_root: Path, out_dir: Path, skip_existing=False, num_processes=8,
                          use_short_data=False):
    dataset_name = "etri_voice_dataset"
    sub_datasets = ["child", "kr_en", "mobile", "telematics"]
    for sub_dataset_name in sub_datasets:
        # Initialize the preprocessing
        dataset_root, logger = _init_preprocess_dataset(dataset_name, datasets_root, out_dir)
        if not dataset_root:
            return

            # Preprocess all speakers
        speaker_dirs = list(dataset_root.joinpath(sub_dataset_name).glob("*"))
        _preprocess_speaker_dirs(speaker_dirs, dataset_name, datasets_root, out_dir, "pcm",
                                 skip_existing, logger, num_processes=num_processes, use_short_data=use_short_data)