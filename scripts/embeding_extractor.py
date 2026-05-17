import os
import torch
import logging
import torchaudio
from pre_commit.commands.clean import clean
from speechbrain.inference import speaker
from torch.backends.quantized import engine
from torch.utils.data import Dataset, DataLoader
from speechbrain.inference.speaker import SpeakerRecognition

log_filename = os.path.join("logs", "running_logs.log")
logging.basicConfig(
    level=logging.INFO,  # Change to logging.DEBUG if you want extremely verbose output
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.FileHandler(log_filename), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class VoxCelebBatchLoader(Dataset):
    def __init__(self, datadir, sample_rate=16000):
        self.data = datadir
        self.sample_rate = sample_rate
        self.files = []

        logger.info("Loading data from directory %s" % datadir)
        for root, dirs, files in os.walk(datadir):
            for file in files:
                if file.endswith(".wav"):
                    file_path = os.path.join(root, file)
                    speaker_id = os.path.basename(
                        os.path.dirname(os.path.dirname(file_path))
                    )
                    self.files.append((file_path, speaker_id))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path, speaker_id = self.files[idx]

        try:
            waveform, actual_sample_rate = torchaudio.load(file_path)

            if waveform.shape[0] > 1:
                logger.warning("Audio file %s has more than one channel" % file_path)
                pass

            if actual_sample_rate != self.sample_rate:
                logger.warning(
                    "Sampling rate for %s is different than expected. Sample rate: %s",
                    file_path,
                    actual_sample_rate,
                )
                # resample = torchaudio.transforms.Resample(orig_freq=actual_sample_rate, new_freq=self.sample_rate)
                # waveform = resample(waveform) # i removed these to perform resampling in cuda

            return waveform.squeeze(0), actual_sample_rate, speaker_id

        except Exception as e:
            logger.error(
                "Failed to load audio file %s with exception: %s", file_path, e
            )
            return None, None, speaker_id


def data_batch(batch):
    clean_batch = []
    for item in batch:
        if item[0] is not None:
            clean_batch.append(item)

    batch = clean_batch
    if len(batch) == 0:
        return None, None, None

    waveforms, sample_rates, speaker_ids = zip(*batch)
    return waveforms, sample_rates, speaker_ids
