import os
import torch
import logging
import torchaudio
from torch.utils.data import Dataset, DataLoader
from speechbrain.inference.speaker import SpeakerRecognition
import configparser

config = configparser.ConfigParser()
config.read(os.path.join(os.getcwd(), "configs", "config.ini"))

log_filename = os.path.join("logs", "running_logs.log")
logging.basicConfig(
    level=logging.INFO,  # Change to logging.DEBUG if you want extremely verbose output
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.FileHandler(log_filename), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class VoxCelebBatchLoader(Dataset):
    def __init__(self, datadir, sample_rate=16000, max_speakers=None):
        self.data = datadir
        self.sample_rate = sample_rate
        self.files = []

        logger.info("Loading data from directory %s" % datadir)

        unique_speakers = set()

        for root, dirs, files in os.walk(datadir):
            for file in files:
                if file.endswith(".wav"):
                    file_path = os.path.join(root, file)
                    speaker_id = os.path.basename(
                        os.path.dirname(os.path.dirname(file_path))
                    )
                    if max_speakers:
                        if (
                            speaker_id not in unique_speakers
                            and len(unique_speakers) >= max_speakers
                        ):
                            continue
                        unique_speakers.add(speaker_id)
                    self.files.append((file_path, speaker_id))
        total_unique_loaded = len(set([s for _, s in self.files]))
        logger.info(
            f"Found {len(self.files)} .wav files from {total_unique_loaded} speakers."
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path, speaker_id = self.files[idx]

        try:
            waveform, actual_sample_rate = torchaudio.load(file_path)

            if waveform.shape[0] > 1:
                # TODO: Define behavior for  multichannel audio files
                logger.warning("Audio file %s has more than one channel" % file_path)
                pass

            if actual_sample_rate != self.sample_rate:
                # TODO Define behavior for sample rates (currently ddownsampling to 16k)
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


def main(split, batch_size=16, target_sample_rate=16000, max_speakers=None):
    datadir = os.path.join(os.getcwd(), "data", split)
    batch_size = batch_size
    sample_rate = target_sample_rate

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        logger.warning("CUDA device not available, using CPU instead")

    logger.info("Starting embedding extraction")

    dataset = VoxCelebBatchLoader(datadir, sample_rate, max_speakers=max_speakers)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=data_batch,
        pin_memory=True,
        num_workers=4,
    )

    logger.info("Loading the pre-trained Speechbrain ECAPA-TDNN model")
    model = SpeakerRecognition.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="pretrained_models/spkrec-ecapa-voxceleb",
        run_opts={"device": device},
    )

    all_embeddings = []
    all_speakers = []

    with torch.no_grad():
        for batch_idx, batch_data in enumerate(dataloader):
            waveforms, sample_rates, speaker_ids = batch_data

            if waveforms is None:
                continue

            # Resample not expected sample rates using cuda
            resampled_waveforms = []
            for wf, sr in zip(waveforms, sample_rates):
                # non_blocking=True utilizes the pinned memory
                wf_cuda = wf.to(device, non_blocking=True)

                if sr != sample_rate:
                    wf_cuda = torchaudio.functional.resample(
                        wf_cuda, orig_freq=sr, new_freq=sample_rate
                    )
                resampled_waveforms.append(wf_cuda)

            # Zero {adding for smaller audio files
            lengths = torch.tensor(
                [w.shape[0] for w in resampled_waveforms], device=device
            )
            padded_waveforms = torch.nn.utils.rnn.pad_sequence(
                resampled_waveforms, batch_first=True
            )
            relative_lengths = lengths / padded_waveforms.shape[1]

            # Embedding extraction
            embeddings = model.encode_batch(
                wavs=padded_waveforms, wav_lens=relative_lengths
            )
            embeddings = embeddings.squeeze(1)

            all_embeddings.append(embeddings.cpu())
            all_speakers.extend(speaker_ids)

            # extreme logging logger log logic
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(dataloader):
                logger.info(f"Processed batch {batch_idx + 1}/{len(dataloader)}")

    if all_embeddings:
        all_embeddings = torch.cat(all_embeddings, dim=0)
        logger.info(
            f"Extraction complete. Total embeddings shape: {all_embeddings.shape}"
        )

        torch.save(
            {"embeddings": all_embeddings, "speakers": all_speakers},
            config["base"]["EMBEDDING_SAVE_DIR"],
        )
        logger.info(
            f"Successfully saved Master database to: {config["base"]["EMBEDDING_SAVE_DIR"]}"
        )
    else:
        logger.warning("Pipeline finished but no embeddings were extracted.")


if __name__ == "__main__":
    main(
        split="vox1_dev_wav", batch_size=16, target_sample_rate=16000, max_speakers=100
    )
