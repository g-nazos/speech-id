import os
import torch
import logging
import torchaudio
from torch.utils.data import Dataset, DataLoader
from speechbrain.inference.speaker import SpeakerRecognition
from speechbrain.utils.fetching import LocalStrategy
import configparser

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

config = configparser.ConfigParser()
config.read(os.path.join(project_root, "configs", "config.ini"))

# Ensure the logs directory exists
os.makedirs(os.path.join(project_root, "logs"), exist_ok=True)
log_filename = os.path.join(project_root, "logs", "running_logs.log")
logging.basicConfig(
    level=logging.INFO,  # Change to logging.DEBUG if you want extremely verbose output
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.FileHandler(log_filename), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class VoxCelebBatchLoader(Dataset):
    def __init__(
        self, datadir, sample_rate=16000, max_speakers=None, processed_files=None
    ):
        self.data = datadir
        self.sample_rate = sample_rate
        self.files = []

        logger.info("Loading data from directory %s" % datadir)

        unique_speakers = set()
        skipped_count = 0
        processed_set = processed_files if processed_files is not None else set()

        for root, dirs, files in os.walk(datadir):
            for file in files:
                if file.endswith(".wav"):
                    file_path = os.path.join(root, file)
                    if file_path in processed_set:
                        skipped_count += 1
                        continue
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

        # Sort by file size in descending order to verify VRAM limits
        self.files.sort(key=lambda x: os.path.getsize(x[0]), reverse=True)

        total_unique_loaded = len(set([s for _, s in self.files]))
        resume_msg = (
            f" (skipped {skipped_count} already processed files)"
            if skipped_count > 0
            else ""
        )
        logger.info(
            f"Found {len(self.files)} .wav files from {total_unique_loaded} speakers{resume_msg}."
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

            return waveform.squeeze(0), actual_sample_rate, speaker_id, file_path

        except Exception as e:
            logger.error(
                "Failed to load audio file %s with exception: %s", file_path, e
            )
            return None, None, speaker_id, file_path


def data_batch(batch):
    clean_batch = []
    for item in batch:
        if item[0] is not None:
            clean_batch.append(item)

    batch = clean_batch
    if len(batch) == 0:
        return None, None, None, None

    waveforms, sample_rates, speaker_ids, file_paths = zip(*batch)
    return waveforms, sample_rates, speaker_ids, file_paths


def main(
    split,
    batch_size=16,
    target_sample_rate=16000,
    data_loader_workers=0,
    max_speakers=None,
):
    datadir = os.path.join(project_root, "data", split)
    batch_size = batch_size
    sample_rate = target_sample_rate
    checkpoint_path = config["base"]["EMBEDDING_SAVE_DIR"]
    if not os.path.isabs(checkpoint_path):
        checkpoint_path = os.path.join(project_root, checkpoint_path)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        logger.warning("CUDA device not available, using CPU instead")

    logger.info("Starting embedding extraction")

    # Initialize checkpoint / resume state
    processed_files = set()
    all_embeddings = []
    all_speakers = []

    if os.path.exists(checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            if isinstance(checkpoint, dict) and "processed_files" in checkpoint:
                processed_files = set(checkpoint["processed_files"])
                all_embeddings = [checkpoint["embeddings"]]
                all_speakers = checkpoint["speakers"]
                logger.info(
                    f"Resuming from checkpoint: loaded {len(processed_files)} already processed embeddings."
                )
            else:
                logger.warning(
                    f"Checkpoint file at {checkpoint_path} is of an older version. It will be overwritten."
                )
        except Exception as e:
            logger.warning(
                f"Failed to load checkpoint file at {checkpoint_path} with exception: {e}. It will be overwritten."
            )

    dataset = VoxCelebBatchLoader(
        datadir, sample_rate, max_speakers=max_speakers, processed_files=processed_files
    )

    if len(dataset) == 0:
        logger.info("All files in the dataset have already been processed!")
        if all_embeddings:
            logger.info("Master database is already complete.")
        return

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=data_batch,
        pin_memory=True,
        num_workers=data_loader_workers,
    )

    logger.info("Loading the pre-trained Speechbrain ECAPA-TDNN model")
    model = SpeakerRecognition.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=os.path.join(
            project_root, "pretrained_models", "spkrec-ecapa-voxceleb"
        ),
        run_opts={"device": device},
        local_strategy=LocalStrategy.COPY,
    )

    def encode_waveforms(waveforms_list):
        lengths = torch.tensor([w.shape[0] for w in waveforms_list], device=device)
        padded = torch.nn.utils.rnn.pad_sequence(waveforms_list, batch_first=True)
        rel_lens = lengths / padded.shape[1]
        try:
            embs = model.encode_batch(wavs=padded, wav_lens=rel_lens)
            return embs.squeeze(1)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            if len(waveforms_list) == 1:
                logger.warning(
                    f"Single waveform of length {waveforms_list[0].shape[0]} OOM'd on GPU. "
                    "Running CPU fallback with no trimming..."
                )
                try:
                    wf_cpu = waveforms_list[0].cpu().unsqueeze(0)
                    rel_lens_cpu = torch.tensor([1.0], device="cpu")

                    model.to("cpu")
                    embs = model.encode_batch(wavs=wf_cpu, wav_lens=rel_lens_cpu)
                    model.to(device)

                    return embs.squeeze(1).to(device)
                except Exception as ex:
                    logger.error(f"Failed to encode on CPU fallback: {ex}")
                    try:
                        model.to(device)
                    except Exception:
                        pass
                    raise ex

            mid = len(waveforms_list) // 2
            logger.warning(
                f"OOM encountered on GPU. Splitting batch of size {len(waveforms_list)} into two sub-batches of size {mid} and {len(waveforms_list) - mid}."
            )
            emb1 = encode_waveforms(waveforms_list[:mid])
            emb2 = encode_waveforms(waveforms_list[mid:])
            return torch.cat([emb1, emb2], dim=0)

    with torch.no_grad():
        for batch_idx, batch_data in enumerate(dataloader):
            waveforms, sample_rates, speaker_ids, file_paths = batch_data

            if waveforms is None:
                continue

            # Resample waveforms using cuda
            resampled_waveforms = []
            for wf, sr in zip(waveforms, sample_rates):
                # non_blocking=True utilizes the pinned memory
                wf_cuda = wf.to(device, non_blocking=True)

                if sr != sample_rate:
                    wf_cuda = torchaudio.functional.resample(
                        wf_cuda, orig_freq=sr, new_freq=sample_rate
                    )
                resampled_waveforms.append(wf_cuda)

            # Embedding extraction with zero-trimming CPU fallback
            embeddings = encode_waveforms(resampled_waveforms)

            all_embeddings.append(embeddings.cpu())
            all_speakers.extend(speaker_ids)
            processed_files.update(file_paths)

            # Periodically save checkpoints every 500 batches
            if (batch_idx + 1) % 500 == 0 or (batch_idx + 1) == len(dataloader):
                logger.info(
                    f"Saving checkpoint at batch {batch_idx + 1}/{len(dataloader)}..."
                )
                curr_embeddings = torch.cat(all_embeddings, dim=0)
                torch.save(
                    {
                        "embeddings": curr_embeddings,
                        "speakers": all_speakers,
                        "processed_files": list(processed_files),
                    },
                    checkpoint_path,
                )
                all_embeddings = [curr_embeddings]

                torch.cuda.empty_cache()
                import gc

                gc.collect()

            # extreme logging logger log logic
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(dataloader):
                logger.info(f"Processed batch {batch_idx + 1}/{len(dataloader)}")

    if all_embeddings:
        all_embeddings = torch.cat(all_embeddings, dim=0)
        logger.info(
            f"Extraction complete. Total embeddings shape: {all_embeddings.shape}"
        )

        torch.save(
            {
                "embeddings": all_embeddings,
                "speakers": all_speakers,
                "processed_files": list(processed_files),
            },
            checkpoint_path,
        )
        logger.info(f"Successfully saved Master database to: {checkpoint_path}")
    else:
        logger.warning("Pipeline finished but no embeddings were extracted.")


if __name__ == "__main__":
    main(
        split="vox1_dev_wav",
        batch_size=16,
        target_sample_rate=16000,
        max_speakers=None,
        data_loader_workers=4,
    )
