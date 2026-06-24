import os
import sys
import logging
import argparse

import torch
import torchaudio
from torch.utils.data import DataLoader
from transformers import WavLMModel, WavLMForXVector, AutoFeatureExtractor

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "scripts"))

from embeding_extractor import VoxCelebBatchLoader, data_batch  # noqa: E402
from configs.models import get_model  # noqa: E402

logger = logging.getLogger("wavlm_extractor")

DATA_OUT_DIR = os.path.join(project_root, "data_base", "data")
TARGET_SR = 16000


def mean_pool(hidden, frame_mask):
    """Masked mean over the time axis. hidden: (B, T, H), frame_mask: (B, T)."""
    m = frame_mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * m).sum(dim=1) / m.sum(dim=1).clamp(min=1e-9)


def encode_waveforms(waveforms_16k, model, feature_extractor, device, head):
    """Encode a list of 16 kHz 1-D waveforms -> (B, dim), with OOM fallback.

    head="meanpool": mean-pool the 12 transformer layers (-> 768).
    head="xvector":  use WavLMForXVector's own TDNN+stats-pool head (-> 512).
    """
    inputs = feature_extractor(
        [w.numpy() for w in waveforms_16k],
        sampling_rate=TARGET_SR,
        return_tensors="pt",
        padding=True,
        return_attention_mask=True,
    )
    x = inputs["input_values"].to(device)
    mask = inputs["attention_mask"].to(device)

    try:
        if head == "xvector":
            out = model(input_values=x, attention_mask=mask)
            return out.embeddings.cpu()  # (B, xvector_output_dim) = 512
        out = model(input_values=x, attention_mask=mask, output_hidden_states=True)
        frame_mask = model._get_feature_vector_attention_mask(
            out.last_hidden_state.shape[1], attention_mask=mask
        )
        # drop hidden_states[0] (CNN feature projection); keep 12 transformer layers
        pooled = [mean_pool(h, frame_mask) for h in out.hidden_states[1:]]
        feats = torch.stack(pooled, dim=1).mean(dim=1)  # (B, 12, 768) -> (B, 768)
        return feats.cpu()
    except torch.OutOfMemoryError:
        torch.cuda.empty_cache()
        if len(waveforms_16k) == 1:
            logger.warning(
                "Single waveform of length %d OOM'd on GPU; CPU fallback.",
                waveforms_16k[0].shape[0],
            )
            model.to("cpu")
            try:
                emb = encode_waveforms(
                    waveforms_16k, model, feature_extractor, "cpu", head
                )
            finally:
                model.to(device)
            return emb
        mid = len(waveforms_16k) // 2
        logger.warning(
            "OOM on batch of %d; splitting into %d + %d.",
            len(waveforms_16k),
            mid,
            len(waveforms_16k) - mid,
        )
        left = encode_waveforms(
            waveforms_16k[:mid], model, feature_extractor, device, head
        )
        right = encode_waveforms(
            waveforms_16k[mid:], model, feature_extractor, device, head
        )
        return torch.cat([left, right], dim=0)


def main(
    split,
    model_name="wavlm",
    batch_size=8,
    data_loader_workers=4,
    max_seconds=20.0,
):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    spec = get_model(model_name)
    datadir = os.path.join(project_root, "data", split)
    out_name = spec.test_pt_file if "test" in split else spec.pt_file
    checkpoint_path = os.path.join(DATA_OUT_DIR, out_name)
    # Cap clip length: WavLM attention cost grows with sequence length, so the
    # long tail (~4% of clips > 20s, up to 145s) would OOM at any useful batch
    # size. 20s keeps 96% of clips whole. 0 disables the cap.
    max_samples = (
        int(max_seconds * TARGET_SR) if max_seconds and max_seconds > 0 else None
    )

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        logger.warning("CUDA device not available, using CPU instead")

    logger.info(
        "WavLM extraction | model=%s dim=%d split=%s cap=%ss -> %s",
        spec.name,
        spec.dim,
        split,
        max_seconds if max_samples else "none",
        checkpoint_path,
    )

    # Resume state
    processed_files = set()
    all_embeddings = []
    all_speakers = []
    if os.path.exists(checkpoint_path):
        try:
            ckpt = torch.load(checkpoint_path, map_location="cpu")
            if isinstance(ckpt, dict) and "processed_files" in ckpt:
                processed_files = set(ckpt["processed_files"])
                all_embeddings = [ckpt["embeddings"]]
                all_speakers = list(ckpt["speakers"])
                logger.info(
                    "Resuming: %d embeddings already extracted.", len(processed_files)
                )
        except Exception as e:
            logger.warning("Could not load checkpoint (%s); it will be overwritten.", e)

    dataset = VoxCelebBatchLoader(datadir, TARGET_SR, processed_files=processed_files)
    if len(dataset) == 0:
        logger.info("Nothing to do: all files already processed.")
        return

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=data_batch,
        pin_memory=True,
        num_workers=data_loader_workers,
    )

    logger.info("Loading WavLM model '%s' (head=%s)...", spec.source, spec.head)
    feature_extractor = AutoFeatureExtractor.from_pretrained(spec.source)
    if spec.head == "xvector":
        model = WavLMForXVector.from_pretrained(spec.source).to(device).eval()
        produced_dim = model.config.xvector_output_dim
    else:
        model = WavLMModel.from_pretrained(spec.source).to(device).eval()
        produced_dim = model.config.hidden_size
    if produced_dim != spec.dim:
        logger.warning(
            "Model output dim %d != registry dim %d for '%s'.",
            produced_dim,
            spec.dim,
            spec.name,
        )

    def save_checkpoint():
        embeddings = torch.cat(all_embeddings, dim=0)
        torch.save(
            {
                "embeddings": embeddings,
                "speakers": all_speakers,
                "processed_files": list(processed_files),
            },
            checkpoint_path,
        )
        return embeddings

    with torch.no_grad():
        for batch_idx, (waveforms, sample_rates, speaker_ids, file_paths) in enumerate(
            dataloader
        ):
            if waveforms is None:
                continue

            resampled = []
            for wf, sr in zip(waveforms, sample_rates):
                if sr != TARGET_SR:
                    wf = torchaudio.functional.resample(
                        wf, orig_freq=sr, new_freq=TARGET_SR
                    )
                if max_samples is not None and wf.shape[0] > max_samples:
                    wf = wf[:max_samples]
                resampled.append(wf)

            embeddings = encode_waveforms(
                resampled, model, feature_extractor, device, spec.head
            )

            all_embeddings.append(embeddings)
            all_speakers.extend(speaker_ids)
            processed_files.update(file_paths)
            # Compact buffers so the running tensor stays a single chunk.
            all_embeddings = [torch.cat(all_embeddings, dim=0)]

            if (batch_idx + 1) % 200 == 0 or (batch_idx + 1) == len(dataloader):
                save_checkpoint()
                logger.info(
                    "Checkpoint at batch %d/%d (%d embeddings).",
                    batch_idx + 1,
                    len(dataloader),
                    len(all_speakers),
                )
                if device.startswith("cuda"):
                    torch.cuda.empty_cache()

            if (batch_idx + 1) % 20 == 0:
                logger.info("Processed batch %d/%d", batch_idx + 1, len(dataloader))

    embeddings = save_checkpoint()
    logger.info(
        "Done. Saved %s embeddings of shape %s to %s",
        spec.name,
        tuple(embeddings.shape),
        checkpoint_path,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract WavLM speaker embeddings.")
    parser.add_argument("--split", default="vox1_dev_wav")
    parser.add_argument("--model", default="wavlm")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=20.0,
        help="Truncate clips longer than this (0 disables). Bounds WavLM memory.",
    )
    args = parser.parse_args()
    main(
        split=args.split,
        model_name=args.model,
        batch_size=args.batch_size,
        data_loader_workers=args.workers,
        max_seconds=args.max_seconds,
    )
