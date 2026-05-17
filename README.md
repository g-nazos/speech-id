# speech-id

Speaker identification using the VoxCeleb dataset and SpeechBrain's ECAPA-TDNN model. Extracts fixed-size speaker embeddings from raw audio, which can then be used for speaker recognition or clustering.

## How it works

`scripts/embeding_extractor.py` walks a VoxCeleb-structured data directory, loads `.wav` files in batches, resamples to 16 kHz on CUDA if needed, and encodes each clip into a 192-dim embedding via the pretrained `speechbrain/spkrec-ecapa-voxceleb` model. All embeddings and their speaker labels are saved to a single `.pt` file.

## Requirements

- Python 3.13+
- CUDA-capable GPU (recommended; falls back to CPU)
- [uv](https://docs.astral.sh/uv/) package manager

## Setup

```bash
# Clone the repo
git clone <repo-url>
cd speech-id

# Create virtual environment and install dependencies
uv sync

# Install dev dependencies (black, pre-commit)
uv sync --group dev
pre-commit install
```

## Data

Download the [VoxCeleb1](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html) dev or test set and extract it under `data/`:

```
data/
  vox1_dev_wav/
    wav/
      id10032/
        <video_id>/
          00001.wav
          ...
```

## Configuration

Edit `configs/config.ini` to set where embeddings are saved:

```ini
[base]
EMBEDDING_SAVE_DIR = voxceleb_embeddings.pt
```

## Usage

```bash
# Run from the project root
uv run python scripts/embeding_extractor.py
```

The script defaults to `split=vox1_dev_wav`, `batch_size=16`, `target_sample_rate=16000`, `max_speakers=100`. To change these, edit the `main(...)` call at the bottom of the script.

The pretrained model is downloaded automatically on first run to `pretrained_models/spkrec-ecapa-voxceleb/`.

Logs are written to `scripts/logs/running_logs.log` and stdout.

## Output

A `.pt` file (path set in config) containing:

```python
{
    "embeddings": torch.Tensor,  # shape (N, 192)
    "speakers":   list[str],     # length N, e.g. ["id10032", ...]
}
```
