from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    name: str  # registry key / CLI value
    source: str  # SpeechBrain / HuggingFace source id for the extractor
    dim: int  # embedding dimension -> VECTOR(dim) and DB schema
    schema: str  # Postgres schema isolating this model's tables
    pt_file: str  # enrollment .pt filename under data_base/data/
    test_pt_file: str  # held-out test/probe .pt filename under data_base/data/
    head: str = "meanpool"  # extraction method: "ecapa", "meanpool", or "xvector"


MODELS: dict[str, ModelSpec] = {
    # ecapa is the existing, already-populated model: it lives in the default
    # `public` schema so its current tables/data are reused untouched (no
    # migration, no re-populate). New models get their own schemas.
    "ecapa": ModelSpec(
        name="ecapa",
        source="speechbrain/spkrec-ecapa-voxceleb",
        dim=192,
        schema="public",
        pt_file="voxceleb_embeddings.pt",
        test_pt_file="voxceleb_test_embeddings.pt",
        head="ecapa",
    ),
    # wavlm: per-utterance embedding = mean-pool of the 12 transformer layers'
    # mean-pooled hidden states -> 768-dim. Uses the base WavLM (no SV head).
    "wavlm": ModelSpec(
        name="wavlm",
        source="microsoft/wavlm-base-plus",
        dim=768,
        schema="wavlm",
        pt_file="voxceleb_embeddings_wavlm.pt",
        test_pt_file="voxceleb_test_embeddings_wavlm.pt",
        head="meanpool",
    ),
    # wavlm_xvector: the VoxCeleb-fine-tuned x-vector speaker head. The model's
    # own TDNN + statistics pooling produces a 512-dim embedding (out.embeddings).
    # Trained for speaker discrimination -> the fair comparison vs ECAPA.
    "wavlm_xvector": ModelSpec(
        name="wavlm_xvector",
        source="microsoft/wavlm-base-plus-sv",
        dim=512,
        schema="wavlm_xvector",
        pt_file="voxceleb_embeddings_wavlm_xvector.pt",
        test_pt_file="voxceleb_test_embeddings_wavlm_xvector.pt",
        head="xvector",
    ),
}

DEFAULT_MODEL = "ecapa"


def get_model(name: str) -> ModelSpec:
    try:
        return MODELS[name]
    except KeyError:
        known = ", ".join(MODELS)
        raise KeyError(f"Unknown model '{name}'. Known models: {known}") from None
