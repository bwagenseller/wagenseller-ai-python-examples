"""
Voice embeddings for speaker identification (see speaker_id.py): one vector per chunk of speech, from pyannote's
speaker-embedding model. pyannote ships with WhisperX (the 'stt' conda environment), so this needs no new packages.

The default model, pyannote/wespeaker-voxceleb-resnet34-LM, is the one pyannote's speaker-diarization-community-1
pipeline uses inside. Diarization itself is not used: each chunk is one utterance, so the model is run once over the
whole chunk, which is much faster than segmenting and clustering.

pyannote's models are hosted on Hugging Face and some are gated. The token is read from the HF_TOKEN environment
variable (the name huggingface_hub itself reads); it is never stored in the repository or a config file.
"""

import os
from typing import List, Optional

# Keep CUDA's GPU numbering in slot order before torch is imported (see whisperx.py for why).
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

import numpy as np
import torch

from amadeo_utils.ai.asr.speaker_id import DEFAULT_EMBEDDING_MODEL

# The sample rate the embedding model expects (and the whole ASR pipeline uses)
SAMPLE_RATE = 16000


class SpeakerEmbedder:
    """
    Loads a pyannote speaker-embedding model once and turns speech into embeddings.

    Not thread-safe on the GPU by itself: the ASR server calls it while holding its GPU lock.
    """

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL, device: str = 'cpu', token: Optional[str] = None):
        """
        Args:
            model_name: the Hugging Face id of a pyannote embedding model.
            device: a torch device string ('cpu', 'cuda:1', ...).
            token: the Hugging Face token; None reads HF_TOKEN from the environment (and a model that is not gated
                needs none).
        """
        # Imported here so a caller that never builds an embedder does not pay for pyannote's import
        from pyannote.audio import Inference, Model

        self.model_name = model_name
        self.device = torch.device(device)
        token = token if token is not None else os.environ.get('HF_TOKEN')
        model = Model.from_pretrained(model_name, token=token)
        if model is None:
            # pyannote returns None (after logging why) instead of raising when a gated model is refused
            raise RuntimeError(f"Could not load speaker-embedding model '{model_name}'. If it is gated on Hugging Face, "
                               f"accept its terms and set HF_TOKEN.")
        # window='whole': one embedding for the whole chunk, rather than one per sliding window
        self.inference = Inference(model, window='whole')
        self.inference.to(self.device)

    def embed(self, audio: np.ndarray) -> List[float]:
        """
        One embedding for a chunk of speech.

        Args:
            audio: mono float32 samples in [-1, 1] at 16 kHz (SAMPLE_RATE).

        Returns:
            List[float]: the embedding (not normalised; speaker_id normalises before comparing).
        """
        waveform = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32))[None, :]
        with torch.inference_mode():
            vector = self.inference({'waveform': waveform, 'sample_rate': SAMPLE_RATE})
        return np.asarray(vector, dtype=np.float64).reshape(-1).tolist()


def resample(audio: np.ndarray, from_rate: int, to_rate: int = SAMPLE_RATE) -> np.ndarray:
    """
    Resamples mono audio (band-limited, via torchaudio), e.g. a 24 kHz WAV to the pipeline's 16 kHz.

    Args:
        audio: mono float32 samples.
        from_rate: its sample rate.
        to_rate: the rate wanted.

    Returns:
        np.ndarray: float32 samples at to_rate (the input itself if the rates already match).
    """
    if from_rate == to_rate:
        return audio
    import torchaudio.functional as F
    return F.resample(torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32)), from_rate, to_rate).numpy()
