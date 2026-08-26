from typing import Any, Dict, List
import logging
import torch
from ...interfaces import VADProvider

logger = logging.getLogger(__name__)

class SileroVAD(VADProvider):
    """Voice Activity Detection using Silero VAD."""
    
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        # Load the Silero VAD model from torch hub
        self.model, self.utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            onnx=False,
            trust_repo=True
        )
        (self.get_speech_timestamps, _, self.read_audio, *_) = self.utils
        logger.info("Loaded Silero VAD model.")

    def _load_audio(self, audio: Any) -> torch.Tensor:
        """Robust audio loading bypassing torchaudio.load() issues in newer versions."""
        import os
        import numpy as np
        import soundfile as sf
        import torchaudio
        
        if isinstance(audio, str) and os.path.exists(audio):
            data, sr = sf.read(audio)
            if data.ndim > 1:
                data = data.mean(axis=1) # to mono
            wav = torch.from_numpy(data).float()
            if sr != 16000:
                wav = torchaudio.functional.resample(wav, sr, 16000)
            return wav
        elif isinstance(audio, np.ndarray):
            wav = torch.from_numpy(audio).float()
            # Assuming 16k if ndarray provided directly
            return wav
        elif isinstance(audio, torch.Tensor):
            return audio.float()
        raise ValueError(f"Unsupported audio type: {type(audio)}")

    def detect(self, audio: Any) -> List[Dict[str, float]]:
        # audio should be path or numpy array (16kHz). 
        wav = self._load_audio(audio)
        
        speech_timestamps = self.get_speech_timestamps(wav, self.model, sampling_rate=16000, threshold=self.threshold)
        
        results = []
        for ts in speech_timestamps:
            results.append({
                "start": ts['start'] / 16000.0,
                "end": ts['end'] / 16000.0
            })
        return results
