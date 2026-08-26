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

    def detect(self, audio: Any) -> List[Dict[str, float]]:
        # audio should be path or numpy array (16kHz). 
        # For simplicity, assuming audio is a path here
        wav = self.read_audio(audio, sampling_rate=16000)
        
        speech_timestamps = self.get_speech_timestamps(wav, self.model, sampling_rate=16000, threshold=self.threshold)
        
        results = []
        for ts in speech_timestamps:
            results.append({
                "start": ts['start'] / 16000.0,
                "end": ts['end'] / 16000.0
            })
        return results
