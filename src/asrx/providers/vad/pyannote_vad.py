"""
Pyannote Voice Activity Detection (VAD) Provider.

Highly accurate neural VAD from the Pyannote audio framework.
Often outperforms Silero in noisy environments.

Install: pip install pyannote.audio
Model:   pyannote/voice-activity-detection
"""

from typing import Any, Dict, List, Optional
import logging
import os
import torch
import numpy as np

from ...interfaces import VADProvider

logger = logging.getLogger(__name__)


class PyannoteVAD(VADProvider):
    """
    Pyannote VAD provider.

    Requires a HuggingFace access token as the model is gated.
    Be sure to accept the user conditions at:
    https://huggingface.co/pyannote/voice-activity-detection

    Example:
        from asrx.providers.vad.pyannote_vad import PyannoteVAD
        
        vad = PyannoteVAD(use_auth_token="hf_...")
        segments = vad.detect("audio.wav")
    """

    def __init__(self, use_auth_token: Optional[str] = None, device: str = None):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.use_auth_token = use_auth_token
        self._pipeline = None
        self._load_model()

    def _load_model(self):
        try:
            from pyannote.audio import Pipeline
        except ImportError:
            raise ImportError(
                "pyannote.audio not installed. Install with: pip install pyannote.audio"
            )

        if not self.use_auth_token:
            logger.warning(
                "Pyannote VAD requires a HuggingFace token. "
                "Set hf_token=... or HF_TOKEN env var."
            )

        token = self.use_auth_token or os.environ.get("HF_TOKEN")
        
        try:
            logger.info("Loading Pyannote VAD pipeline...")
            self._pipeline = Pipeline.from_pretrained(
                "pyannote/voice-activity-detection",
                use_auth_token=token
            )
            self._pipeline.to(self.device)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load Pyannote VAD (did you accept the HF terms?): {e}"
            )

    def detect(self, audio: Any) -> List[Dict[str, float]]:
        """
        Detect speech segments in audio.
        Returns a list of dicts: [{"start": 0.0, "end": 2.5}, ...]
        """
        import torchaudio

        if isinstance(audio, str) and os.path.exists(audio):
            input_audio = audio
        elif isinstance(audio, torch.Tensor) or isinstance(audio, np.ndarray):
            # Pyannote pipeline expects either a file path or a dict:
            # {"waveform": (channel, time), "sample_rate": 16000}
            if isinstance(audio, np.ndarray):
                waveform = torch.from_numpy(audio).float()
            else:
                waveform = audio.float()
                
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)
            input_audio = {"waveform": waveform, "sample_rate": 16000}
        else:
            raise ValueError(f"Unsupported audio type for Pyannote VAD: {type(audio)}")

        logger.info("Running Pyannote VAD...")
        vad_result = self._pipeline(input_audio)

        segments = []
        for speech_turn in vad_result.get_timeline().support():
            segments.append({
                "start": round(speech_turn.start, 3),
                "end": round(speech_turn.end, 3)
            })

        return segments
