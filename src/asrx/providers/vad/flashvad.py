from typing import Any, Dict, List, Optional
import logging
import torch
import numpy as np
import soundfile as sf
import io
import os

from ...interfaces import VADProvider

logger = logging.getLogger(__name__)


class FlashVAD(VADProvider):
    """
    FlashVAD Voice Activity Detection (oss-codes/flashvad).
    
    Ultra-lightweight ONNX model (~46K params). Emits speech probability every 10ms.
    Designed for low-latency voice agent pipelines (LiveKit, Pipecat, etc.)
    
    Requires: pip install git+https://github.com/oss-codes/flashvad.git
    
    Example:
        from asrx.providers.vad.flashvad import FlashVAD
        vad = FlashVAD(threshold=0.5)
        segments = vad.detect("audio.wav")
    """

    def __init__(
        self,
        threshold: float = 0.5,
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 100,
        sample_rate: int = 16000,
    ):
        self.threshold = threshold
        self.min_speech_duration_ms = min_speech_duration_ms
        self.min_silence_duration_ms = min_silence_duration_ms
        self.sample_rate = sample_rate
        self._model = None
        self._load_model()

    def _load_model(self):
        try:
            from flashvad import FlashVADModel
            self._model = FlashVADModel()
            logger.info("FlashVAD model loaded successfully.")
        except ImportError:
            raise ImportError(
                "FlashVAD is not installed. Install with:\n"
                "  pip install git+https://github.com/oss-codes/flashvad.git"
            )

    def _load_audio(self, audio: Any) -> np.ndarray:
        """Load and normalize audio to 16kHz mono float32 numpy array."""
        if isinstance(audio, str) and os.path.exists(audio):
            data, sr = sf.read(audio, dtype="float32")
            if data.ndim > 1:
                data = data.mean(axis=1)  # stereo -> mono
            if sr != self.sample_rate:
                import torchaudio
                tensor = torch.from_numpy(data).unsqueeze(0)
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)
                data = resampler(tensor).squeeze(0).numpy()
            return data
        elif isinstance(audio, np.ndarray):
            return audio.astype(np.float32)
        elif isinstance(audio, torch.Tensor):
            return audio.float().cpu().numpy()
        else:
            raise ValueError(f"Unsupported audio type: {type(audio)}")

    def detect(self, audio: Any) -> List[Dict[str, float]]:
        """
        Detect speech segments in audio.
        
        Returns:
            List of {"start": float, "end": float} dicts with speech segment boundaries in seconds.
        """
        wav = self._load_audio(audio)
        duration = len(wav) / self.sample_rate

        # FlashVAD processes 10ms frames
        frame_size_ms = 10
        frame_size_samples = int(self.sample_rate * frame_size_ms / 1000)
        n_frames = len(wav) // frame_size_samples

        min_speech_frames = self.min_speech_duration_ms // frame_size_ms
        min_silence_frames = self.min_silence_duration_ms // frame_size_ms

        # Get per-frame speech probabilities from FlashVAD
        probs = self._model.predict(wav)

        # Convert probabilities to binary speech/silence decisions
        is_speech = [p >= self.threshold for p in probs[:n_frames]]

        # Merge segments
        segments = []
        in_speech = False
        speech_start = 0
        silence_count = 0

        for i, speech in enumerate(is_speech):
            t = i * frame_size_ms / 1000.0

            if speech:
                if not in_speech:
                    speech_start = t
                    in_speech = True
                silence_count = 0
            else:
                if in_speech:
                    silence_count += 1
                    if silence_count >= min_silence_frames:
                        seg_end = (i - silence_count) * frame_size_ms / 1000.0
                        seg_dur_frames = (i - silence_count) - (speech_start / (frame_size_ms / 1000.0))
                        if seg_dur_frames >= min_speech_frames:
                            segments.append({
                                "start": round(speech_start, 3),
                                "end": round(seg_end, 3)
                            })
                        in_speech = False
                        silence_count = 0

        # Handle trailing speech segment
        if in_speech:
            segments.append({
                "start": round(speech_start, 3),
                "end": round(min(n_frames * frame_size_ms / 1000.0, duration), 3)
            })

        logger.info(f"FlashVAD detected {len(segments)} speech segments.")
        return segments
