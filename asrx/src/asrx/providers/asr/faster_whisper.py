from typing import Any, Dict, List
import logging
from ...interfaces import ASRProvider

logger = logging.getLogger(__name__)

class FasterWhisperASR(ASRProvider):
    """Wrapper for faster-whisper CTranslate2 models."""
    
    def __init__(self, model_size_or_path: str = "large-v3", device: str = None, compute_type: str = "float16"):
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            raise ImportError("Please install faster-whisper: pip install faster-whisper")
            
        self.model = WhisperModel(model_size_or_path, device=device, compute_type=compute_type)
        logger.info(f"Loaded faster-whisper model: {model_size_or_path}")

    def transcribe(self, audio: Any, language: str = None) -> List[Dict[str, Any]]:
        # faster-whisper accepts file path, bytes, or numpy array
        segments, info = self.model.transcribe(audio, language=language, beam_size=5)
        
        results = []
        for segment in segments:
            results.append({
                "text": segment.text.strip(),
                "start": segment.start,
                "end": segment.end
            })
        return results
