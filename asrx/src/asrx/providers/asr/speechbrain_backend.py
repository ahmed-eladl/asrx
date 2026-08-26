from typing import Any, Dict, List
import logging
import os
import torch
from ...interfaces import ASRProvider

logger = logging.getLogger(__name__)

class SpeechBrainASRProvider(ASRProvider):
    """Wrapper for SpeechBrain ASR models (e.g., asafaya hubert, speechbrain wav2vec2)."""
    
    def __init__(self, model_id: str, device: str = None):
        try:
            from speechbrain.inference.ASR import EncoderASR, EncoderDecoderASR
        except ImportError:
            try:
                from speechbrain.pretrained import EncoderASR, EncoderDecoderASR
            except ImportError:
                raise ImportError("Please install SpeechBrain: pip install speechbrain")
            
        self.device = "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"
        
        logger.info(f"Loading SpeechBrain model: {model_id} onto {self.device}")
        
        import sys
        sys.modules['speechbrain.lobes.models.huggingface_wav2vec'] = __import__('speechbrain.lobes.models.huggingface_wav2vec', fromlist=[''])
        sys.modules['speechbrain.lobes.models.huggingface_wav2vec.huggingface'] = sys.modules['speechbrain.lobes.models.huggingface_wav2vec']
        
        # Determine whether it's an Encoder-only (CTC) or Encoder-Decoder model
        try:
            self.model = EncoderASR.from_hparams(source=model_id, run_opts={"device": self.device})
        except Exception:
            try:
                self.model = EncoderDecoderASR.from_hparams(source=model_id, run_opts={"device": self.device})
            except Exception as e:
                raise RuntimeError(f"Could not load SpeechBrain model {model_id}: {e}")

    def transcribe(self, audio: Any, language: str = None) -> List[Dict[str, Any]]:
        import soundfile as sf
        import tempfile
        import os
        
        # SpeechBrain typically requires an audio path for its built-in transcribe_file()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
            
        try:
            sf.write(tmp_path, audio, 16000)
            text = self.model.transcribe_file(tmp_path)
            
            return [{"text": text, "start": 0.0, "end": 0.0}]
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
