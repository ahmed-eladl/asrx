from typing import Any, Dict, List, Optional
import os
import torch
import soundfile as sf
import librosa
import numpy as np
import tempfile
import logging
from ...interfaces import ASRProvider

logger = logging.getLogger(__name__)

class QwenASRProvider(ASRProvider):
    """
    Native ASR Provider for Qwen3-ASR and Audar-ASR models.
    Supports official `qwen_asr.Qwen3ASRModel` wrapper.
    """
    
    def __init__(self, model_id: str = "audarai/Audar-ASR-V1-Turbo", device: str = "cuda"):
        self.model_id = model_id
        self.device = device
        self.device_map = "cuda:0" if device == "cuda" and torch.cuda.is_available() else "cpu"
        
        try:
            from qwen_asr import Qwen3ASRModel
            self.model = Qwen3ASRModel.from_pretrained(
                self.model_id,
                dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
                device_map=self.device_map,
                max_inference_batch_size=1,
                max_new_tokens=256,
            )
            logger.info(f"Loaded Qwen/Audar ASR model {model_id} via Qwen3ASRModel.")
        except Exception as e:
            logger.error(f"Failed to load Qwen3ASRModel: {e}")
            raise e

    def transcribe(self, audio: Any, language: Optional[str] = "Arabic") -> List[Dict[str, Any]]:
        # Handle string audio filepath vs numpy array / tensor
        tmp_file = None
        if isinstance(audio, str):
            audio_path = audio
        else:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                if isinstance(audio, torch.Tensor):
                    audio = audio.detach().cpu().numpy()
                sf.write(tmp.name, audio, 16000)
                audio_path = tmp.name
                tmp_file = tmp.name

        try:
            lang_list = [language] if language else ["Arabic"]
            results = self.model.transcribe(audio=[audio_path], language=lang_list)
            
            text = results[0].text if results else ""
            
            return [{
                "text": text.strip(),
                "start": 0.0,
                "end": 0.0
            }]
        finally:
            if tmp_file and os.path.exists(tmp_file):
                os.remove(tmp_file)
