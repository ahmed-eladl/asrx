from typing import Any, Dict, List
import logging
from ...interfaces import ASRProvider

logger = logging.getLogger(__name__)

class NeMoASRProvider(ASRProvider):
    """Wrapper for NVIDIA NeMo models (Parakeet, Conformer, Nemotron)."""
    
    def __init__(self, model_id: str, device: str = "cuda"):
        try:
            import nemo.collections.asr as nemo_asr
            import torch
        except ImportError:
            raise ImportError("Please install NeMo: pip install nemo_toolkit[asr]")
            
        self.device = device
        
        # Determine the class based on the model type (CTC or RNNT/EncDec)
        # Most Parakeet/Conformer open models are EncDecCTCModelBPE
        logger.info(f"Loading NeMo model: {model_id}")
        
        # Clean up the model ID if the user appended "(lm)" or "(greedy)"
        clean_model_id = model_id.split(" ")[0]
        
        try:
            self.model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(model_name=clean_model_id)
        except Exception:
            # Fallback for RNNT or character-based models
            try:
                self.model = nemo_asr.models.EncDecRNNTModel.from_pretrained(model_name=clean_model_id)
            except Exception as e:
                raise RuntimeError(f"Could not load NeMo model {clean_model_id}: {e}")

        if device == "cuda" and torch.cuda.is_available():
            self.model = self.model.cuda()
        else:
            self.model = self.model.cpu()
            
        self.model.eval()

    def transcribe(self, audio: Any, language: str = None) -> List[Dict[str, Any]]:
        import soundfile as sf
        import tempfile
        import os
        
        # NeMo typically requires audio paths for its built-in transcribe() batching
        # We will write the numpy array to a temp wav file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
            
        try:
            sf.write(tmp_path, audio, 16000)
            
            # NeMo transcribe returns a list of transcriptions (one per file)
            kwargs = {}
            if language:
                kwargs["language"] = language
                
            transcriptions = self.model.transcribe(paths2audio_files=[tmp_path], **kwargs)
            
            # NeMo returns either a list of strings, or a tuple (list_of_strings, logits/alignments)
            if isinstance(transcriptions, tuple):
                texts = transcriptions[0]
            else:
                texts = transcriptions
                
            text = texts[0] if (isinstance(texts, list) and len(texts) > 0) else str(texts)
                
            return [{"text": text, "start": 0.0, "end": 0.0}]
            
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
