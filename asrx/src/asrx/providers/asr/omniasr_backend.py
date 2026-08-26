from typing import Any, Dict, List, Optional, Union
import logging
from ...interfaces import ASRProvider

logger = logging.getLogger(__name__)

# Mapping standard ISO / common language codes to FLORES-200 / NLLB language codes used by OmniASR
OMNI_LANGUAGE_MAP = {
    "ar": "ara_Arab",
    "arabic": "ara_Arab",
    "en": "eng_Latn",
    "english": "eng_Latn",
    "fr": "fra_Latn",
    "french": "fra_Latn",
    "de": "deu_Latn",
    "german": "deu_Latn",
    "es": "spa_Latn",
    "spanish": "spa_Latn",
    "it": "ita_Latn",
    "italian": "ita_Latn",
    "ja": "jpn_Jpan",
    "japanese": "jpn_Jpan",
    "zh": "zho_Hans",
    "chinese": "zho_Hans",
    "ru": "rus_Cyrl",
    "russian": "rus_Cyrl",
    "pt": "por_Latn",
    "portuguese": "por_Latn",
    "tr": "tur_Latn",
    "turkish": "tur_Latn",
    "hi": "hin_Deva",
    "hindi": "hin_Deva",
    "ko": "kor_Hang",
    "korean": "kor_Hang",
}

class OmniASRProvider(ASRProvider):
    """
    ASR Provider for Meta's Omnilingual ASR models (omniASR_LLM_1B, 3B, 7B, omniASR_CTC_1B, etc.)
    via the `omnilingual-asr` library.
    """
    def __init__(self, model_id: str = "omniASR_LLM_1B_v2", device: str = "cuda"):
        self.model_id = model_id
        self.device = device
        
        # Clean model name (strip org if provided e.g. omnilingual-asr/omniASR_LLM_1B -> omniASR_LLM_1B_v2)
        card = model_id.split("/")[-1]
        if not card.endswith("_v2") and "omniASR" in card:
            card = f"{card}_v2"
        self.model_card = card
        
        logger.info(f"Initializing OmniASR with model_card='{self.model_card}'...")
        try:
            from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline
            self.pipeline = ASRInferencePipeline(model_card=self.model_card)
        except Exception as e:
            logger.warning(f"Failed to load '{self.model_card}', trying fallback without '_v2': {e}")
            from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline
            self.model_card = self.model_card.replace("_v2", "")
            self.pipeline = ASRInferencePipeline(model_card=self.model_card)

    def transcribe(self, audio: Any, language: Optional[str] = None) -> Union[str, List[Dict[str, Any]], Dict[str, Any]]:
        import os
        import tempfile
        import soundfile as sf
        import numpy as np

        # Resolve audio path
        temp_audio_path = None
        if isinstance(audio, str) and os.path.exists(audio):
            audio_path = audio
        elif isinstance(audio, np.ndarray):
            temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            sf.write(temp_file.name, audio, 16000)
            temp_audio_path = temp_file.name
            audio_path = temp_audio_path
        else:
            raise ValueError(f"Unsupported audio type: {type(audio)}")

        try:
            # Map language to NLLB code
            lang_key = str(language).lower() if language else "arabic"
            nllb_lang = OMNI_LANGUAGE_MAP.get(lang_key, "ara_Arab")
            
            # Run inference
            results = self.pipeline.transcribe([audio_path], lang=[nllb_lang], batch_size=1)
            
            if isinstance(results, list) and len(results) > 0:
                text = results[0]
            else:
                text = str(results)
                
            return text.strip()
        finally:
            if temp_audio_path and os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)
