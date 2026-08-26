import logging
from typing import Callable, Dict, Optional, Type, Union
from ...interfaces import ASRProvider
from .hf_transformers import HFTransformersASR
from .faster_whisper import FasterWhisperASR
from .custom import CustomASRProvider

logger = logging.getLogger(__name__)

# Registry for custom user-registered providers
_CUSTOM_PROVIDERS: Dict[str, Union[Type[ASRProvider], Callable[..., ASRProvider]]] = {}

def register_asr_provider(name: str, provider_cls_or_factory: Union[Type[ASRProvider], Callable[..., ASRProvider]]):
    """
    Registers a new ASR provider dynamically.
    Allows ANY new model or framework released in the future to be added with a single call.
    """
    _CUSTOM_PROVIDERS[name.lower()] = provider_cls_or_factory

def create_asr_provider(model_id: str, device: str = None, backend: Optional[str] = None, **kwargs) -> ASRProvider:
    """
    Universal Factory function that routes to the correct ASRProvider based on model ID, local path, server URL, or backend.
    """
    model_id_lower = str(model_id).lower()
    
    # 0. Check custom dynamic registry
    for key, provider in _CUSTOM_PROVIDERS.items():
        if key in model_id_lower or backend == key:
            logger.info(f"Routing to registered custom provider: '{key}'...")
            return provider(model_id=model_id, device=device, **kwargs)

    # 1. Faster-Whisper Routing
    if backend == "faster-whisper" or ("whisper" in model_id_lower and not "turbo" in model_id_lower and not "v3" in model_id_lower):
        logger.info("Routing to FasterWhisper backend...")
        try:
            return FasterWhisperASR(model_size_or_path=model_id, device=device)
        except Exception as e:
            logger.warning(f"FasterWhisper failed to load, falling back to HF Transformers: {e}")

    # 2. NVIDIA NeMo Routing
    if any(x in model_id_lower for x in ["nvidia", "nemotron", "parakeet", "conformer"]):
        logger.info("Routing to NeMo backend...")
        from .nemo_backend import NeMoASRProvider
        return NeMoASRProvider(model_id=model_id, device=device)
        
    # 3. SpeechBrain Routing
    if any(x in model_id_lower for x in ["speechbrain", "asafaya"]):
        logger.info("Routing to SpeechBrain backend...")
        from .speechbrain_backend import SpeechBrainASRProvider
        return SpeechBrainASRProvider(model_id=model_id, device=device)

    # 4. Audio LLMs (Voxtral, Gemma Audio, etc.)
    if any(x in model_id_lower for x in ["voxtral", "gemma", "vibe"]):
        logger.info("Routing to LLM Audio backend...")
        from .llm_backend import LLMASRProvider
        return LLMASRProvider(model_id=model_id, device=device)
        
    # 5. Standard Universal Hugging Face Pipeline / Local Folder Path
    logger.info("Routing to universal Hugging Face Pipeline backend...")
    return HFTransformersASR(model_id=model_id, device=device)
