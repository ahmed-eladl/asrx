from typing import Any, Callable, Dict, List, Optional, Union
from ...interfaces import ASRProvider

class CustomASRProvider(ASRProvider):
    """
    Universal Wrapper that adapts ANY custom function, SDK, or model object into an ASRProvider.
    
    Example:
        # Wrap any Python function
        my_asr = CustomASRProvider(lambda audio, language: my_model.generate(audio))
        
        # Use directly with UniversalPipeline
        pipeline = UniversalPipeline(asr_provider=my_asr, alignment_provider=aligner)
    """
    def __init__(self, transcribe_fn: Callable[..., Any]):
        self.transcribe_fn = transcribe_fn

    def transcribe(self, audio: Any, language: Optional[str] = None) -> Union[str, List[Dict[str, Any]], Dict[str, Any]]:
        return self.transcribe_fn(audio, language=language)
