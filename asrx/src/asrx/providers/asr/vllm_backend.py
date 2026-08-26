from typing import Any, Dict, List, Optional, Union
import logging
import base64
import os
import requests
from ...interfaces import ASRProvider

logger = logging.getLogger(__name__)

class VLLMASRProvider(ASRProvider):
    """
    Connects to any local or remote server running vLLM, SGLang, TGI, or an OpenAI-compatible audio API.
    
    You do NOT need to load model weights in Python memory.
    Just pass the server URL (e.g. "http://localhost:8000/v1")!
    
    Example:
        asr = VLLMASRProvider(base_url="http://localhost:8000/v1")
        pipeline = UniversalPipeline(asr_provider=asr, alignment_provider=aligner)
    """
    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        api_key: Optional[str] = "EMPTY",
        model: Optional[str] = None,
        timeout: int = 120,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "EMPTY")
        self.model = model
        self.timeout = timeout
        
        # If model is not given, auto-query /v1/models from the server
        if not self.model:
            try:
                resp = requests.get(
                    f"{self.base_url}/models",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    timeout=5
                )
                if resp.status_code == 200:
                    models_data = resp.json().get("data", [])
                    if models_data:
                        self.model = models_data[0].get("id")
                        logger.info(f"Auto-detected model on vLLM server: '{self.model}'")
            except Exception as e:
                logger.debug(f"Could not auto-fetch models list from {self.base_url}: {e}")

    def transcribe(self, audio: Any, language: Optional[str] = None) -> Union[str, List[Dict[str, Any]], Dict[str, Any]]:
        # Check if server supports standard /audio/transcriptions endpoint
        audio_path = None
        temp_created = False
        
        if isinstance(audio, str) and os.path.exists(audio):
            audio_path = audio
        elif hasattr(audio, "read"):
            # File-like object
            pass
            
        headers = {"Authorization": f"Bearer {self.api_key}"}
        
        # Try /v1/audio/transcriptions (standard OpenAI API format)
        if audio_path:
            with open(audio_path, "rb") as f:
                files = {"file": (os.path.basename(audio_path), f, "audio/wav")}
                data = {"model": self.model or "default"}
                if language:
                    data["language"] = language
                    
                resp = requests.post(
                    f"{self.base_url}/audio/transcriptions",
                    headers=headers,
                    files=files,
                    data=data,
                    timeout=self.timeout
                )
                
                if resp.status_code == 200:
                    res_json = resp.json()
                    return res_json.get("text", res_json)
                else:
                    raise RuntimeError(f"vLLM server error ({resp.status_code}): {resp.text}")
                    
        raise ValueError(f"Unsupported audio input for vLLM: {type(audio)}")
