from typing import Any, Dict, List
import logging
from ...interfaces import ASRProvider

logger = logging.getLogger(__name__)

class HFTransformersASR(ASRProvider):
    """Generic wrapper for any Hugging Face automatic-speech-recognition pipeline (e.g. SeamlessM4T, Qwen-Audio, Whisper fine-tunes)."""
    
    def __init__(self, model_id: str, device: str = "cuda"):
        try:
            from transformers import pipeline
            import torch
        except ImportError:
            raise ImportError("Please install transformers: pip install transformers torch")
            
        device_id = 0 if device == "cuda" and torch.cuda.is_available() else -1
        kwargs = {
            "model": model_id,
            "device": device_id,
            "trust_remote_code": True
        }
        
        # Qwen and Audar crash with chunk_length_s due to tensor size mismatches
        if "qwen" not in model_id.lower() and "audar" not in model_id.lower():
            kwargs["chunk_length_s"] = 30
            
        self.pipe = pipeline(
            "automatic-speech-recognition",
            **kwargs
        )
        logger.info(f"Loaded HF transformers model: {model_id} (kwargs: {kwargs})")

    def transcribe(self, audio: Any, language: str = None) -> List[Dict[str, Any]]:
        generate_kwargs = {}
        if language:
            generate_kwargs["language"] = language
            
        def _run_pipe(ret_ts: bool, gen_kw: dict):
            kwargs = {}
            if ret_ts:
                kwargs["return_timestamps"] = True
            if gen_kw:
                kwargs["generate_kwargs"] = gen_kw
            return self.pipe(audio, **kwargs)

        try:
            outputs = _run_pipe(ret_ts=True, gen_kw=generate_kwargs)
        except Exception as e:
            err_str = str(e)
            # 1. Try without return_timestamps
            try:
                outputs = _run_pipe(ret_ts=False, gen_kw=generate_kwargs)
            except Exception as e2:
                # 2. Try without generate_kwargs (e.g. models that reject 'language')
                try:
                    outputs = _run_pipe(ret_ts=True, gen_kw={})
                except Exception as e3:
                    # 3. Try with neither
                    outputs = _run_pipe(ret_ts=False, gen_kw={})
        
        results = []
        if isinstance(outputs, dict) and "chunks" in outputs:
            for chunk in outputs["chunks"]:
                start, end = chunk["timestamp"]
                results.append({
                    "text": chunk["text"].strip(),
                    "start": start,
                    "end": end if end is not None else start + 1.0 # fallback
                })
        else:
            # Fallback when timestamps aren't returned
            text = outputs["text"] if isinstance(outputs, dict) else outputs
            results.append({
                "text": text,
                "start": 0.0,
                "end": 0.0
            })
            
        return results
