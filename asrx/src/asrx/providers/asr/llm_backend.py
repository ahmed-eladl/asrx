from typing import Any, Dict, List
import logging
import torch
from ...interfaces import ASRProvider

logger = logging.getLogger(__name__)

class LLMASRProvider(ASRProvider):
    """Wrapper for bleeding edge multimodal LLMs (Qwen3, Audar, Gemma-4)."""
    
    def __init__(self, model_id: str, device: str = None):
        try:
            from transformers import AutoProcessor, AutoModelForCausalLM
        except ImportError:
            raise ImportError("Please install transformers.")
            
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        
        logger.info(f"Loading LLM ASR model: {model_id} onto {self.device}")
        
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id, 
                trust_remote_code=True,
                torch_dtype=self.dtype,
                device_map="auto"
            ).eval()
        except Exception:
            from transformers import AutoModelForSpeechSeq2Seq
            self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_id, 
                trust_remote_code=True,
                torch_dtype=self.dtype,
                device_map="auto"
            ).eval()

    def transcribe(self, audio: Any, language: str = None) -> List[Dict[str, Any]]:
        system_prompt = "Transcribe the following speech."
        if language == "ar":
            system_prompt = "فرّغ الكلام العربي التالي."
            
        try:
            conv = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [{"type": "audio"}]}
            ]
            text = self.processor.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
        except Exception:
            text = "<audio> Please transcribe this audio into text."
            
        inputs = self.processor(
            text=text, 
            audio=audio, 
            sampling_rate=16000, 
            return_tensors="pt"
        ).to(self.device)
        
        if "input_features" in inputs:
            inputs["input_features"] = inputs["input_features"].to(self.dtype)
            
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=440, do_sample=False)
            
        try:
            transcription = self.processor.batch_decode(
                out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
            )[0].strip()
        except Exception:
            transcription = self.processor.batch_decode(out, skip_special_tokens=True)[0].strip()
            
        return [{"text": transcription, "start": 0.0, "end": 0.0}]
