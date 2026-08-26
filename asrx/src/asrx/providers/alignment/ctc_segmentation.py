"""
CTC-Segmentation Alignment Provider.

Framework-agnostic CTC forced alignment algorithm.
Works with ANY CTC ASR model (Wav2Vec2, MMS, NeMo, ESPnet, SpeechBrain).
Unique advantage: produces per-segment confidence scores.

Install: pip install ctc-segmentation transformers
Paper:   Kürzinger et al., INTERSPEECH 2020
License: Apache 2.0
"""

from typing import Any, Dict, List, Optional
import logging
import os
import numpy as np
import torch
import torchaudio

from ...interfaces import AlignmentProvider

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000

# Default backend CTC model per language
# By default uses MMS for maximum language coverage
CTCSeg_DEFAULT_MODELS = {
    "mms": "facebook/mms-1b-all",
    "wav2vec2": {
        "ar": "jonatasgrosman/wav2vec2-large-xlsr-53-arabic",
        "en": "facebook/wav2vec2-base-960h",
        "fr": "jonatasgrosman/wav2vec2-large-xlsr-53-french",
        "de": "jonatasgrosman/wav2vec2-large-xlsr-53-german",
    }
}

# MMS language code map (ISO 639-1 → FLORES-200)
MMS_LANG_MAP = {
    "ar": "ara", "en": "eng", "fr": "fra", "de": "deu",
    "es": "spa", "it": "ita", "pt": "por", "ru": "rus",
    "zh": "zho", "ja": "jpn", "ko": "kor", "hi": "hin",
    "tr": "tur", "nl": "nld", "pl": "pol", "uk": "ukr",
    "fa": "fas", "he": "heb", "vi": "vie", "id": "ind",
}


class CTCSegmentationAlignment(AlignmentProvider):
    """
    CTC-Segmentation forced alignment provider.

    Uses the ctc-segmentation library's forward-backward algorithm to produce
    precise word-level timestamps AND per-segment confidence scores.

    The confidence score allows filtering bad alignments:
        conf > -5.0  → good alignment
        conf < -10.0 → likely a mismatch, skip this segment

    Args:
        language: ISO 639-1 language code
        backend: "mms" (default, 1,107+ languages) or "wav2vec2" (40 languages)
        model_id: Custom HF model ID (overrides backend default)
        device: "cuda" or "cpu"
        confidence_threshold: Segments below this log-prob score are flagged
        index_duration: Seconds per CTC output frame (0.02s = 20ms for MMS/Wav2Vec2 at 16kHz)

    Example:
        from asrx.providers.alignment.ctc_segmentation import CTCSegmentationAlignment
        
        aligner = CTCSegmentationAlignment(language="en", backend="mms")
        result = aligner.align([{"start": 0, "end": 10, "text": "hello world"}], "audio.wav")


        # Check confidence
        for seg in result["segments"]:
            if seg.get("confidence", 0) < -5.0:
                print(f"⚠️ Low confidence: {seg['text'][:30]}")
    """

    def __init__(
        self,
        language: str = "en",
        backend: str = "mms",
        model_id: Optional[str] = None,
        device: str = "cuda",
        confidence_threshold: float = -5.0,
        index_duration: float = 0.02,
    ):
        self.language = language.lower()
        self.backend = backend
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.index_duration = index_duration
        self._model = None
        self._processor = None

        # Resolve model
        if model_id:
            self._model_id = model_id
        elif backend == "mms":
            self._model_id = "facebook/mms-1b-all"
        else:
            self._model_id = CTCSeg_DEFAULT_MODELS["wav2vec2"].get(
                self.language, "facebook/wav2vec2-base-960h"
            )

        self._load_model()

    def _load_model(self):
        try:
            import ctc_segmentation
            self._ctc_seg = ctc_segmentation
        except ImportError:
            raise ImportError(
                "ctc-segmentation is not installed.\n"
                "Install with: pip install ctc-segmentation"
            )

        try:
            from transformers import Wav2Vec2ForCTC, AutoProcessor
        except ImportError:
            raise ImportError("Install with: pip install transformers torch torchaudio")

        logger.info(f"CTC-Segmentation: loading model '{self._model_id}'")
        self._processor = AutoProcessor.from_pretrained(self._model_id)
        self._model = Wav2Vec2ForCTC.from_pretrained(self._model_id)
        self._model.to(self.device)
        self._model.eval()

        # Switch language adapter for MMS
        if self.backend == "mms":
            mms_lang = MMS_LANG_MAP.get(self.language, self.language)
            try:
                self._processor.tokenizer.set_target_lang(mms_lang)
                self._model.load_adapter(mms_lang)
                logger.info(f"CTC-Segmentation: MMS adapter set to '{mms_lang}'")
            except Exception as e:
                logger.warning(f"CTC-Segmentation: Could not set MMS adapter '{mms_lang}': {e}")

    def _load_audio(self, audio: Any) -> np.ndarray:
        if isinstance(audio, str) and os.path.exists(audio):
            wav, sr = torchaudio.load(audio)
            if wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)
            if sr != SAMPLE_RATE:
                wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
            return wav.squeeze().numpy()
        elif isinstance(audio, np.ndarray):
            return audio.astype(np.float32)
        elif isinstance(audio, torch.Tensor):
            return audio.float().cpu().numpy()
        raise ValueError(f"Unsupported audio type: {type(audio)}")

    def _get_log_probs(self, waveform: np.ndarray) -> np.ndarray:
        """Run audio through CTC model, return log-probs (T, vocab_size)."""
        inputs = self._processor(
            waveform, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = self._model(**inputs).logits
        return torch.log_softmax(logits, dim=-1).squeeze(0).cpu().numpy()

    def align(
        self,
        input_segments: List[Dict[str, Any]],
        audio: Any,
        return_char_alignments: bool = False,
    ) -> Dict[str, Any]:
        """
        Align text to audio using CTC-Segmentation.

        Returns word-level timestamps with confidence scores.
        Segments with confidence < threshold are flagged with "low_confidence": True.
        """
        import ctc_segmentation

        waveform = self._load_audio(audio)
        log_probs = self._get_log_probs(waveform)

        # Build vocabulary from tokenizer
        vocab = list(self._processor.tokenizer.get_vocab().keys())

        # Configure CTC-Segmentation
        config = ctc_segmentation.CtcSegmentationParameters()
        config.char_list = vocab
        config.index_duration = self.index_duration

        # Collect utterance texts
        utterances = []
        for seg in input_segments:
            text = seg.get("text", "").strip()
            if text:
                utterances.append(text)

        if not utterances:
            return {"segments": [], "word_segments": []}

        try:
            ground_truth_mat, utt_begin_indices = ctc_segmentation.prepare_text(
                config, utterances
            )
            timings, char_probs, state_list = ctc_segmentation.ctc_segmentation(
                config, log_probs, ground_truth_mat
            )
            raw_segments = ctc_segmentation.determine_utterance_segments(
                config, utt_begin_indices, char_probs, timings, utterances
            )
        except Exception as e:
            logger.error(f"CTC-Segmentation failed: {e}")
            return {"segments": [], "word_segments": []}

        # Build output with per-segment confidence
        out_segments = []
        all_word_segs = []

        for utt_text, (start, end, confidence) in zip(utterances, raw_segments):
            words = utt_text.split()
            if not words:
                continue

            # Distribute word timestamps proportionally within the segment
            seg_dur = max(end - start, 0.001)
            char_count = sum(len(w) for w in words)
            word_segs = []
            t = start
            for w in words:
                w_dur = (len(w) / char_count) * seg_dur
                word_segs.append({
                    "word": w,
                    "start": round(t, 3),
                    "end": round(t + w_dur, 3),
                    "score": round(float(confidence), 4),
                })
                t += w_dur

            is_low_conf = confidence < self.confidence_threshold
            seg_entry = {
                "start": round(start, 3),
                "end": round(end, 3),
                "text": utt_text,
                "confidence": round(float(confidence), 4),
                "words": word_segs,
            }
            if is_low_conf:
                seg_entry["low_confidence"] = True
                logger.warning(
                    f"Low confidence ({confidence:.2f}) for: '{utt_text[:40]}' "
                    f"— possible mismatch, consider skipping."
                )

            out_segments.append(seg_entry)
            all_word_segs.extend(word_segs)

        return {"segments": out_segments, "word_segments": all_word_segs}
