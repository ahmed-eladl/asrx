from typing import Any, Dict, List, Optional, Union
import logging
import os
import soundfile as sf
import numpy as np
import torch

from .interfaces import AlignmentProvider, DiarizationProvider, VADProvider


logger = logging.getLogger(__name__)

LANGUAGE_CODE_MAP = {
    "arabic": "ar",
    "english": "en",
    "french": "fr",
    "german": "de",
    "spanish": "es",
    "italian": "it",
    "japanese": "ja",
    "chinese": "zh",
    "portuguese": "pt",
    "russian": "ru",
    "turkish": "tr",
    "hindi": "hi",
    "korean": "ko",
}

class AlignmentPipeline:
    """
    Core ASRX Alignment & Segmentation Pipeline.
    
    Takes pre-computed text (from ANY external model, API, or human transcript) and raw audio,
    and produces millisecond-level word timestamps, confidence scores, VAD-guided segments,
    and speaker diarization labels.
    """
    def __init__(
        self,
        language: str = "ar",
        aligner: Optional[AlignmentProvider] = None,
        vad: Optional[VADProvider] = None,
        diarizer: Optional[DiarizationProvider] = None,
        device: str = "cuda",
    ):
        self.language = language
        self.device = device
        
        # 1. Initialize Aligner
        if aligner is not None:
            self.aligner = aligner
        else:
            lang_code = LANGUAGE_CODE_MAP.get(str(language).lower(), language)
            self.aligner = Wav2Vec2Alignment(model_id=lang_code, device=device)
            
        # 2. VAD & Diarization
        self.vad = vad
        self.diarizer = diarizer

    def _get_audio_duration(self, audio: Any) -> float:
        if isinstance(audio, str) and os.path.exists(audio):
            info = sf.info(audio)
            return float(info.duration)
        elif isinstance(audio, np.ndarray):
            return float(len(audio) / 16000.0)
        elif isinstance(audio, torch.Tensor):
            return float(audio.shape[-1] / 16000.0)
        return 0.0

    def _cluster_words_into_segments(
        self, 
        words: List[Dict[str, Any]], 
        speech_segments: Optional[List[Dict[str, float]]] = None,
        max_pause_threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        """Groups aligned words into natural utterance segments."""
        if not words:
            return []

        if speech_segments and len(speech_segments) > 0:
            clusters: List[List[Dict[str, Any]]] = [[] for _ in speech_segments]
            for w in words:
                w_mid = (w.get("start", 0.0) + w.get("end", 0.0)) / 2.0
                for idx, vad_seg in enumerate(speech_segments):
                    if (vad_seg["start"] - 0.1) <= w_mid <= (vad_seg["end"] + 0.1):
                        clusters[idx].append(w)
                        break

            final_segments = []
            for cluster in clusters:
                if cluster:
                    seg_start = cluster[0].get("start", 0.0)
                    seg_end = cluster[-1].get("end", 0.0)
                    seg_text = " ".join([w.get("word", "") for w in cluster]).strip()
                    final_segments.append({
                        "start": round(seg_start, 2),
                        "end": round(seg_end, 2),
                        "text": seg_text,
                        "words": cluster
                    })
            if final_segments:
                return final_segments

        # Pause-based clustering fallback
        segments = []
        current_cluster = [words[0]]
        
        for i in range(1, len(words)):
            prev_word = words[i - 1]
            curr_word = words[i]
            
            gap = curr_word.get("start", 0.0) - prev_word.get("end", 0.0)
            if gap > max_pause_threshold:
                seg_start = current_cluster[0].get("start", 0.0)
                seg_end = current_cluster[-1].get("end", 0.0)
                seg_text = " ".join([w.get("word", "") for w in current_cluster]).strip()
                segments.append({
                    "start": round(seg_start, 2),
                    "end": round(seg_end, 2),
                    "text": seg_text,
                    "words": current_cluster
                })
                current_cluster = [curr_word]
            else:
                current_cluster.append(curr_word)

        if current_cluster:
            seg_start = current_cluster[0].get("start", 0.0)
            seg_end = current_cluster[-1].get("end", 0.0)
            seg_text = " ".join([w.get("word", "") for w in current_cluster]).strip()
            segments.append({
                "start": round(seg_start, 2),
                "end": round(seg_end, 2),
                "text": seg_text,
                "words": current_cluster
            })

        return segments

    def align(
        self, 
        audio: Any, 
        text: Union[str, List[Dict[str, Any]]],
        language: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Aligns text to audio waveform and returns structured output.
        """
        lang = language or self.language
        lang_code = LANGUAGE_CODE_MAP.get(str(lang).lower(), str(lang).lower())
        audio_dur = self._get_audio_duration(audio)

        # 1. Detect VAD speech segments if active
        speech_segments = None
        if self.vad:
            try:
                speech_segments = self.vad.detect(audio)
            except Exception as e:
                logger.warning(f"VAD detection failed: {e}")

        # 2. Format input text into segments structure for aligner
        if isinstance(text, str):
            input_segments = [{"start": 0.0, "end": round(audio_dur, 2), "text": text.strip()}]
        elif isinstance(text, list):
            input_segments = text
        else:
            input_segments = [{"start": 0.0, "end": round(audio_dur, 2), "text": str(text).strip()}]

        # 3. Run Forced Alignment
        aligned_result = self.aligner.align(input_segments, audio)
        
        segments = []
        word_segments = []
        if isinstance(aligned_result, dict):
            segments = aligned_result.get("segments", [])
            word_segments = aligned_result.get("word_segments", [])
        elif isinstance(aligned_result, list):
            word_segments = aligned_result

        # 4. VAD Segment Clustering
        if self.vad and word_segments:
            clustered = self._cluster_words_into_segments(word_segments, speech_segments=speech_segments)
            if clustered:
                segments = clustered

        # 5. Diarization (Optional)
        if self.diarizer:
            try:
                from .providers.diarization.pyannote import assign_word_speakers
                diarize_df = self.diarizer.diarize(audio)
                transcript_dict = {"segments": segments, "word_segments": word_segments}
                assigned = assign_word_speakers(diarize_df, transcript_dict)
                segments = assigned.get("segments", segments)
                word_segments = assigned.get("word_segments", word_segments)
            except Exception as e:
                logger.warning(f"Diarization failed: {e}")
        else:
            for seg in segments:
                if "speaker" not in seg:
                    seg["speaker"] = "SPEAKER_00"
                for w in seg.get("words", []):
                    if "speaker" not in w:
                        w["speaker"] = seg["speaker"]
            for w in word_segments:
                if "speaker" not in w:
                    w["speaker"] = "SPEAKER_00"

        output_preview = " ".join([s["text"] for s in segments if s.get("text")]).strip()
        if not output_preview and isinstance(text, str):
            output_preview = text.strip()

        return {
            "segments": segments,
            "word_segments": word_segments,
            "language": lang_code,
            "output_preview": output_preview
        }
