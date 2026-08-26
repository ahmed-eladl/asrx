from typing import Any, Dict, List, Optional, Union
import logging
import soundfile as sf
import numpy as np
import os
import torch

from .interfaces import ASRProvider, VADProvider, AlignmentProvider, DiarizationProvider

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

class UniversalPipeline:
    """
    Universal ASR & Alignment Pipeline combining:
    1. VAD (Voice Activity Detection)
    2. ASR (Speech-to-Text Transcription)
    3. Forced Alignment (Word-level timestamps via Wav2Vec2)
    4. Diarization (Speaker identification via Pyannote)
    
    Always outputs the standardized schema:
    {
      "segments": [
        {
          "start": float,
          "end": float,
          "text": str,
          "speaker": Optional[str],
          "words": [
            {"word": str, "start": float, "end": float, "score": float, "speaker": Optional[str]}
          ]
        }
      ],
      "word_segments": [
        {"word": str, "start": float, "end": float, "score": float, "speaker": Optional[str]}
      ],
      "language": str,
      "output_preview": str
    }
    """
    def __init__(
        self,
        asr_provider: ASRProvider,
        vad_provider: Optional[VADProvider] = None,
        alignment_provider: Optional[AlignmentProvider] = None,
        diarization_provider: Optional[DiarizationProvider] = None,
    ):
        self.asr = asr_provider
        self.vad = vad_provider
        self.aligner = alignment_provider
        self.diarizer = diarization_provider

    def _get_audio_info(self, audio: Any) -> float:
        """Returns audio duration in seconds."""
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
        """
        Groups aligned words into natural sentence/utterance segments based on VAD intervals and pause gaps.
        """
        if not words:
            return []

        # If VAD intervals exist, group words whose midpoints fall into the same VAD chunk
        if speech_segments and len(speech_segments) > 0:
            clusters: List[List[Dict[str, Any]]] = [[] for _ in speech_segments]
            unmatched: List[Dict[str, Any]] = []

            for w in words:
                w_mid = (w.get("start", 0.0) + w.get("end", 0.0)) / 2.0
                matched = False
                for idx, vad_seg in enumerate(speech_segments):
                    # Add a small 0.1s margin to VAD boundary
                    if (vad_seg["start"] - 0.1) <= w_mid <= (vad_seg["end"] + 0.1):
                        clusters[idx].append(w)
                        matched = True
                        break
                if not matched:
                    unmatched.append(w)

            # Build segments from non-empty clusters
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

        # Fallback / Pause-based clustering (splits on silence gap > max_pause_threshold)
        segments = []
        current_cluster = [words[0]]
        
        for i in range(1, len(words)):
            prev_word = words[i - 1]
            curr_word = words[i]
            
            gap = curr_word.get("start", 0.0) - prev_word.get("end", 0.0)
            if gap > max_pause_threshold:
                # Close current segment
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

    def process(self, audio: Any, language: Optional[str] = "en") -> Dict[str, Any]:
        """
        Runs the full ASR pipeline on the given audio and returns structured segments & word timestamps.
        """
        lang_str = str(language).lower() if language else "en"
        lang_code = LANGUAGE_CODE_MAP.get(lang_str, lang_str)
        audio_duration = self._get_audio_info(audio)
        
        # 1. VAD Detection (Optional)
        speech_segments = None
        if self.vad:
            try:
                speech_segments = self.vad.detect(audio)
                logger.info(f"VAD detected {len(speech_segments)} active speech intervals.")
            except Exception as e:
                logger.warning(f"VAD detection failed: {e}. Falling back to single chunk.")
                speech_segments = None
                
        # 2. ASR Transcription
        raw_asr_results = self.asr.transcribe(audio, language=language)
        
        # Normalize raw ASR results into standard segments list
        segments = []
        if isinstance(raw_asr_results, str):
            segments = [{
                "start": 0.0,
                "end": round(audio_duration, 2),
                "text": raw_asr_results.strip(),
                "words": []
            }]
        elif isinstance(raw_asr_results, list):
            for seg in raw_asr_results:
                if isinstance(seg, dict):
                    text = seg.get("text", "").strip()
                    start = float(seg.get("start", 0.0))
                    end = float(seg.get("end", audio_duration))
                    if end == 0.0 and audio_duration > 0:
                        end = round(audio_duration, 2)
                    segments.append({
                        "start": round(start, 2),
                        "end": round(end, 2),
                        "text": text,
                        "words": seg.get("words", [])
                    })
        elif isinstance(raw_asr_results, dict):
            text = raw_asr_results.get("text", "").strip()
            segments = [{
                "start": float(raw_asr_results.get("start", 0.0)),
                "end": float(raw_asr_results.get("end", audio_duration)),
                "text": text,
                "words": raw_asr_results.get("words", [])
            }]
            
        if not segments:
            segments = [{
                "start": 0.0,
                "end": round(audio_duration, 2),
                "text": "",
                "words": []
            }]

        # 3. Forced Alignment (Word-level timestamps)
        word_segments = []
        if self.aligner:
            try:
                aligned_data = self.aligner.align(segments, audio)
                if isinstance(aligned_data, dict):
                    segments = aligned_data.get("segments", segments)
                    word_segments = aligned_data.get("word_segments", [])
                elif isinstance(aligned_data, list):
                    word_segments = aligned_data
            except Exception as e:
                logger.warning(f"Alignment failed: {e}. Falling back to unaligned words.")
                
        # If no aligner was used or alignment produced empty words, synthesize from segments
        if not word_segments:
            for seg in segments:
                if "words" in seg and seg["words"]:
                    word_segments.extend(seg["words"])
                else:
                    words_in_text = seg["text"].split()
                    if words_in_text:
                        num_w = len(words_in_text)
                        seg_dur = max(seg["end"] - seg["start"], 0.1)
                        w_dur = seg_dur / num_w
                        seg_words = []
                        for i, w in enumerate(words_in_text):
                            w_start = round(seg["start"] + i * w_dur, 2)
                            w_end = round(w_start + w_dur, 2)
                            seg_words.append({
                                "word": w,
                                "start": w_start,
                                "end": w_end,
                                "score": 1.0
                            })
                        seg["words"] = seg_words
                        word_segments.extend(seg_words)

        # 4. Re-cluster segments based on VAD speech intervals (if VAD is active)
        if self.vad and word_segments:
            clustered_segments = self._cluster_words_into_segments(word_segments, speech_segments=speech_segments)
            if clustered_segments:
                segments = clustered_segments

        # 5. Speaker Diarization
        if self.diarizer:
            try:
                from .providers.diarization.pyannote import assign_word_speakers
                diarize_df = self.diarizer.diarize(audio)
                transcript_dict = {"segments": segments, "word_segments": word_segments}
                assigned = assign_word_speakers(diarize_df, transcript_dict)
                segments = assigned.get("segments", segments)
                word_segments = assigned.get("word_segments", word_segments)
            except Exception as e:
                logger.warning(f"Diarization failed: {e}.")
        else:
            # Default speaker if no diarizer is attached
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

        return {
            "segments": segments,
            "word_segments": word_segments,
            "language": lang_code,
            "output_preview": output_preview
        }
