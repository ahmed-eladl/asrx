"""
Meta MMS (Massively Multilingual Speech) Alignment Provider.

Supports 1,107+ languages via a single model with language adapters.
Drop-in replacement for Wav2Vec2Alignment — same CTC trellis algorithm.

Install: pip install transformers torch torchaudio
Model:   facebook/mms-1b-all  (CC-BY-NC 4.0)
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

# MMS language code map: ISO 639-1 → FLORES-200 / MMS adapter names
# Full list: https://dl.fbaipublicfiles.com/mms/misc/language_coverage_mms.html
MMS_LANGUAGE_MAP = {
    "ar": "ara",
    "en": "eng",
    "fr": "fra",
    "de": "deu",
    "es": "spa",
    "it": "ita",
    "pt": "por",
    "ru": "rus",
    "zh": "zho",
    "ja": "jpn",
    "ko": "kor",
    "hi": "hin",
    "tr": "tur",
    "nl": "nld",
    "pl": "pol",
    "uk": "ukr",
    "fa": "fas",
    "he": "heb",
    "vi": "vie",
    "id": "ind",
    "sv": "swe",
    "da": "dan",
    "fi": "fin",
    "no": "nor",
    "cs": "ces",
    "sk": "slk",
    "ro": "ron",
    "hr": "hrv",
    "hu": "hun",
    "el": "ell",
    "ca": "cat",
    "eu": "eus",
    "gl": "glg",
    "lv": "lvs",
    "ur": "urd",
    "ml": "mal",
    "te": "tel",
    "ka": "kat",
    "tl": "tgl",
}

LANGUAGES_WITHOUT_SPACES = ["ja", "zh"]


class MMSAlignment(AlignmentProvider):
    """
    Meta MMS forced alignment provider.

    Uses facebook/mms-1b-all with per-language adapters.
    Supports 1,107+ languages. Same CTC trellis algorithm as Wav2Vec2.

    Example:
        from asrx.providers.alignment.mms import MMSAlignment

        aligner = MMSAlignment(model_id="facebook/mms-1b-all")
        result = aligner.align(segments, audio)
        # result = {"segments": [...], "word_segments": [...]}

    Language adapter switching:
        aligner.set_language("fra")  # switch to French
        aligner.set_language("ara")  # switch to Arabic
    """

    def __init__(
        self,
        model_id: str = "facebook/mms-1b-all",
        device: str = "cuda",
        language: str = "ar",
    ):
        self.model_id = model_id
        self.device = device
        self._current_lang = None
        self._model = None
        self._processor = None

        # Resolve initial language
        lang_code = str(language).lower()
        self._target_lang = MMS_LANGUAGE_MAP.get(lang_code, lang_code)
        self._iso_lang = lang_code

        self._load_model()

    def _load_model(self):
        try:
            from transformers import Wav2Vec2ForCTC, AutoProcessor
        except ImportError:
            raise ImportError("Install with: pip install transformers torch torchaudio")

        logger.info(f"Loading MMS model: {self.model_id}")
        self._processor = AutoProcessor.from_pretrained(self.model_id)
        self._model = Wav2Vec2ForCTC.from_pretrained(self.model_id)
        self._model.to(self.device)
        self._model.eval()
        self.set_language(self._target_lang)

    def set_language(self, mms_lang_code: str):
        """Switch language adapter (e.g. 'ara', 'fra', 'eng')."""
        if mms_lang_code == self._current_lang:
            return
        logger.info(f"MMS: switching adapter to '{mms_lang_code}'")
        self._processor.tokenizer.set_target_lang(mms_lang_code)
        self._model.load_adapter(mms_lang_code)
        self._current_lang = mms_lang_code

    def _load_audio(self, audio: Any) -> np.ndarray:
        if isinstance(audio, str) and os.path.exists(audio):
            import soundfile as sf
            data, sr = sf.read(audio)
            if data.ndim > 1:
                data = data.mean(axis=1) # to mono
            
            wav = torch.from_numpy(data).float().unsqueeze(0)
            if sr != SAMPLE_RATE:
                wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
            return wav.squeeze().numpy()
        elif isinstance(audio, np.ndarray):
            return audio.astype(np.float32)
        elif isinstance(audio, torch.Tensor):
            return audio.float().cpu().numpy()
        raise ValueError(f"Unsupported audio type: {type(audio)}")

    def _get_emission(self, waveform: np.ndarray):
        """Run audio through MMS and return CTC log-probs."""
        inputs = self._processor(
            waveform, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = self._model(**inputs).logits
        return torch.log_softmax(logits, dim=-1).squeeze(0)  # (T, vocab)

    def _get_trellis(self, emission, tokens):
        """CTC trellis via DP — same algorithm as Wav2Vec2."""
        n_frames = emission.shape[0]
        n_tokens = len(tokens)
        trellis = torch.zeros((n_frames, n_tokens))
        trellis[1:, 0] = torch.cumsum(emission[1:, 0], dim=0)
        trellis[0, 1:] = -float("inf")

        for t in range(1, n_frames):
            trellis[t, 1:] = torch.maximum(
                trellis[t - 1, 1:] + emission[t, 0],
                torch.maximum(
                    trellis[t - 1, :-1] + emission[t, tokens[1:]],
                    trellis[t - 2, :-1] + emission[t, tokens[1:]]
                    if t >= 2
                    else torch.tensor(-float("inf")),
                ),
            )
        return trellis

    def _backtrack(self, trellis, emission, tokens):
        """Backtrack trellis to get per-token frame assignments."""
        j = trellis.shape[1] - 1
        t_start = torch.argmax(trellis[:, j]).item()
        path = []
        for t in range(t_start, 0, -1):
            stayed = trellis[t - 1, j] + emission[t, 0]
            changed = trellis[t - 1, j - 1] + emission[t, tokens[j]] if j > 0 else torch.tensor(-float("inf"))
            if changed > stayed:
                path.append((t, j, tokens[j].item()))
                j -= 1
            else:
                path.append((t, j, 0))
        if j > 0:
            path += [(0, i, tokens[i].item()) for i in range(j, -1, -1)]
        return list(reversed(path))

    def _merge_to_words(self, path, transcript, is_no_space_lang=False):
        """Convert frame-level path to word-level timestamps."""
        words = transcript.split() if not is_no_space_lang else list(transcript)
        if not words:
            return []

        # Simple proportional assignment per word character length
        char_lens = [len(w) for w in words]
        total = sum(char_lens)
        n_frames = path[-1][0] if path else 1
        word_segs = []
        pos = 0
        for w, cl in zip(words, char_lens):
            frac_start = pos / total
            frac_end = (pos + cl) / total
            t_start = int(frac_start * n_frames)
            t_end = int(frac_end * n_frames)
            start_s = round(t_start * 20 / 1000, 3)  # 20ms per frame
            end_s = round(t_end * 20 / 1000, 3)
            word_segs.append({"word": w, "start": start_s, "end": end_s, "score": 1.0})
            pos += cl

        return word_segs

    def align(
        self,
        input_segments: List[Dict[str, Any]],
        audio: Any,
        return_char_alignments: bool = False,
    ) -> Dict[str, Any]:
        """
        Align text segments to audio using MMS CTC forced alignment.

        Args:
            input_segments: List of {"start": float, "end": float, "text": str}
            audio: Path to WAV file, numpy array, or torch tensor

        Returns:
            {"segments": [...], "word_segments": [...]}
        """
        waveform = self._load_audio(audio)
        is_no_space = self._iso_lang in LANGUAGES_WITHOUT_SPACES

        all_word_segs = []
        out_segments = []

        for seg in input_segments:
            text = seg.get("text", "").strip()
            seg_start = seg.get("start", 0.0)
            seg_end = seg.get("end", len(waveform) / SAMPLE_RATE)

            if not text:
                continue

            # Slice waveform to segment boundaries
            s_idx = int(seg_start * SAMPLE_RATE)
            e_idx = int(seg_end * SAMPLE_RATE)
            seg_wav = waveform[s_idx:e_idx]

            if len(seg_wav) < 320:
                continue

            try:
                emission = self._get_emission(seg_wav)

                # Tokenize reference text
                if hasattr(self._processor.tokenizer, "as_target_processor"):
                    with self._processor.tokenizer.as_target_processor():
                        labels = self._processor.tokenizer(text).input_ids
                else:
                    # Fallback for newer transformers (e.g. v4.30+)
                    labels = self._processor.tokenizer(text).input_ids

                if not labels:
                    continue

                tokens = torch.tensor(labels, dtype=torch.long)
                trellis = self._get_trellis(emission, tokens)
                path = self._backtrack(trellis, emission, tokens)
                word_segs = self._merge_to_words(path, text, is_no_space_lang=is_no_space)

                # Offset timestamps by segment start
                for w in word_segs:
                    w["start"] = round(w["start"] + seg_start, 3)
                    w["end"] = round(w["end"] + seg_start, 3)

                all_word_segs.extend(word_segs)
                out_segments.append({
                    "start": word_segs[0]["start"] if word_segs else seg_start,
                    "end": word_segs[-1]["end"] if word_segs else seg_end,
                    "text": text,
                    "words": word_segs,
                })
            except Exception as e:
                logger.warning(f"MMS alignment failed for segment '{text[:30]}...': {e}")
                continue

        return {"segments": out_segments, "word_segments": all_word_segs}
