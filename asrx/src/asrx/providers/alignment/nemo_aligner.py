"""
NeMo Forced Aligner (NFA) Provider.

Uses NVIDIA NeMo's Viterbi CTC forced alignment with Conformer/FastConformer models.
Produces token + word + segment level timestamps via the official NeMo aligner utilities.

Install: pip install "nemo_toolkit[asr]>=2.5.0"
Models:  stt_en_fastconformer_hybrid_large_pc  (English, best quality)
         stt_fr_fastconformer_ctc_large         (French)
         stt_ar_conformer_ctc_large             (Arabic)
         Any EncDecCTCModel or EncDecHybridRNNTCTCModel from NGC

Reference: https://github.com/NVIDIA-NeMo/Speech/tree/main/tools/nemo_forced_aligner
"""

from typing import Any, Dict, List, Optional
import logging
import os
import json
import tempfile
import numpy as np
import torch

from ...interfaces import AlignmentProvider

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000

# Default NeMo CTC models per language
NEMO_DEFAULT_MODELS = {
    "en": "stt_en_fastconformer_hybrid_large_pc",
    "fr": "stt_fr_fastconformer_ctc_large",
    "de": "stt_de_fastconformer_ctc_large",
    "es": "stt_es_fastconformer_ctc_large",
    "ru": "stt_ru_fastconformer_ctc_large",
    "zh": "stt_zh_citrinet_512",
    "ja": "stt_ja_fastconformer_ctc_large",
    "hi": "stt_hi_fastconformer_ctc_large",
    "ar": "stt_ar_conformer_ctc_large",
    "it": "stt_it_fastconformer_ctc_large",
    "ko": "stt_ko_fastconformer_ctc_large",
    "pl": "stt_pl_fastconformer_ctc_large",
    "uk": "stt_ua_fastconformer_hybrid_transducer_ctc",
}

LANGUAGES_WITHOUT_SPACES = ["ja", "zh"]


class NeMoForcedAligner(AlignmentProvider):
    """
    NeMo Forced Aligner (NFA) provider.

    Uses NVIDIA NeMo's Viterbi CTC forced alignment internally.
    Works with any EncDecCTCModel or EncDecHybridRNNTCTCModel.

    Example:
        from asrx.providers.alignment.nemo_aligner import NeMoForcedAligner

        # Auto-select NeMo model for language
        aligner = NeMoForcedAligner(language="en")

        # Or specify model explicitly
        aligner = NeMoForcedAligner(
            model_name="stt_en_fastconformer_hybrid_large_pc"
        )

        result = aligner.align(segments, audio)
    """

    def __init__(
        self,
        language: str = "en",
        model_name: Optional[str] = None,
        model_path: Optional[str] = None,
        device: str = "cuda",
        use_local_attention: bool = True,
    ):
        self.language = language.lower()
        self.device = device
        self.use_local_attention = use_local_attention
        self._model = None
        self._output_timestep_duration = None

        # Resolve model name
        if model_path:
            self.model_name = None
            self.model_path = model_path
        else:
            self.model_name = model_name or NEMO_DEFAULT_MODELS.get(
                self.language, "stt_en_fastconformer_hybrid_large_pc"
            )
            self.model_path = None

        self._load_model()

    def _load_model(self):
        try:
            import nemo.collections.asr as nemo_asr
            from nemo.collections.asr.models.ctc_models import EncDecCTCModel
            from nemo.collections.asr.models.hybrid_rnnt_ctc_models import EncDecHybridRNNTCTCModel
        except ImportError:
            raise ImportError(
                "NeMo Forced Aligner requires nemo_toolkit >= 2.5.0.\n"
                "Install with: pip install 'nemo_toolkit[asr]>=2.5.0'"
            )

        if self.model_path:
            logger.info(f"NeMo Forced Aligner: loading local model from {self.model_path}")
            self._model = nemo_asr.models.ASRModel.restore_from(self.model_path)
        else:
            logger.info(f"NeMo Forced Aligner: downloading model '{self.model_name}'")
            self._model = nemo_asr.models.ASRModel.from_pretrained(self.model_name)

        # If Hybrid RNNT-CTC, switch to CTC decoding mode (required for NFA)
        if hasattr(self._model, "change_decoding_strategy"):
            try:
                self._model.change_decoding_strategy(decoder_type="ctc")
                logger.info("NeMo: switched Hybrid model to CTC decoding mode.")
            except Exception:
                pass

        # Enable local attention to prevent OOM on long audio (Conformer models)
        if self.use_local_attention and hasattr(self._model, "change_attention_model"):
            try:
                self._model.change_attention_model(
                    self_attention_model="rel_pos_local_attn",
                    att_context_size=[64, 64]
                )
                logger.info("NeMo: enabled local attention (context=[64,64]).")
            except Exception:
                pass

        self._model.eval()

    def _audio_to_wav(self, audio: Any) -> str:
        """Returns path to a WAV file (creates temp file if needed)."""
        import soundfile as sf

        if isinstance(audio, str) and os.path.exists(audio):
            return audio
        elif isinstance(audio, np.ndarray):
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            sf.write(tmp.name, audio, samplerate=SAMPLE_RATE)
            return tmp.name
        elif isinstance(audio, torch.Tensor):
            data = audio.float().cpu().numpy()
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            sf.write(tmp.name, data, samplerate=SAMPLE_RATE)
            return tmp.name
        raise ValueError(f"Unsupported audio type: {type(audio)}")

    def align(
        self,
        input_segments: List[Dict[str, Any]],
        audio: Any,
        return_char_alignments: bool = False,
    ) -> Dict[str, Any]:
        """
        Align text segments to audio using NeMo Viterbi CTC forced alignment.

        Args:
            input_segments: List of {"start": float, "end": float, "text": str}
            audio: Path to WAV file, numpy array, or torch tensor

        Returns:
            {"segments": [...], "word_segments": [...]}
        """
        try:
            from nemo.collections.asr.parts.utils.aligner_utils import (
                get_batch_variables,
                viterbi_decoding,
                add_t_start_end_to_utt_obj,
            )
        except ImportError:
            raise ImportError(
                "NeMo aligner_utils not found. "
                "Upgrade to: pip install 'nemo_toolkit[asr]>=2.5.0'"
            )

        audio_path = self._audio_to_wav(audio)
        tmp_manifest = None
        all_word_segs = []
        out_segments = []

        try:
            for seg in input_segments:
                text = seg.get("text", "").strip()
                seg_start = seg.get("start", 0.0)
                seg_end = seg.get("end", None)

                if not text:
                    continue

                # Build NeMo manifest entry
                manifest_entry = {
                    "audio_filepath": audio_path,
                    "offset": seg_start,
                    "duration": (seg_end - seg_start) if seg_end else None,
                    "text": text,
                }
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".json", delete=False
                ) as mf:
                    mf.write(json.dumps(manifest_entry))
                    tmp_manifest = mf.name

                # Get CTC log-probs + reference token sequences
                (
                    log_probs_batch,
                    y_batch,
                    T_batch,
                    U_batch,
                    utt_obj_batch,
                    output_timestep_duration,
                ) = get_batch_variables(
                    audio=[audio_path],
                    model=self._model,
                    segment_separators=[".", "?", "!", "..."],
                    align_using_pred_text=False,
                    audio_filepath_parts_in_utt_id=1,
                    gt_text_batch=[text],
                    output_timestep_duration=self._output_timestep_duration,
                )

                # Cache timestep duration after first batch
                if self._output_timestep_duration is None:
                    self._output_timestep_duration = output_timestep_duration

                # Viterbi forced alignment
                alignments_batch = viterbi_decoding(
                    log_probs_batch, y_batch, T_batch, U_batch, self._model.device
                )

                # Extract timestamps from alignment path
                for utt_obj, alignment_utt in zip(utt_obj_batch, alignments_batch):
                    utt_obj = add_t_start_end_to_utt_obj(
                        utt_obj, alignment_utt, output_timestep_duration
                    )

                    word_segs = []
                    
                    # Handle NeMo < 3.0.0 (.words) and NeMo >= 3.0.0 (.segments_and_tokens)
                    words_list = []
                    if hasattr(utt_obj, "words"):
                        words_list = utt_obj.words
                    elif hasattr(utt_obj, "segments_and_tokens"):
                        for seg in utt_obj.segments_and_tokens:
                            if hasattr(seg, "words_and_tokens"):
                                for w in seg.words_and_tokens:
                                    if type(w).__name__ == "Word":
                                        words_list.append(w)
                    
                    for word_obj in words_list:
                        word_segs.append({
                            "word": word_obj.text,
                            "start": round(float(word_obj.t_start) + seg_start, 3),
                            "end": round(float(word_obj.t_end) + seg_start, 3),
                            "score": 1.0,
                        })

                    all_word_segs.extend(word_segs)
                    if word_segs:
                        out_segments.append({
                            "start": word_segs[0]["start"],
                            "end": word_segs[-1]["end"],
                            "text": text,
                            "words": word_segs,
                        })

        except Exception as e:
            logger.error(f"NeMo Forced Aligner failed: {e}")
        finally:
            if tmp_manifest and os.path.exists(tmp_manifest):
                os.remove(tmp_manifest)

        return {"segments": out_segments, "word_segments": all_word_segs}
