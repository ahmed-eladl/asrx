from typing import Any, Dict, List, Optional, Union
import logging

from .interfaces import AlignmentProvider, DiarizationProvider, VADProvider
from .aligner import AlignmentPipeline


logger = logging.getLogger(__name__)

# Cached pipeline instances for one-liner re-use
_GLOBAL_ALIGNERS: Dict[str, AlignmentPipeline] = {}

# Alignment backend registry — string name → factory function
_ALIGNMENT_BACKENDS = {
    "wav2vec2": lambda language, device, **kw: _build_w2v(language, device, **kw),
    "mms": lambda language, device, **kw: _build_mms(language, device, **kw),
    "nemo": lambda language, device, **kw: _build_nemo(language, device, **kw),
    "ctc_segmentation": lambda language, device, **kw: _build_ctc_seg(language, device, **kw),
}

# VAD backend registry
_VAD_BACKENDS = {
    "silero": lambda **kw: _build_silero(**kw),
    "flashvad": lambda **kw: _build_flashvad(**kw),
    "pyannote": lambda hf_token, device, **kw: _build_pyannote_vad(hf_token, device, **kw),
    "nemo": lambda device, **kw: _build_nemo_vad(device, **kw),
}

# Diarization backend registry
_DIARIZATION_BACKENDS = {
    "pyannote": lambda hf_token, device, **kw: _build_pyannote_diar(hf_token, device, **kw),
    "sortformer": lambda hf_token, device, **kw: _build_sortformer(device, **kw),
}


# ── Private builder functions (lazy imports) ─────────────────────────────────

def _build_w2v(language: str, device: str, **kw):
    from .providers.alignment.wav2vec2 import Wav2Vec2Alignment
    return Wav2Vec2Alignment(
        model_id=kw.get("model_id", language), device=device
    )

def _build_silero(**kw):
    from .providers.vad.silero import SileroVAD
    return SileroVAD()

def _build_pyannote_diar(hf_token: str, device: str, **kw):
    from .providers.diarization.pyannote import PyannoteDiarization
    return PyannoteDiarization(
        use_auth_token=hf_token, device=device, model_name=kw.get("model_name")
    )

def _build_mms(language: str, device: str, **kw):
    from .providers.alignment.mms import MMSAlignment
    return MMSAlignment(
        model_id=kw.get("model_id", "facebook/mms-1b-all"),
        device=device,
        language=language,
    )

def _build_nemo(language: str, device: str, **kw):
    from .providers.alignment.nemo_aligner import NeMoForcedAligner
    return NeMoForcedAligner(
        language=language,
        model_name=kw.get("model_name"),
        model_path=kw.get("model_path"),
        device=device,
    )

def _build_ctc_seg(language: str, device: str, **kw):
    from .providers.alignment.ctc_segmentation import CTCSegmentationAlignment
    return CTCSegmentationAlignment(
        language=language,
        backend=kw.get("ctc_backend", "mms"),
        model_id=kw.get("model_id"),
        device=device,
        confidence_threshold=kw.get("confidence_threshold", -5.0),
    )

def _build_flashvad(**kw):
    from .providers.vad.flashvad import FlashVAD
    return FlashVAD(
        threshold=kw.get("threshold", 0.5),
        min_speech_duration_ms=kw.get("min_speech_duration_ms", 250),
    )

def _build_pyannote_vad(hf_token: Optional[str], device: str, **kw):
    from .providers.vad.pyannote_vad import PyannoteVAD
    return PyannoteVAD(use_auth_token=hf_token, device=device)

def _build_nemo_vad(device: str, **kw):
    from .providers.vad.nemo_vad import NeMoMarbleNetVAD
    return NeMoMarbleNetVAD(
        model_name=kw.get("model_name", "vad_multilingual_marblenet"),
        device=device,
    )

def _build_sortformer(device: str, **kw):
    from .providers.diarization.sortformer import SortformerDiarization
    return SortformerDiarization(
        model_name=kw.get(
            "model_name", "nvidia/diar_streaming_sortformer_4spk-v2.1"
        ),
        device=device,
    )


# ── Public API ────────────────────────────────────────────────────────────────

def load_aligner(
    language: str = "ar",
    aligner: Union[str, AlignmentProvider] = "wav2vec2",
    vad: Union[bool, str, VADProvider] = True,
    diarize: Union[bool, str, DiarizationProvider] = False,
    device: str = "cuda",
    hf_token: Optional[str] = None,
    **backend_kwargs,
) -> AlignmentPipeline:
    """
    Loads and returns a reusable ASRX AlignmentPipeline instance.

    Args:
        language: ISO 639-1 language code ('ar', 'en', 'fr', 'de', etc.)
        aligner:  Alignment backend. One of:
                    - "wav2vec2"       (default, 40 languages)
                    - "mms"            (Meta MMS, 1,107+ languages, CC-BY-NC)
                    - "nemo"           (NVIDIA Conformer-CTC, high accuracy)
                    - "ctc_segmentation" (any CTC model + confidence scores)
                    - An AlignmentProvider instance (custom)
        vad:      VAD backend. One of:
                    - True / "silero"  (default)
                    - "pyannote"       (highly accurate neural VAD)
                    - "nemo"           (NVIDIA MarbleNet)
                    - "flashvad"       (ultra-lightweight ONNX)
                    - False            (disable)
                    - A VADProvider instance
        diarize:  Diarization backend. One of:
                    - True / "pyannote" (default when enabled)
                    - "sortformer"      (NVIDIA NeMo, up to 4 speakers)
                    - False             (disable)
                    - A DiarizationProvider instance
        device:   'cuda' or 'cpu'
        hf_token: HuggingFace token (required for Pyannote diarization)
        **backend_kwargs: Extra kwargs passed to the backend factories, e.g.:
                    model_id=, model_name=, model_path=,
                    ctc_backend=, confidence_threshold=,
                    threshold= (FlashVAD)

    Example:
        # Default Wav2Vec2 alignment (fast)
        pipeline = asrx.load_aligner(language="en")
        
        # MMS (1,107+ languages)
        pipeline = asrx.load_aligner(language="sw", aligner="mms")
        
        # NVIDIA NeMo (Viterbi CTC)
        pipeline = asrx.load_aligner(language="en", aligner="nemo")
        
        # CTC-Segmentation (Confidence scores)
        pipeline = asrx.load_aligner(language="en", aligner="ctc_segmentation",
                                     min_confidence=-2.0)
                                     
        # Ultra-lightweight FlashVAD
        pipeline = asrx.load_aligner(language="en", vad="flashvad")
        
        # NVIDIA Sortformer Diarization
        pipeline = asrx.load_aligner(language="en", diarize="sortformer")

        # Combining specific backends
        pipeline = asrx.load_aligner(
            language="en",
            aligner="mms",
            vad="pyannote",
            diarize="sortformer",
            device="cuda",
        )
    """
    # ── Resolve aligner ──────────────────────────────────────────────────────
    if isinstance(aligner, str):
        aligner_key = aligner.lower()
        if aligner_key not in _ALIGNMENT_BACKENDS:
            raise ValueError(
                f"Unknown aligner '{aligner}'. "
                f"Choose from: {list(_ALIGNMENT_BACKENDS.keys())}"
            )
        try:
            align_provider = _ALIGNMENT_BACKENDS[aligner_key](
                language, device, **backend_kwargs
            )
        except Exception as e:
            logger.warning(
                f"Could not load '{aligner}' aligner: {e}\n"
                f"Falling back to wav2vec2."
            )
            align_provider = _build_w2v(language, device)
    elif isinstance(aligner, AlignmentProvider):
        align_provider = aligner
    else:
        align_provider = _build_w2v(language, device)

    # ── Resolve VAD ──────────────────────────────────────────────────────────
    vad_provider = None
    if vad is not False and vad is not None:
        if isinstance(vad, VADProvider):
            vad_provider = vad
        else:
            vad_key = (vad if isinstance(vad, str) else "silero").lower()
            if vad_key not in _VAD_BACKENDS:
                logger.warning(f"Unknown VAD '{vad_key}', falling back to silero.")
                vad_key = "silero"
            try:
                vad_provider = _VAD_BACKENDS[vad_key](**backend_kwargs)
            except Exception as e:
                logger.warning(f"Could not load VAD '{vad_key}': {e}")

    # ── Resolve Diarizer ─────────────────────────────────────────────────────
    diarizer_provider = None
    if diarize is not False and diarize is not None:
        if isinstance(diarize, DiarizationProvider):
            diarizer_provider = diarize
        else:
            diar_key = (diarize if isinstance(diarize, str) else "pyannote").lower()
            if diar_key not in _DIARIZATION_BACKENDS:
                logger.warning(f"Unknown diarizer '{diar_key}', falling back to pyannote.")
                diar_key = "pyannote"
            try:
                diarizer_provider = _DIARIZATION_BACKENDS[diar_key](
                    hf_token=hf_token, device=device, **backend_kwargs
                )
            except Exception as e:
                logger.warning(f"Could not load diarizer '{diar_key}': {e}")

    return AlignmentPipeline(
        language=language,
        aligner=align_provider,
        vad=vad_provider,
        diarizer=diarizer_provider,
        device=device,
    )


def align(
    audio: Any,
    text: Union[str, List[Dict[str, Any]]],
    language: str = "ar",
    aligner: Union[str, AlignmentProvider] = "wav2vec2",
    vad: Union[bool, str, VADProvider] = True,
    diarize: Union[bool, str, DiarizationProvider] = False,
    device: str = "cuda",
    hf_token: Optional[str] = None,
    **backend_kwargs,
) -> Dict[str, Any]:
    """
    One-liner forced alignment. Takes audio + pre-existing text → word timestamps.

    Args:
        audio:    Path to audio file, numpy array, or torch tensor
        text:     Pre-computed transcript string or list of segment dicts
        language: ISO 639-1 language code
        aligner:  "wav2vec2" | "mms" | "nemo" | "ctc_segmentation" | AlignmentProvider
        vad:      True | "silero" | "flashvad" | False | VADProvider
        diarize:  False | True | "pyannote" | "sortformer" | DiarizationProvider
        device:   "cuda" or "cpu"
        hf_token: HuggingFace token for gated models

    Example:
        import asrx

        text = "Hello, I don't know where we are going, but the most important thing is the place is quiet."
        result = asrx.align(
            audio="meeting.wav",
            text=text,
            language="en",
            vad=True,
        )
        print(result["word_segments"])
    """
    aligner_str = aligner if isinstance(aligner, str) else "custom"
    vad_str = vad if isinstance(vad, (str, bool)) else "custom"
    diar_str = diarize if isinstance(diarize, (str, bool)) else "custom"
    cache_key = f"{language}_{aligner_str}_{vad_str}_{diar_str}_{device}"

    if cache_key not in _GLOBAL_ALIGNERS:
        _GLOBAL_ALIGNERS[cache_key] = load_aligner(
            language=language,
            aligner=aligner,
            vad=vad,
            diarize=diarize,
            device=device,
            hf_token=hf_token,
            **backend_kwargs,
        )

    pipeline = _GLOBAL_ALIGNERS[cache_key]
    return pipeline.align(audio=audio, text=text, language=language)
