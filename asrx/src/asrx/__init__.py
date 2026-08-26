__version__ = "0.3.0"

from .interfaces import AlignmentProvider, DiarizationProvider, VADProvider
from .aligner import AlignmentPipeline
from .shortcuts import align, load_aligner

# ── Alignment Providers ───────────────────────────────────────────────────────

def Wav2Vec2Alignment(**kwargs):
    """Wav2Vec2 Alignment — 40+ languages. Install: pip install transformers"""
    from .providers.alignment.wav2vec2 import Wav2Vec2Alignment as _W2V
    return _W2V(**kwargs)

def MMSAlignment(**kwargs):
    """Meta MMS — 1,107+ languages. Install: pip install transformers
    License: CC-BY-NC 4.0 (non-commercial)"""
    from .providers.alignment.mms import MMSAlignment as _MMS
    return _MMS(**kwargs)

def NeMoForcedAligner(**kwargs):
    """NVIDIA NeMo Forced Aligner — CTC Viterbi, Conformer/FastConformer models.
    Install: pip install 'nemo_toolkit[asr]>=2.5.0'"""
    from .providers.alignment.nemo_aligner import NeMoForcedAligner as _NFA
    return _NFA(**kwargs)

def CTCSegmentationAlignment(**kwargs):
    """CTC-Segmentation — any CTC model + per-segment confidence scores.
    Install: pip install ctc-segmentation"""
    from .providers.alignment.ctc_segmentation import CTCSegmentationAlignment as _CTCS
    return _CTCS(**kwargs)

# ----------------- VAD Providers -----------------

def SileroVAD(**kwargs):
    """Silero VAD — Robust, fast VAD."""
    from .providers.vad.silero import SileroVAD as _SV
    return _SV(**kwargs)

def FlashVAD(**kwargs):
    """FlashVAD — ultra-lightweight ONNX VAD (~46K params, 10ms frames).
    Install: pip install git+https://github.com/oss-codes/flashvad.git"""
    from .providers.vad.flashvad import FlashVAD as _FV
    return _FV(**kwargs)

def PyannoteVAD(**kwargs):
    """Pyannote VAD — Highly accurate neural VAD.
    Install: pip install pyannote.audio"""
    from .providers.vad.pyannote_vad import PyannoteVAD as _PV
    return _PV(**kwargs)

def NeMoMarbleNetVAD(**kwargs):
    """NVIDIA NeMo MarbleNet — Multilingual VAD.
    Install: pip install 'nemo_toolkit[asr]>=2.5.0'"""
    from .providers.vad.nemo_vad import NeMoMarbleNetVAD as _NV
    return _NV(**kwargs)

# ----------------- Diarization Providers -----------------

def PyannoteDiarization(**kwargs):
    """Pyannote Diarization — Highly accurate speaker diarization.
    Install: pip install pyannote.audio"""
    from .providers.diarization.pyannote import PyannoteDiarization as _PD
    return _PD(**kwargs)

def SortformerDiarization(**kwargs):
    """NVIDIA Sortformer — streaming diarization, up to 4 speakers.
    Install: pip install 'nemo_toolkit[asr]>=2.5.0'"""
    from .providers.diarization.sortformer import SortformerDiarization as _SF
    return _SF(**kwargs)


# ── Public API ────────────────────────────────────────────────────────────────

__all__ = [
    # Core
    "align",
    "load_aligner",
    "AlignmentPipeline",
    # Alignment Backends
    "Wav2Vec2Alignment",      # 40 languages, default
    "MMSAlignment",           # 1,107+ languages (CC-BY-NC)
    "NeMoForcedAligner",      # NVIDIA Conformer-CTC
    "CTCSegmentationAlignment",  # any CTC model + confidence scores
    # VAD Backends
    "SileroVAD",              # default
    "FlashVAD",               # ultra-lightweight ONNX
    "PyannoteVAD",            # highly accurate neural
    "NeMoMarbleNetVAD",       # NVIDIA MarbleNet
    # Diarization Backends
    "PyannoteDiarization",    # default, neural speaker embeddings
    "SortformerDiarization",  # NVIDIA streaming, up to 4 speakers
    # Abstract Interfaces
    "AlignmentProvider",
    "DiarizationProvider",
    "VADProvider",
]
