from typing import Any, Dict, List, Optional
import logging
import os
import json
import tempfile
import numpy as np
import soundfile as sf

from ...interfaces import DiarizationProvider

logger = logging.getLogger(__name__)


class SortformerDiarization(DiarizationProvider):
    """
    NVIDIA Sortformer Streaming Speaker Diarization.
    
    Uses nvidia/diar_streaming_sortformer_4spk-v2.1 via NeMo toolkit.
    Supports up to 4 speakers. Works in both offline and streaming modes.
    
    Requires: pip install git+https://github.com/NVIDIA/NeMo.git@main#egg=nemo_toolkit[asr]
    
    Example:
        from asrx.providers.diarization.sortformer import SortformerDiarization
        diarizer = SortformerDiarization(model_name="nvidia/diar_streaming_sortformer_4spk-v2.1")
        df = diarizer.diarize("audio.wav")
    """

    def __init__(
        self,
        model_name: str = "nvidia/diar_streaming_sortformer_4spk-v2.1",
        device: str = None,
        streaming_mode: bool = False,
        chunk_len: int = 340,
    ):
        self.model_name = model_name
        import torch
        self.device = device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.streaming_mode = streaming_mode
        self.chunk_len = chunk_len
        self._model = None
        self._load_model()

    def _load_model(self):
        try:
            import nemo.collections.asr as nemo_asr
            logger.info(f"Loading Sortformer model: {self.model_name}")
            self._model = nemo_asr.models.EncDecDiarLabelModel.from_pretrained(
                model_name=self.model_name
            )
            if self.streaming_mode:
                self._model.streaming_mode = True
            self._model.eval()
            logger.info("Sortformer diarization model loaded successfully.")
        except ImportError:
            raise ImportError(
                "NeMo is not installed. Install with:\n"
                "  pip install Cython packaging\n"
                "  pip install git+https://github.com/NVIDIA/NeMo.git@main#egg=nemo_toolkit[asr]"
            )

    def _audio_to_wav(self, audio: Any) -> str:
        """Saves audio to a temp WAV file and returns its path."""
        if isinstance(audio, str) and os.path.exists(audio):
            return audio
        elif isinstance(audio, np.ndarray):
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            sf.write(tmp.name, audio, samplerate=16000)
            return tmp.name
        elif hasattr(audio, "numpy"):
            data = audio.float().cpu().numpy()
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            sf.write(tmp.name, data, samplerate=16000)
            return tmp.name
        else:
            raise ValueError(f"Unsupported audio type: {type(audio)}")

    def diarize(self, audio: Any) -> Any:
        """
        Perform speaker diarization on the given audio.
        
        Returns:
            A pandas DataFrame with columns ["start", "end", "speaker"] 
            compatible with the ASRX standard diarization output format.
        """
        import pandas as pd

        audio_path = self._audio_to_wav(audio)
        tmp_path = None

        try:
            # Build NeMo manifest for inference
            manifest_entry = {
                "audio_filepath": audio_path,
                "offset": 0,
                "duration": None,
                "label": "infer",
                "text": "-",
                "num_speakers": None,
            }

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False
            ) as mf:
                mf.write(json.dumps(manifest_entry))
                tmp_path = mf.name

            # Run Sortformer inference
            logger.info("Running Sortformer diarization inference...")
            self._model.diarize(
                paths2audio_files=[audio_path],
                batch_size=1,
            )

            # Parse the RTTM output produced by NeMo
            rttm_path = audio_path.replace(".wav", ".rttm")
            if not os.path.exists(rttm_path):
                output_dir = os.path.dirname(audio_path)
                basename = os.path.splitext(os.path.basename(audio_path))[0]
                rttm_path = os.path.join(output_dir, f"{basename}.rttm")

            if not os.path.exists(rttm_path):
                logger.warning("Sortformer: RTTM output not found, returning empty diarization.")
                return pd.DataFrame(columns=["start", "end", "speaker"])

            rows = []
            with open(rttm_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if parts[0] == "SPEAKER":
                        start = float(parts[3])
                        duration = float(parts[4])
                        speaker = parts[7]
                        rows.append({
                            "start": round(start, 3),
                            "end": round(start + duration, 3),
                            "speaker": speaker
                        })

            df = pd.DataFrame(rows)
            logger.info(f"Sortformer detected {df['speaker'].nunique() if len(df) > 0 else 0} speakers.")
            return df

        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)
