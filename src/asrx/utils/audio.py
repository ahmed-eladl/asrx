import numpy as np
import subprocess
import os

def load_audio(file: str, sr: int = 16000) -> np.ndarray:
    """
    Open an audio file and read as mono waveform, resampling as necessary.
    Uses ffmpeg underneath.
    """
    try:
        import librosa
        audio, _ = librosa.load(file, sr=sr, mono=True)
        return audio
    except ImportError:
        # Fallback to ffmpeg subprocess
        cmd = [
            "ffmpeg",
            "-nostdin",
            "-threads", "0",
            "-i", file,
            "-f", "s16le",
            "-ac", "1",
            "-acodec", "pcm_s16le",
            "-ar", str(sr),
            "-"
        ]
        try:
            out = subprocess.run(cmd, capture_output=True, check=True).stdout
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

        return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0
