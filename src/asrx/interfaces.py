from abc import ABC, abstractmethod
from typing import List, Dict, Any

class VADProvider(ABC):
    """Abstract Base Class for Voice Activity Detection."""
    @abstractmethod
    def detect(self, audio: Any) -> List[Dict[str, float]]:
        """
        Takes audio and returns a list of speech segments.
        Expected format: [{'start': 0.0, 'end': 2.5}, ...]
        """
        pass

class AlignmentProvider(ABC):
    """Abstract Base Class for Forced Alignment."""
    @abstractmethod
    def align(self, text_segments: List[Dict[str, Any]], audio: Any) -> List[Dict[str, Any]]:
        """
        Takes text segments and audio, and returns word-level timestamps.
        Expected format: [{'word': 'Hello', 'start': 0.0, 'end': 0.5}, ...]
        """
        pass

class DiarizationProvider(ABC):
    """Abstract Base Class for Speaker Diarization."""
    @abstractmethod
    def diarize(self, audio: Any) -> List[Dict[str, Any]]:
        """
        Takes audio and returns speaker segments.
        Expected format: [{'speaker': 'SPEAKER_00', 'start': 0.0, 'end': 2.5}, ...]
        """
        pass

