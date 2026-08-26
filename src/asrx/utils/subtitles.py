import json
from typing import Dict, Any

def format_timestamp(seconds: float, separator: str = ",") -> str:
    """Format seconds into HH:MM:SS,mmm or HH:MM:SS.mmm format."""
    assert seconds >= 0, "non-negative timestamp expected"
    milliseconds = round(seconds * 1000.0)
    
    hours = milliseconds // 3_600_000
    milliseconds -= hours * 3_600_000
    
    minutes = milliseconds // 60_000
    milliseconds -= minutes * 60_000
    
    seconds = milliseconds // 1_000
    milliseconds -= seconds * 1_000
    
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}{separator}{milliseconds:03d}"

def write_srt(result: Dict[str, Any], filepath: str):
    """Write transcription result to SRT file."""
    with open(filepath, "w", encoding="utf-8") as f:
        for i, segment in enumerate(result["segments"], start=1):
            start = format_timestamp(segment["start"])
            end = format_timestamp(segment["end"])
            
            # If we have diarization info in words, we can prepend speaker
            speaker = ""
            if "words" in result and result["words"]:
                # Try to find speaker for this segment
                for word in result["words"]:
                    if word["start"] >= segment["start"] and word["end"] <= segment["end"]:
                        if "speaker" in word:
                            speaker = f"[{word['speaker']}] "
                            break

            f.write(f"{i}\n")
            f.write(f"{start} --> {end}\n")
            f.write(f"{speaker}{segment['text']}\n\n")

def write_json(result: Dict[str, Any], filepath: str):
    """Write transcription result to JSON file."""
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)
