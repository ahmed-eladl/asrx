import argparse
import logging
from asrx.pipeline import UniversalPipeline
from asrx.providers.asr.hf_transformers import HFTransformersASR
from asrx.providers.alignment.wav2vec2 import Wav2Vec2Alignment
from asrx.utils.subtitles import write_srt, write_json
from asrx.utils.audio import load_audio

logging.basicConfig(level=logging.INFO)

def main():
    parser = argparse.ArgumentParser(description="Test run the Universal ASRX pipeline.")
    parser.add_argument("audio", type=str, help="Path to audio file")
    parser.add_argument("--model", type=str, default="openai/whisper-tiny", help="Hugging Face Model ID")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run on (cpu/cuda)")
    args = parser.parse_args()

    # 1. Initialize Providers
    # For this test, we skip VAD and Diarization to keep it lightweight.
    # In a full run, we would inject SileroVAD and PyannoteDiarization.
    print(f"Loading ASR model {args.model}...")
    asr = HFTransformersASR(model_id=args.model, device=args.device)
    
    print("Loading Alignment model...")
    aligner = Wav2Vec2Alignment(device=args.device)
    
    # 2. Build Pipeline
    pipeline = UniversalPipeline(
        asr_provider=asr,
        alignment_provider=aligner
    )
    
    # 3. Load Audio
    print(f"Loading audio {args.audio}...")
    audio_data = load_audio(args.audio)
    
    # 4. Process
    print("Running pipeline...")
    result = pipeline.process(audio_data)
    
    # 5. Output
    print("Writing output files...")
    write_srt(result, "output.srt")
    write_json(result, "output.json")
    print("Done! Check output.srt and output.json")

if __name__ == "__main__":
    main()
