# download_model.py
import torch
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Wav2Vec2Processor,
    Wav2Vec2ForCTC,
)

from model_registry import SUPPORTED_MODELS
USE_FLOAT16 = True
torch_dtype = torch.float16 if USE_FLOAT16 and torch.cuda.is_available() else torch.float32

# 1. Download models
for model_name, spec in SUPPORTED_MODELS.items():
    model_type = spec.get("type", "whisper")
    print(f"Downloading and caching {model_type.upper()}-model: {model_name}")
    if model_type == "whisper":
        WhisperProcessor.from_pretrained(model_name)
        WhisperForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch_dtype)
    else:
        Wav2Vec2Processor.from_pretrained(model_name)
        Wav2Vec2ForCTC.from_pretrained(model_name)
    print(f"{model_name} downloaded and cached successfully.")

# 2. Download Silero VAD model
print("Downloading and caching Silero VAD model...")
torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=True, onnx=False, trust_repo=True)
print("VAD-model downloaded and cached successfully.")

print("\nAll required models are downloaded and cached.")
print("Translation is now handled by TartuNLP Tahetorn_9B API in the frontend.")
