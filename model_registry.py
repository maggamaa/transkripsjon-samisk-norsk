"""Central registry for ASR model metadata shared across backend utilities."""

from transformers import (
    # WhisperProcessor,
    # WhisperForConditionalGeneration,
    Wav2Vec2Processor,
    Wav2Vec2ForCTC,
)

# Public server uses a single fixed ASR model to ensure stability and low GPU usage
SERVER_MODEL_NAME = "GetmanY1/wav2vec2-large-sami-cont-pt-22k-finetuned"

DEFAULT_CONFIG = {
    "MODEL_NAME": SERVER_MODEL_NAME, # Replaced: "MODEL_NAME": "NbAiLab/nb-whisper-large",
    "USE_FLOAT16": True, # ignored for wav2vec2 (CTC runs in float 32)
    "MAX_BUFFER_SECONDS": 10,
    "TARGET_SAMPLERATE": 16000,
    "VAD_THRESHOLD": 0.5,
    "SILENCE_DURATION_S": 1.0,
    # Added for public server----------
    "PUNCTUATION_ENABLED": True,
    "PUNCTUATION_LANG": "sme",
    #----------------------------------    
}

SUPPORTED_MODELS = {
    # Whisper models (disabled in public server version)
    # Kept for internal testing / development setups
    # "NbAiLab/nb-whisper-large": {
    #     "label": "Sámi → Norsk",
    #     "language": "no",
    #     "type": "whisper",
    #     "processor_cls": WhisperProcessor,
    #     "model_cls": WhisperForConditionalGeneration,
    #     "prefer_float16": True,
    # },
    # "NbAiLab/whisper-large-sme": {
    #     "label": "Sámi → Sámi",
    #     "language": None,
    #     "type": "whisper",
    #     "processor_cls": WhisperProcessor,
    #     "model_cls": WhisperForConditionalGeneration,
    #     "prefer_float16": True,
    # },
    SERVER_MODEL_NAME: { # Replaced: "GetmanY1/wav2vec2-large-sami-cont-pt-22k-finetuned": {
        "label": "Sámi Parliament (CTC)",
        "language": "sme",
        "type": "ctc",
        "processor_cls": Wav2Vec2Processor,
        "model_cls": Wav2Vec2ForCTC,
        "prefer_float16": False,
    }
}