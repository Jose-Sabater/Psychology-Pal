import whisper
import torch
from typing import Optional, Union


class AudioTranscriber:
    """Transcriber class that uses the whisper library to transcribe audio files"""

    def __init__(self, model: str = "small"):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = whisper.load_model(model, device=device)
        self.transcript = None

    # Transcribe method that takes either a file path or a file object
    def transcribe(self, audio: Optional[Union[str, object]] = None, **kwargs) -> str:
        """Transcribes an audio file using the whisper library"""
        if audio:
            self.transcript = self.model.transcribe(audio, **kwargs)
        else:
            raise ValueError("Please provide either a file path or a file object")

        return self.transcript
