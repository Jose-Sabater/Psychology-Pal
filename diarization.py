from pyannote.audio import Pipeline
import pyannote
import torch
from config import secrets


class DiarizationHelper:
    def __init__(self):
        self.pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=secrets["hf_token"],
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pipeline.to(self.device)

    def __call__(self, path: str) -> pyannote.core.Annotation:
        return self.run_diarization

    def run_diarization(self, filepath: str) -> pyannote.core.Annotation:
        diarization = self.pipeline(filepath)
        return diarization
