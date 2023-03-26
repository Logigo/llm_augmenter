from enum import Enum
from dataclasses import dataclass

@dataclass
class Evidence:
    text: str
    source: str

class Actions(Enum):
    AcquireEvidence = 'AcquireEvidence'
    GenerateCandidateResponse = 'GenerateCandidateResponse'
    ReturnResponseToUser = 'ReturnResponseToUser'
