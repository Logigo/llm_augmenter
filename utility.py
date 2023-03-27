from typing import List
from toolz.curried import pipe
from dataclasses import dataclass

from utils.utils import query_gpt3

import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration, T5Tokenizer
from torch.nn.utils.rnn import pack_sequence

import torchtext
from transformers import EncoderDecoderModel

"""
Given a candidate response o, Utility generates score u and feedback f using task-specific utility functions. These utility functions access alignment of LLM's responses with user expectations/business requirements.

e.g. in 'information seeking' dialog, all LLM responses should be grounded in external evidence to avoid generating misleading/inaccurate info.
In 'restaurant reservation' dialog, responses should be conversational and focused on guiding the user through reservation process vs. chitchats

There are two types of utility functions to generate **u**:

**model-based** utility assign preference scores to different dimensions of a response, e.g. fluency, informativeness, and factuality (why is it called model-based?).
    these functiosn are trained on pre-collected human preference data or annotated log data (what is that?)

**rule-based** utility, implemented using heuristics or programmed functions, measure whether a response complies with a specific  rule

Generating *feedback f:*

f is generated with a text generation model Q parametrized by phi, implemented as seq2seq or autoregression language model.

takes q, e, o, h_q as input and generates f = Q_phi(q, e, o, h_q)

can also use LLMs or  rule-based natural language generator for feedback generation
"""
tokenizer = T5Tokenizer.from_pretrained('t5-small')

@dataclass
class ModelBasedResponseAlignment():
    """
    Represents alignment of response with user expectations/business requirements

    Model-based:
        - fluency
        - informativeness
        - factuality
    """
    fluency: float
    informativeness: float
    factuality: float

@dataclass
class RuleBasedResponseAlignment():
    """
    Represents alignment of response with user expectations/business requirements

    Rule-based:
        - compliance with task-specific rules
    """
    rule: function
    complies: bool


class UtilityModel():
    def __init__(self) -> None:
        pass



class Feedback(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int, output_size: int) -> None:
        self.encoder = nn.Sequential([
            tokenizer,
            nn.Embedding(input_dim, hidden_size),
            nn.GRU(hidden_size, hidden_size, num_layers=1, batch_first=True)
        ])
        self.decoder = nn.Sequential([
            nn.Embedding(output_size, hidden_size),
            nn.GRU(hidden_size, hidden_size, num_layers=1, batch_first=True),
            nn.Linear(hidden_size, output_size)
        ])


    def generate_feedback_gpt3(self, q: str, e: str, o: str, h_q: List[str]) -> List[str]:
        """
        Generates feedback f from q, e, o, h_q by querying ChatGPT
        """
        return pipe([
            (q, e, o, h_q),
            _generate_feedback_prompt,
            query_gpt3,
            lambda x: x.split('\n')
        ])


    def forward(self, q: str, e: str, o: str, h_q: List[str]) -> torch.Tensor:
        """
        Forward function for feedback generation using seq2seq
        """
        embedded_sequences = [self.embedding(seq) for seq in [q, e, o, h_q]]
        # Since we are passing in multiple sequences, we need to pack them to treat them as 1 sequence.
        packed_sequences = pack_sequence(embedded_sequences, enforce_sorted=False)
        out = self.decoder(packed_sequences)
        return out


    def generate_feedback_seq2seq(self, q: str, e: str, o: str, h_q: List[str]) -> List[str]:
        """
        Generates feedback f from q, e, o, h_q using seq2seq model
        """
        return pipe([
            (q, e, o, h_q),
            self.forward,
        ])

def _generate_feedback_prompt(q: str, e: str, o: str, h_q: List[str]) -> str:
    """
    Generates prompt for feedback generation
    """
    return f"""
        I am going to give you a query, a response, some evidence, and a history of queries. For the response to the query, I want you to generate feedback that will help me improve my response generation.

        History: {h_q}
        Query: {q}
        Response: {o}
        Evidence: {e}
    """
