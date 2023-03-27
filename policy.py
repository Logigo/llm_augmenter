import torch
from torch import nn
from transformers import T5ForConditionalGeneration, T5Tokenizer
from oop_models._types import Actions

"""
- Selects the next *system_action* that leads to the best expected reward R.
- These actions:
    - *include acquiring evidence for q*
    - *generating a candidate response*
    - *returning response to user*
- Can be implemented using manually crafted rules or by training on human-system interactions. In this study, they implement a neural network to act as policy (pi) with parameters theta. pi_theta is optimized with REINFORCE, to maximize expected rewards as (maximizing rewards in expectation given sampled state, actions):


- Policy learning usually requires large amounts of human-machine interactions (e.g. in RLHF. To get around this, they train this:
    1. 'Bootstrapping from rule-based policy': domain experts encode task-specific knowledge and business logic into IF-THEN rules
      (e.g. if product name in user query for customer service, always call KC to collect info about product from product database).
    2. A language model simulates how human users interact with LLM-Augmenter. Any valid response from the augmenter that passes Utility's evaluation can be used as a training example.
        This sort of allows the augmenter to self-improve because it can self-generate examples.
    3. Actually interacting with humans

"""

# T5 Base model
model = T5ForConditionalGeneration.from_pretrained('t5-small')
tokenizer = T5Tokenizer.from_pretrained('t5-small')

# TODO: Implement train method with REINFORCE
# argmax(E[R(s,a)])

