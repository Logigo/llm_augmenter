"""
Given a candidate response o, Utility generates score u and feedback f using task-specific utility functions. These utility functions access alignment of LLM's responses with user expectations/business requirements.

e.g. in 'information seeking' dialog, all LLM responses should be grounded in external evidence to avoid generating misleading/inaccurate info.
In 'restaurant reservation' dialog, responses should be conversational and focused on guiding the user through reservation process vs. chitchats

There are two types of utility functions to generate **u**:

**model-based** utility assign preference scores to different dimensions of a response, e.g. fluency, informativeness, and factuality (why is it called model-based?). these functiosn are trained on pre-collected human preference data or annotated log data (what is that?)

**rule-based** utility, implemented using heuristics or programmed functions, measure whether a response complies with a specific  rule

Generating **********************feedback f:**********************

f is generated with a text generation model Q parametrized by phi, implemented as seq2seq or autoregression language model.

takes q, e, o, h_q as input and generates f = Q_phi(q, e, o, h_q)

can also use LLMs or  rule-based natural language generator for feedback generation
"""

