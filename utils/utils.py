from typing import List

import openai
from config.config import OpenAI_Config

# Set the API key
openai.api_key = OpenAI_Config.api_key

# Function that communicates with GPT-3 API
# TODO: Add prompt details as above
def query_gpt3(prompt: str, **kwargs) -> str:
    # Build the request
    default_params = {
        'engine': 'gpt-3.5-turbo',
        'temperature': 0.9,
        'max_tokens': 2048,
        'top_p': 1,
        'frequency_penalty': 0,
        'presence_penalty': 0.6,
    }

    response = openai.ChatCompletion.create(
        prompt=prompt,
        **(default_params | kwargs)
    )

    return response.choices[0].message

def query_gpt2(prompt: str, **kwargs) -> str:
    # Build the request
    default_params = {
        'engine': 'davinci',
        'temperature': 0.9,
        'max_tokens': 150,
        'top_p': 1,
        'frequency_penalty': 0,
        'presence_penalty': 0.6,
    }

    response = openai.Completion.create(
        prompt=prompt,
        **(default_params | kwargs)
    )

    return response.choices[0].text


def query_wikipedia(query: str) -> str:
    # TODO: Implement
    pass

def query_bing(query: str) -> List[str]:
    # TODO: Implement
    pass
