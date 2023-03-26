import yaml
import sys

OPENAI_CONF = yaml.safe_load(open('config\openai.yaml'))

class OpenAI_Config:
    api_key = OPENAI_CONF['api_key']

if __name__ == '__main__':
    print(sys.path)
    print(OpenAI_Config.api_key)
