import os
from dotenv import load_dotenv
from openai import OpenAI, AsyncOpenAI

env_path=os.path.join(os.path.dirname(__file__), '.env.txt')
load_dotenv(dotenv_path=env_path)

openAI_api_key=os.getenv('OpenAI_API_key')
SerpAPI_key=os.getenv('SerpAPI_key')
admin_password=os.getenv('admin_password')
uip_id=os.getenv('uip_assistant_id')
rg_id=os.getenv('rg_assistant_id')

client=OpenAI(api_key=openAI_api_key)
async_client=AsyncOpenAI(api_key=openAI_api_key)