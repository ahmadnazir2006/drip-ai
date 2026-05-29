import os
from dotenv import load_dotenv
from huggingface_hub import whoami

load_dotenv()
token = os.getenv('HF_TOKEN')

try:
    user = whoami(token=token)
    print(f"✅ Success! You are logged in as: {user['name']}")
except Exception as e:
    print(f"❌ Token Error: {e}")
    print("Check your .env file. It should be: HF_TOKEN=hf_your_actual_token")