from groq import Client
from main.py import client
client = Client()
models = client.models.list()
for r in models.data:
    print(r.id)