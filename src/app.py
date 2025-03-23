from fastapi import FastAPI, HTTPException
from src.data_preparation import prepare_data
from src.embedding_generator import generate_embeddings
from src.vector_store import create_vector_store, load_vector_store
from src.llm_integration import generate_response
from src.config import PLAID_CLIENT_ID, PLAID_SECRET, PLAID_ENV
from sentence_transformers import SentenceTransformer
import json
import datetime
from openai import OpenAI

# Initialize Sentence Transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize OpenAI client for OpenRouter API
class LLaMAClient:
    def __init__(self, api_key):
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key
        )

    def generate(self, prompt):
        start_time = datetime.datetime.now()

        # Request OpenRouter LLaMA-3 with provided settings
        completion = self.client.chat.completions.create(
            extra_body={},
            model="nvidia/llama-3.1-nemotron-70b-instruct:free",  # OpenRouter Free Tier Model
            logprobs=True,
            messages=[{"role": "assistant", "content": prompt}],
            max_tokens=500,
            temperature=0.3,
            seed=2309,
        )

        # Extract the response text
        return completion.choices[0].message.content


llama3_client = LLaMAClient(api_key="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")

app = FastAPI()

# Load Plaid data
with open("data/plaid_data.json", "r") as file:
    plaid_data = json.load(file)

# Prepare data
data = prepare_data(plaid_data)

# Generate embeddings and create vector store
embeddings, texts = generate_embeddings(data)
index = create_vector_store(embeddings)

@app.get("/")
async def root():
    return {"message": "Hello from FastAPI on Render!"}

@app.get("/query")
def query_chatbot(question: str):
    try:
        result = generate_response(question, data, index, model, llama3_client)
        return {"response": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
