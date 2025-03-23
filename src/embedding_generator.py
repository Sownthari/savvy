from sentence_transformers import SentenceTransformer

# Load the Sentence Transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

def generate_embeddings(data):
    texts = [
        f"{item['date']} - {item['merchant']} - {item['amount']} USD - {item['category']}"
        for item in data
    ]
    embeddings = model.encode(texts, convert_to_tensor=True)
    return embeddings, texts
