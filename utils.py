
import openai
import tiktoken
import faiss
import numpy as np
from tenacity import retry, stop_after_attempt, wait_fixed

def get_text_chunks(text, chunk_size=500):
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    return chunks[:4]

@retry(stop=stop_after_attempt(5), wait=wait_fixed(2))
def create_embedding(text, model="text-embedding-3-small"):
    return openai.Embedding.create(input=text, model=model)
 

def get_embeddings(text_list, model="text-embedding-3-small"):
    response = create_embedding(text_list)
    return [record["embedding"] for record in response["data"]]

def search_index(query, index, texts, k=3):
    query_vec = get_embeddings([query])[0]
    D, I = index.search(np.array([query_vec]).astype("float32"), k)
    return [texts[i] for i in I[0]]
