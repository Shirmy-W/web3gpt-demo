
import openai
import tiktoken
import faiss
import numpy as np

def get_text_chunks(text, chunk_size=500):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

def get_embeddings(text_list, model="text-embedding-3-small"):
    response = openai.Embedding.create(
        input=text_list,
        model=model
    )
    return [record["embedding"] for record in response["data"]]

def search_index(query, index, texts, k=3):
    query_vec = get_embeddings([query])[0]
    D, I = index.search(np.array([query_vec]).astype("float32"), k)
    return [texts[i] for i in I[0]]
