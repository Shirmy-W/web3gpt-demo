
import openai
import tiktoken
import faiss
import numpy as np
import time
from os import error
# from tenacity import retry, stop_after_attempt, wait_random_exponential
def get_text_chunks(text, chunk_size=500):
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    return chunks[:4]

# @retry(stop=stop_after_attempt(6), wait=wait_random_exponential(min=1, max=60))
# def create_embedding(**kwargs):
    # return openai.Embedding.create(**kwargs)
 

def get_embeddings(text_list, model="text-embedding-3-small"):
    try:
        response = openai.Embedding.create(input=text_list, model=model)
        return [record["embedding"] for record in response["data"]]
    except error:
        time.sleep(10)
        return  get_embeddings(text_list)

def search_index(query, index, texts, k=3):
    query_vec = get_embeddings([query])[0]
    D, I = index.search(np.array([query_vec]).astype("float32"), k)
    return [texts[i] for i in I[0]]
