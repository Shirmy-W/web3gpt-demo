
import streamlit as st
import openai
import PyPDF2
import faiss
import numpy as np
from utils import get_text_chunks, get_embeddings, search_index

st.title("🧠 Web3GPT - 区块链知识问答助手")

openai.api_key = st.secrets["OPENAI_API_KEY"]

uploaded_file = st.file_uploader("📄 上传区块链相关PDF文件", type=["pdf"])

if uploaded_file:
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    full_text = "".join([page.extract_text() or "" for page in pdf_reader.pages])

    chunks = get_text_chunks(full_text)
    embeds = get_embeddings(chunks)

    dim = len(embeds[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeds).astype("float32"))

    question = st.text_input("🤔 输入你的问题：")

    if question:
        related_chunks = search_index(question, index, chunks)
        context = "\n".join(related_chunks)
        prompt = f"基于以下内容回答问题：\n{context}\n\n问题：{question}\n回答："

        with st.spinner("正在思考中..."):
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "你是一个区块链知识专家"},
                    {"role": "user", "content": prompt}
                ]
            )
            st.markdown("🧠 回答：")
            st.write(response.choices[0].message.content)
