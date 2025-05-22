import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

st.set_page_config(page_title="Web3GPT - 区块链知识问答", layout="wide")

st.title("📘 Web3GPT - 区块链知识问答助手")
st.write("上传你的区块链 PDF 文档，提出问题，让 GPT 为你回答。")

api_key = st.text_input("🔑 输入你的 OpenAI API Key", type="password")

uploaded_file = st.file_uploader("📄 上传区块链 PDF 文件", type="pdf")
question = st.text_input("💬 你想问什么？")

if uploaded_file and openai_api_key:
    # ✅ 读取上传 PDF 文件内容（兼容 Streamlit）
    pdf_reader = PdfReader(io.BytesIO(uploaded_file.read()))
    raw_text = ""
    for page in pdf_reader.pages:
        raw_text += page.extract_text() or ""

    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_text(raw_text)

   embeddings = OpenAIEmbeddings(
    model_name="text-embedding-ada-002",
    openai_api_key="sk-你的key"
)
    vectorstore = FAISS.from_texts(texts, embeddings)
    st.success("PDF 已成功上传并处理为向量。")

    llm = OpenAI(temperature=0)
    chain = load_qa_chain(llm, chain_type="stuff")
    response = chain.run(input_documents=docs, question=question)

    st.write("🧠 GPT 的回答：")
    st.success(response)
