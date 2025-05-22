import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import streamlit as st
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

load_dotenv()  # 加载 .env 文件中的环境变量

st.title("Web3GPT 文档问答")

uploaded_file = st.file_uploader("上传 PDF 文件", type="pdf")

if uploaded_file is not None:
    pdf_reader = PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    # 设置 OpenAI Embeddings
    embeddings = OpenAIEmbeddings(
        model="text-embedding-ada-002",
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

    vectorstore = FAISS.from_texts([text], embeddings)

    st.success("PDF 已处理完毕，可用于问答。")
