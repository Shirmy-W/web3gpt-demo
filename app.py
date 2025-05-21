import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

st.set_page_config(page_title="Web3GPT", layout="wide")
st.header("📄 Web3 PDF 向量问答演示")

uploaded_file = st.file_uploader("上传PDF文件", type="pdf")
if uploaded_file is not None:
    pdf_reader = PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_text(text)

    embeddings = OpenAIEmbeddings(
        model_name="text-embedding-ada-002",
        openai_api_key=os.getenv("OPENAI_API_KEY")  # 确保部署环境中配置了环境变量
    )
    vectorstore = FAISS.from_texts(texts, embeddings)
    st.success("PDF 已成功上传并处理为向量。")

    question = st.text_input("请输入你的问题")
    if question:
        docs = vectorstore.similarity_search(question)
        st.write("🤖 回答：", docs[0].page_content.strip())
