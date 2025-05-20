import streamlit as st
import io
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI

st.set_page_config(page_title="Web3GPT", layout="wide")

st.title("📘 Web3GPT - 区块链知识问答助手")
openai_api_key = st.sidebar.text_input("🔑 请输入你的 OpenAI API Key", type="password")

uploaded_file = st.file_uploader("📄 上传你的区块链知识 PDF 文件", type="pdf")

if uploaded_file and openai_api_key:
    # ✅ 读取上传 PDF 文件内容（兼容 Streamlit）
    pdf_reader = PdfReader(io.BytesIO(uploaded_file.read()))
    raw_text = ""
    for page in pdf_reader.pages:
        raw_text += page.extract_text() or ""

    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_text(raw_text)

    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectorstore = FAISS.from_texts(texts, embeddings)

    query = st.text_input("💬 输入你的问题（基于上传内容）")
    if query:
        llm = OpenAI(openai_api_key=openai_api_key, temperature=0)
        chain = load_qa_chain(llm, chain_type="stuff")
        docs = vectorstore.similarity_search(query)
        response = chain.run(input_documents=docs, question=query)
        st.markdown("### 🤖 回答：")
        st.write(response)
elif not openai_api_key:
    st.info("🔐 请在左侧输入你的 OpenAI API Key")
