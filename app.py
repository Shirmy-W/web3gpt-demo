
import streamlit as st
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
import os

st.set_page_config(page_title="Web3GPT - 区块链知识问答", layout="wide")

st.title("📘 Web3GPT - 区块链知识问答助手")
st.write("上传你的区块链 PDF 文档，提出问题，让 GPT 为你回答。")

api_key = st.text_input("🔑 输入你的 OpenAI API Key", type="password")

uploaded_file = st.file_uploader("📄 上传区块链 PDF 文件", type="pdf")
question = st.text_input("💬 你想问什么？")

if uploaded_file and api_key and question:
    os.environ["OPENAI_API_KEY"] = api_key

    pdf_reader = PdfReader(uploaded_file)
    raw_text = ""
    for page in pdf_reader.pages:
        raw_text += page.extract_text()

    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_text(raw_text)

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts, embeddings)
    docs = vectorstore.similarity_search(question)

    llm = OpenAI(temperature=0)
    chain = load_qa_chain(llm, chain_type="stuff")
    response = chain.run(input_documents=docs, question=question)

    st.write("🧠 GPT 的回答：")
    st.success(response)
