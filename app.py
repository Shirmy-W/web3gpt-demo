import streamlit as st
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from PyPDF2 import PdfReader
import io

st.set_page_config(page_title="Web3GPT", layout="wide")
st.title("📄 Web3 合约知识问答助手")

uploaded_file = st.file_uploader("上传一份PDF文档", type="pdf")
if uploaded_file is not None:
    pdf_reader = PdfReader(io.BytesIO(uploaded_file.read()))
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    st.success("✅ 文档上传成功，开始构建知识库中...")

    embeddings = OpenAIEmbeddings(model_name="text-embedding-ada-002") 
    texts = [text[i:i+1000] for i in range(0, len(text), 1000)]
    vectorstore = FAISS.from_texts(texts, embeddings)

    st.success("✅ 知识库构建完成！你可以开始提问了")

    question = st.text_input("请输入你的问题")
    if question:
        docs = vectorstore.similarity_search(question)
        st.write("🤖 回答：", docs[0].page_content.strip())
