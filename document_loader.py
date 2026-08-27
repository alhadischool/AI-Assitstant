import os
import streamlit as st
from langchain_community.document_loaders import TextLoader, Docx2txtLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

@st.cache_resource
def load_and_vectorize_docs():
    all_documents = []
    docs_folder = './docs'
    
    if os.path.exists(docs_folder):
        for root, _, files in os.walk(docs_folder):
            for file in files:
                file_path = os.path.join(root, file)
                ext = file.lower().split('.')[-1]
                
                try:
                    docs = []
                    if ext == 'txt':
                        loader = TextLoader(file_path, encoding='utf-8')
                        docs = loader.load()
                    elif ext in ['docx', 'doc']:
                        loader = Docx2txtLoader(file_path)
                        docs = loader.load()
                    elif ext == 'pdf':
                        loader = PyPDFLoader(file_path)
                        docs = loader.load()
                    
                    # ثبت دقیق نام فایل در متادیتا برای فیلتر سایدبار
                    for doc in docs:
                        doc.metadata["source"] = file
                    
                    all_documents.extend(docs)
                except Exception as e:
                    st.warning(f"خطا در خواندن فایل {file}: {e}")
                    
    if not all_documents:
        return None

    # تکه‌تکه کردن متن‌ها
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(all_documents)

    # ساخت دیتابیس در حافظه
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    vector_store = Chroma.from_documents(documents=splits, embedding=embeddings)
    
    return vector_store