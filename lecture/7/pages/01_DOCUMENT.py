import streamlit as st
import time
from langchain.chat_models import ChatOllama
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OllamaEmbeddings, CacheBackedEmbeddings
from langchain.vectorstores import FAISS
from langchain.storage import LocalFileStore
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough

st.set_page_config(
    page_title= "DOCUMENT",
    page_icon="📄",
)

def embed_file(file):
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    cache_dir = LocalFileStore(f"./.cache/embedings/{file.name}")
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader("./.cache/files/Chaptor One.txt")
    docs = loader.load_and_split(text_splitter=splitter)
    embeddings = OllamaEmbeddings()
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    retriever = vectorstore.as_retriever()
    return retriever

st.title("ga111o! DOCUMENT")

st.markdown("""
            # welcome!

            ### this is ga111o! DOCUMENT!
            
            upload documents!
""")


file = st.file_uploader("upload file", type=["pdf","txt","docs","jpg","png"])



if file:
    retriever = embed_file(file)
    s = retriever.invoke("winston")
    s