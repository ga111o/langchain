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

if "messages" not in st.session_state:
    st.session_state["messages"] = []

@st.cache_data(show_spinner="Embedding file...")
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
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    embeddings = OllamaEmbeddings()
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    retriever = vectorstore.as_retriever()
    return retriever

def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        st.session_state["messages"].append({"messages": message, "role":role})


st.title("ga111o! DOCUMENT")

st.markdown("""
    ### UPLOAD FILE ON THE SIDEBAR
""")

with st.sidebar:
    file = st.file_uploader("upload file", type=["pdf","txt","docs","jpg","png"])



if file:
    retriever = embed_file(file)
    send_message("READY!", "ai")
    message = st.chat_input("")
    if message:
        send_message(message, "human")
    