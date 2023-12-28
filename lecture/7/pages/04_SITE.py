import streamlit as st
from langchain.chat_models import ChatOllama
from langchain.callbacks.base import BaseCallbackHandler
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OllamaEmbeddings, CacheBackedEmbeddings
from langchain.vectorstores import FAISS
from langchain.storage import LocalFileStore
from langchain.prompts import ChatPromptTemplate
from langchain.document_loaders import SitemapLoader
from langchain.document_transformers import Html2TextTransformer


st.set_page_config(
    page_title="SITE",
    page_icon="📄",
)

html2text_transformer = Html2TextTransformer()

st.title("ga111o! SITE")

template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Answer the question using ONLY the following context. If you don't know the answer just say you don't know. DON'T make anything up.
            
            Context: {context}
            """,
        ),
        ("human", "{question}"),
    ]
)

st.markdown("""
    ### INPUT URL ON THE SIDEBAR
""")

with st.sidebar:
    link = st.text_input("INPUT URL HERE!", placeholder="https://ga111o.me")

if link:
    loader = AsyncChromiumLoader([link])
    docs = loader.load()
    transformed = html2text_transformer.transform_documents(docs)
    st.write(docs)
