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
    link = st.text_input(
        "INPUT URL HERE!",
        placeholder="https://ga111o.me"
    )


def parse_page(soup):
    header = soup.find("header")
    footer = soup.find("footer")
    if header:
        header.decompose()
    if footer:
        footer.decompose()
    return str(soup.get_text()).replace("\n", " ").replace("\xa0", " ")


@st.cache_data
def load_website(link):
    loader = SitemapLoader(
        link,
        parsing_function=parse_page
    )
    loader.requests_per_second = 5
    docs = loader.load()
    return docs


if link:
    if ".xml" not in link:
        with st.sidebar:
            st.error("need to SITEMAP URL")
    else:
        docs = load_website(link)
        st.write(docs)
