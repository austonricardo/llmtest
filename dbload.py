import chromadb
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from langchain.vectorstores.chroma import Chroma
from langchain_community import embeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader

#1. Carrega chromaDB
persist_directory = "./db"
vectorstore = Chroma(
    collection_name="rag-chroma",
    persist_directory=persist_directory,
    embedding_function=embeddings.OllamaEmbeddings(model='nomic-embed-text')
)
vectorstore.as_retriever()

print('Vectorstore', len(vectorstore))