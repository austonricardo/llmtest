import chromadb
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from langchain.vectorstores.chroma import Chroma
from langchain_community import embeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader


# 0.Carrega os docs para carga
loader = DirectoryLoader('./docs', glob="**/*.txt")
docs = loader.load()
print('Quantiadades docs', len(docs))

#print(docs)

# 1.Split data into chunks
#docs_list = [item for sublist in docs for item in sublist]
text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=7500, chunk_overlap=100)
doc_splits = text_splitter.split_documents(docs)

#2. Convert documents to Embeddings and Store them
persist_directory = "./db"
vectorstore = Chroma.from_documents(
    documents=doc_splits,
    collection_name="rag-chroma",
    embedding=embeddings.OllamaEmbeddings(model='nomic-embed-text'),
    persist_directory=persist_directory
)

vectorstore.persist()