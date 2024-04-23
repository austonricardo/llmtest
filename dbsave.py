from typing import List
import chromadb
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from langchain.vectorstores.chroma import Chroma
from langchain_community import embeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.docstore import document

# 0.Carrega os docs para carga
loader = DirectoryLoader('./docs', glob="**/*.txt")
docs = loader.load()
print('Quantiadades docs', len(docs))

#print(docs)

# 1.Split data into chunks
#docs_list = [item for sublist in docs for item in sublist]
text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=7500, chunk_overlap=100)
doc_splits = text_splitter.split_documents(docs)

from transformers import AutoTokenizer, AutoModel
import torch

# Carregar o tokenizer e o modelo de embeddings
tokenizer = AutoTokenizer.from_pretrained("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True, safe_serialization=True)
model = AutoModel.from_pretrained("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True, safe_serialization=True)

from typing import List
class LocalEmbeddings(chromadb.Embeddings):
    def __init__(self,  tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model

    def embed_documents(self, documents) -> List[List[float]]:
        embeddings = []
        for doc in documents:
            # Tokenizar o documento
            inputs = self.tokenizer(doc, return_tensors="pt", padding=True, truncation=True)

            # Obter os embeddings do modelo
            with torch.no_grad():
                outputs = self.model(**inputs)

            # Extrair os embeddings da última camada oculta
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze()
            embeddings.append(embedding.tolist())
        
        return embeddings

embedding_function=LocalEmbeddings(tokenizer, model)


#2. Convert documents to Embeddings and Store them
persist_directory = "./db"
vectorstore = Chroma.from_documents(
    documents=doc_splits,
    collection_name="rag-chroma",
    embedding=embedding_function,
    persist_directory=persist_directory
)
vectorstore.as_retriever()
print('Vectorstore', len(vectorstore))

vectorstore.persist()
print('Adicionando novo doc de teste')
doc = {page_content: 'teste de conteudo', 'metadata': {'file': 'teste.txt', 'processo':'123456'}}(Document)
vectorstore.add_documents([doc])

print('Vectorstore', len(vectorstore))