from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from langchain.vectorstores.chroma import Chroma
from langchain_community import embeddings
from langchain_community.chat_models import ollama
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import CharacterTextSplitter

model_local = ollama.ChatOllama(model="mistral") #ptbr

#1. Carrega chromaDB
persist_directory = "./db"
vectorstore = Chroma(
    collection_name="rag-chroma",
    persist_directory=persist_directory,
    embedding_function=embeddings.OllamaEmbeddings(model='nomic-embed-text')
)
retriever = vectorstore.as_retriever()

# 3. before RAG
print("Before RAG\n")
before_rag_template = "Quem é o {topic}"
before_rag_prompt = ChatPromptTemplate.from_template(before_rag_template)
before_rag_chain = before_rag_prompt | model_local | StrOutputParser()
print(before_rag_chain.invoke({"topic":"Presidente do Tribunal Regional do Trabalho da 6a região"}))

# 4. After RAG
print("\n########\nAfter RAG\n")
after_rag_template = """ Answer the question based only on the following context:
{context}
Question: {question}
"""
after_rag_prompt = ChatPromptTemplate.from_template(after_rag_template)
after_rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | after_rag_prompt
    | model_local
    | StrOutputParser()
)
print(after_rag_chain.invoke("Quem é o Presidente do Tribunal Regional do Trabalho da 6a região?"))