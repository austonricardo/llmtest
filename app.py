from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from langchain.vectorstores.chroma import Chroma
from langchain_community import embeddings
from langchain_community.chat_models import ollama
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import CharacterTextSplitter

model_local = ollama.ChatOllama(model="mistralptbr")



urls = [
    "https://www.trt6.jus.br/portal/institucional/composicao"
]

# 1.Split data into chunks
docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]
text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=7500, chunk_overlap=100)
doc_splits = text_splitter.split_documents(docs_list)


#2. Convert documents to Embeddings and Store them
vectorstore = Chroma.from_documents(
    documents=doc_splits,
    collection_name="rag-chroma",
    embedding=embeddings.OllamaEmbeddings(model='nomic-embed-text')
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
