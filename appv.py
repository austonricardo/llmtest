import gradio as gr
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from langchain.vectorstores.chroma import Chroma
from langchain_community import embeddings
from langchain_community.chat_models import ollama
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import CharacterTextSplitter

urls = [
    "https://ollama.com",
    "https://ollama.com/blog/windows-preview",
    "https://ollama.com/blog/openai-compatibility",
]

def process_input(urls, question):
    model_local = ollama.ChatOllama(model="mistralptbr")
    urls_list = urls.split(" ")

    # 1.Split data into chunks
    docs = [WebBaseLoader(url).load() for url in urls_list]
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

    # 4. After RAG
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
    return after_rag_chain.invoke(question)

#Define Gradio Interface
iface = gr.Interface(fn=process_input,
                     inputs=[gr.Textbox(label="Enter URLs separated by space"), gr.Textbox(label="Question")],
                     outputs="text",
                     title="Document Query with Ollama",
                     description="Enter URLs and a query any on question"
                     )
iface.launch()