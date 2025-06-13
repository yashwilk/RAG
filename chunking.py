from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
import os
from dotenv import load_dotenv
from pprint import pprint
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_openai import OpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


data_dir = "./Big Star Collectibles"

files = os.listdir(data_dir)
file_texts = []
for file in files:
    with open(f"{data_dir}/{file}" ,"r", encoding="utf-8", errors="replace") as f:
        file_text = f.read()
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=128, chunk_overlap=32, 
    )
    texts = text_splitter.split_text(file_text)
    for i, chunked_text in enumerate(texts):
        file_texts.append(Document(page_content=chunked_text,metadata={ 
                    "doc_title": file.split(".")[0], 
                    "chunk_num": i}))


pprint(file_texts[0].page_content)
pprint(file_texts[0].metadata)


embeddings = HuggingFaceEmbeddings()
vector_store = FAISS.from_documents(
    file_texts, 
    embedding=embeddings
)

retriever = vector_store.as_retriever(top_k=4)
retriever.invoke("What year was Big Star Collectibles Started?")



load_dotenv()
llm = OpenAI()

template="""You are a helpful assistant. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
Question: {question} 
Context: {context} 
Answer:"""
prompt = ChatPromptTemplate.from_template(template)

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
response = chain.invoke("When did Big Star Collectibles Launch?")
print(response)
