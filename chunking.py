from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
import os

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

from pprint import pprint

pprint(file_texts[0].page_content)
pprint(file_texts[0].metadata)

from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings()
vector_store = FAISS.from_documents(
    file_texts, 
    embedding=embeddings
)

retriever = vector_store.as_retriever(top_k=4)
retriever.invoke("What year was Big Star Collectibles Started?")
