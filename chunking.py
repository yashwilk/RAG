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