import os

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

from dotenv import dotenv_values

if __name__ == "__main__":

    print("Hello Vector store in local memory.")
    config = dotenv_values(".env")

    cwd = os.getcwd()
    pdf_path_rel = "pdf_source/ml.pdf"
    pdf_path_abs = os.path.join(cwd, pdf_path_rel)

    loader = PyPDFLoader(file_path=pdf_path_abs)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100, separator="\n")
    docs = text_splitter.split_documents(documents=documents)
    print("Number of documents: ", len(docs))
