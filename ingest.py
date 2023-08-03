import os

from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings
from langchain.vectorstores import FAISS

from dotenv import dotenv_values

if __name__ == "__main__":

    print("Storing documents in local memory.")
    config = dotenv_values(".env")

    cwd = os.getcwd()
    pdf_path_rel = "pdf_source"
    pdf_path_abs = os.path.join(cwd, pdf_path_rel)

    loader = PyPDFDirectoryLoader(path=pdf_path_abs)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100, separator="\n")
    docs = text_splitter.split_documents(documents=documents)
    print("Number of documents: ", len(docs))

    #embeddings = HuggingFaceEmbeddings()
    embeddings = OpenAIEmbeddings(openai_api_key=config["OPENAI_API_KEY"])
    vectorstore = FAISS.from_documents(documents=docs, embedding=embeddings)
    vectorstore.save_local("faiss_index_pdf_1000_100_openAI")