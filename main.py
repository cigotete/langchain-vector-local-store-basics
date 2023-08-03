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
    embeddings = OpenAIEmbeddings(openai_api_key=config["OPENAI_API_KEY"])
    vectorstore = FAISS.load_local(folder_path="faiss_index_pdf_1000_100_openAI", embeddings=embeddings)

    qa = RetrievalQA.from_chain_type(llm=OpenAI(openai_api_key=config["OPENAI_API_KEY"]), chain_type="stuff", retriever=vectorstore.as_retriever())
    res = qa.run(query="""
    Query here
    """
    )
    print(res)
