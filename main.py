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

    embeddings = OpenAIEmbeddings(openai_api_key=config["OPENAI_API_KEY"])
    vectorstore = FAISS.from_documents(documents=docs, embedding=embeddings)
    vectorstore.save_local("faiss_index_ml_pdf_1000_100")

    new_vectorstore = FAISS.load_local(folder_path="faiss_index_ml_pdf_1000_100", embeddings=embeddings)
    qa = RetrievalQA.from_chain_type(llm=OpenAI(openai_api_key=config["OPENAI_API_KEY"]), chain_type="stuff", retriever=new_vectorstore.as_retriever())
    res = qa.run(query="""
    Query here
    """
    )
    print(res)
