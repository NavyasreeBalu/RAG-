import os
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.storage import InMemoryStore
from langchain.retrievers import ParentDocumentRetriever
from dotenv import load_dotenv

load_dotenv()

def load_papers(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Directory {path} not found")
    
    try:
        loader = DirectoryLoader(path, glob="*.pdf", loader_cls=PyPDFLoader)
        documents = loader.load()
        print(f"Loaded {len(documents)} pages from PDF files")
        return documents
    except Exception as e:
        print(f"Error loading papers: {e}")
        return []

def create_parent_document_retriever(docs):
    # This text splitter is used to create the parent documents
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    
    # This text splitter is used to create the child documents
    # It should create documents smaller than the parent
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=100)
    
    # The vectorstore to use to index the child chunks
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(
        collection_name="split_parents",
        embedding_function=embeddings,
        persist_directory="./chroma_db_improved"
    )
    
    # The storage layer for the parent documents
    store = InMemoryStore()
    
    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
    )
    
    retriever.add_documents(docs)
    print("Parent document retriever created and documents added.")
    return retriever

def main():
    print("Starting paper ingestion for the improved approach...")
    docs = load_papers("research_papers/")
    print(f"Successfully loaded {len(docs)} document pages")
    
    retriever = create_parent_document_retriever(docs)
    print("Vector store and document store for improved approach created successfully!")

main()
