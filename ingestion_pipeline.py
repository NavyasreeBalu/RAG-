import os
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

def validate_document(doc):
    return len(doc.page_content.strip()) > 50 and not doc.page_content.isspace()

def extract_section_header(text):
    lines = text.split('\n')[:3]
    for line in lines:
        if line.strip() and (line.isupper() or any(word in line.lower() for word in ['abstract', 'introduction', 'conclusion', 'method'])):
            return line.strip()[:50]
    return "content"

def load_papers(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Directory {path} not found")
    
    try:
        loader = DirectoryLoader(path, glob="*.pdf", loader_cls=PyPDFLoader)
        documents = loader.load()
        
        valid_docs = [doc for doc in documents if validate_document(doc)]
        print(f"Loaded {len(valid_docs)} valid pages from {len(documents)} total pages")
        return valid_docs
    except Exception as e:
        print(f"Error loading papers: {e}")
        return []

def split_documents(docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500, 
        chunk_overlap=300, 
        add_start_index=True,  
    )
    all_splits = text_splitter.split_documents(docs)
    
    for i, chunk in enumerate(all_splits):
        source = chunk.metadata.get('source', 'unknown').split('/')[-1]
        chunk.metadata.update({
            'chunk_id': f"{source}_{i}",
            'total_chunks': len(all_splits),
            'section': extract_section_header(chunk.page_content)
        })
    
    print(f"Split documents into {len(all_splits)} chunks.")
    return all_splits

def create_vector_store(splits):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    print(f"Created vector store with {len(splits)} documents")
    return vectorstore

def main():
    print("Starting paper ingestion...")
    docs = load_papers("research_papers/")
    print(f"Successfully loaded {len(docs)} document pages")
    
    splits = split_documents(docs)
    print(f"Created {len(splits)} text chunks")
    
    vectorstore = create_vector_store(splits)
    print("Vector store created successfully!")

main()