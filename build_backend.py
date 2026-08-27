import os
from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Import specialized file loaders
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_community.document_loaders.word_document import Docx2txtLoader

print("1. Reading files from 'docs' folder...")
all_raw_documents = []

# Loop through every file in the docs folder
for filename in os.listdir("docs"):
    file_path = os.path.join("docs", filename)
    
    # --- READ .TXT FILES ---
    if filename.endswith(".txt"):
        with open(file_path, "r", encoding="utf-8") as file:
            text_content = file.read()
            doc = Document(page_content=text_content, metadata={"source": filename, "category": "HR"})
            all_raw_documents.append(doc)
            print(f" Loaded TXT: {filename}")
            
    # --- READ .PDF FILES ---
    elif filename.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
        pdf_docs = loader.load()
        for doc in pdf_docs:
            doc.metadata["category"] = "HR" # Add metadata tag
            all_raw_documents.append(doc)
        print(f" Loaded PDF: {filename} ({len(pdf_docs)} pages)")
        
    # --- READ .DOCX FILES ---
    elif filename.endswith(".docx"):
        loader = Docx2txtLoader(file_path)
        docx_docs = loader.load()
        for doc in docx_docs:
            doc.metadata["category"] = "HR" # Add metadata tag
            all_raw_documents.append(doc)
        print(f" Loaded DOCX: {filename}")

print(f"\nSuccessfully loaded {len(all_raw_documents)} document sections!")

# --- CHOP INTO CHUNKS ---
print("Chopping text into chunks...")
splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = splitter.split_documents(all_raw_documents)

# --- SAVE TO VECTOR DATABASE ---
print("Loading local AI embedding model...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

print("Building vector database...")
db = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./my_database"
)

print("\n🎉 DONE! Your database has been rebuilt with all TXT, PDF, and Word files.")