from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

print("Checking database contents...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = Chroma(persist_directory="./my_database", embedding_function=embeddings)

# Fetch all stored items
collection = db._collection
count = collection.count()

print(f"\nTotal document chunks in database: {count}")

if count > 0:
    results = db.similarity_search("company", k=2)
    print(f"\nSample search test (found {len(results)} matches):")
    for i, doc in enumerate(results, 1):
        print(f"\n--- Chunk {i} ---")
        print(doc.page_content[:150])
else:
    print("❌ Your database is completely empty! Re-run build_backend.py.")