# api.py
import os
import re
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from document_loader import load_and_vectorize_docs # Your existing file

# Initialize FastAPI
app = FastAPI(title="Al-Hadi Smart Assistant API")

# Allow the frontend to communicate with this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, change this to your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Database and LLM on startup
print("Loading Database and LLM...")
db = load_and_vectorize_docs()
MY_GROQ_KEY = os.environ.get("GROQ_API_KEY", "YOUR_API_KEY_HERE")
llm = ChatGroq(api_key=MY_GROQ_KEY, model_name="llama-3.3-70b-versatile", temperature=0)

# Define the data format we expect from the frontend
class ChatRequest(BaseModel):
    question: str
    selected_files: list[str] = [] # Optional: if you want to keep the file filter feature

prompt_template = PromptTemplate.from_template("""
شما یک دستیار هوشمند، دقیق و مستقیم برای شرکت هستید.
دستورالعمل‌ها:
1. زبان پاسخگویی: دقیقاً به همان زبانی که کاربر سوال پرسیده پاسخ دهید.
2. منبع اصلی: از اطلاعات موجود در متن زیر برای پاسخ به سوال استفاده کنید.
3. خروجی مستقیم: مستقیماً پاسخ نهایی را بنویسید. 

متن اسناد:
{context}

سوال کاربر: {question}

پاسخ نهایی:
""")

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    if db is None:
        raise HTTPException(status_code=500, detail="Database is empty.")
    
    # Handle optional file filtering just like your Streamlit app
    search_kwargs = {"k": 8}
    if request.selected_files:
        if len(request.selected_files) == 1:
            search_kwargs["filter"] = {"source": request.selected_files[0]}
        else:
            search_kwargs["filter"] = {"$or": [{"source": f} for f in request.selected_files]}
            
    # Retrieve documents
    retriever = db.as_retriever(search_kwargs=search_kwargs)
    found_docs = retriever.invoke(request.question)
    hidden_text = "\n\n".join([doc.page_content for doc in found_docs])
    
    # Generate Answer
    final_instructions = prompt_template.format(context=hidden_text, question=request.question)
    
    try:
        raw_response = llm.invoke(final_instructions)
        clean_answer = str(raw_response.content)
        
        # Your existing regex cleanup
        clean_answer = re.sub(r'(?i)(here\'s a thinking process|analyze user input)[\s\S]*?(?=\n[آ-ی1-9]|\n\n|$)', '', clean_answer)
        clean_answer = re.sub(r'<think>[\s\S]*?</think>', '', clean_answer)
        
        return {"answer": clean_answer.strip()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run this file using: uvicorn api:app --reload