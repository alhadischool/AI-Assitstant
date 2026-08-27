import warnings
import os
import re
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

import streamlit as st
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
# ==============================================================================
# 🔑 PASTE YOUR GEMINI API KEY HERE
# ==============================================================================
MY_GEMINI_KEY = st.secrets["GEMINI_API_KEY"]
# ==============================================================================
# --- 1. تنظیمات صفحه و استایل کامل RTL و مرتب‌سازی سایدبار ---
st.set_page_config(page_title="دستیار هوشمند الهادی", page_icon="🤖", layout="centered")

st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Vazirmatn:wght@400;700&display=swap" rel="stylesheet">

    <style>
        /* 1. اعمال فونت وزیرمتن فقط به متون (بدون دستکاری آیکون‌های سیستمی) */
        h1, h2, h3, h4, p, label, .stMarkdown, .stText, 
        [data-testid="stSidebarUserContent"], 
        [data-testid="stChatMessageContent"], 
        .stChatInput textarea {
            font-family: 'Vazirmatn', sans-serif !important;
        }

        /* 2. راست‌چین کردن محتوای صفحه، سایدبار و چت (بدون خراب کردن فیزیک سایدبار) */
        .block-container, 
        [data-testid="stSidebarUserContent"],
        .stChatInput textarea,
        [data-testid="stChatMessageContent"] {
            direction: rtl !important;
            text-align: right !important;
        }

        /* 3. راست‌چین کردن عناوین، متن‌ها و چک‌باکس‌ها */
        h1, h2, h3, p, .stMarkdown, .stCheckbox {
            direction: rtl !important;
            text-align: right !important;
        }

        /* 4. تنظیم فاصله خطوط متون برای خوانایی بهتر */
        p, div {
            line-height: 1.8;
        }
    </style>
""", unsafe_allow_html=True)

# --- 2. تابع تشخیص زبان متن ---
def render_styled_text(text: str):
    if re.search(r'[\u0600-\u06FF]', text):
        st.markdown(f'<div style="direction: rtl; text-align: right;">{text}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div style="direction: ltr; text-align: left;">{text}</div>', unsafe_allow_html=True)

# --- 3. بارگذاری هوشمند اسامی فایل‌ها در سایدبار ---
st.sidebar.title("📁 انتخاب منابع جستجو")
st.sidebar.write("فایل‌هایی که می‌خواهید هوش مصنوعی در آن‌ها جستجو کند را انتخاب کنید:")

# دریافت لیست تمام فایل‌های موجود در پوشه docs
docs_folder = "docs"
available_files = []
if os.path.exists(docs_folder):
    available_files = [f for f in os.listdir(docs_folder) if f.endswith(('.txt', '.pdf', '.docx'))]

selected_files = []
if available_files:
    # ساخت یک چک‌باکس به ازای هر فایل
    for file_name in available_files:
        if st.sidebar.checkbox(file_name, value=True, key=file_name):
            selected_files.append(file_name)
else:
    st.sidebar.warning("هیچ فایلی در پوشه docs یافت نشد!")

# --- 4. عنوان اصلی و تاریخچه چت ---
st.title("دستیار هوشمند الهادی 🤖")
st.write("سوال خود را بپرسید! من پاسخ‌ها را با تحلیل اسناد انتخاب‌شده پیدا می‌کنم.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        render_styled_text(msg["content"])

# --- 5. بارگذاری ابزارها ---
@st.cache_resource
def load_tools():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    db = Chroma(persist_directory="./my_database", embedding_function=embeddings)
    
    if not MY_GEMINI_KEY or "PASTE_YOUR" in MY_GEMINI_KEY:
        st.error("⚠️ لطفاً کلید API جیمینای خود را در فایل app.py قرار دهید.")
        st.stop()
        
    llm = ChatGoogleGenerativeAI(
        api_key=MY_GEMINI_KEY, 
        google_api_key=MY_GEMINI_KEY,
        model="gemini-3.6-flash", 
        temperature=0
    )
    return db, llm

db, llm = load_tools()

# --- 6. پرامپت سیستم ---
prompt_template = PromptTemplate.from_template("""
شما یک دستیار هوشمند و دقیق برای شرکت هستید.

دستورالعمل‌ها:
1. زبان پاسخگویی: دقیقاً به همان زبانی که کاربر سوال پرسیده پاسخ دهید.
2. منبع اصلی: از اطلاعات موجود در متن زیر برای پاسخ به سوال استفاده کنید.
3. تحلیل و شمارش: شما مجاز هستید موارد را بشمارید، لیست‌ها را خلاصه کنید یا بر اساس متن تحلیل انجام دهید.
4. عدم اشاره به ساختار: هرگز از جملاتی مثل "طبق متن داده شده" یا "بر اساس اسناد" استفاده نکنید.

متن اسناد:
{context}

سوال کاربر: {question}

پاسخ:
""")

# --- 7. دریافت سوال و فیلتر بر اساس فایل‌های انتخاب‌شده ---
user_question = st.chat_input("سوال خود را اینجا بنویسید...")

if user_question:
    if not selected_files:
        st.warning("لطفاً حداقل یک فایل را از سایدبار انتخاب کنید.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        render_styled_text(user_question)

    with st.chat_message("assistant"):
        with st.spinner("در حال جستجو در فایل‌های انتخاب‌شده..."):
            
            # ایجاد فیلتر بر اساس نام فایل‌های انتخاب‌شده در سایدبار
            if len(selected_files) == 1:
                filter_dict = {"source": selected_files[0]}
            else:
                filter_dict = {"$or": [{"source": f} for f in selected_files]}

            # دریافت اسناد با اعمال فیلتر سایدبار
            retriever = db.as_retriever(search_kwargs={"k": 20, "filter": filter_dict})
            found_docs = retriever.invoke(user_question)

            # --- بخش عیب‌یابی دیتابیس (با فاصله‌گذاری استاندارد) ---
        with st.expander("🔍 رادیولوژی دیتابیس (چه چیزی پیدا شد؟)"):
            st.write(f"تعداد تکه‌های پیدا شده: {len(found_docs)}")
            for i, doc in enumerate(found_docs):
                st.info(f"تکه {i+1}: {doc.page_content}")
            
            hidden_text = "\n\n".join([doc.page_content for doc in found_docs])
            final_instructions = prompt_template.format(context=hidden_text, question=user_question)
            
            raw_response = llm.invoke(final_instructions)
            
            if hasattr(raw_response, "content"):
                res_content = raw_response.content
                if isinstance(res_content, list):
                    clean_answer = "".join([
                        item["text"] if isinstance(item, dict) and "text" in item else str(item) 
                        for item in res_content
                    ])
                else:
                    clean_answer = str(res_content)
            else:
                clean_answer = str(raw_response)
            
            render_styled_text(clean_answer)
            st.session_state.messages.append({"role": "assistant", "content": clean_answer})