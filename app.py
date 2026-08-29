import warnings
import os
import re
import streamlit as st
from groq import Groq
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

from helpers import apply_rtl_styles, render_styled_text
from document_loader import load_and_vectorize_docs

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

# تنظیمات صفحه
st.set_page_config(page_title="دستیار هوشمند الهادی", page_icon="🤖", layout="centered")
apply_rtl_styles()

MY_GROQ_KEY = st.secrets.get("GROQ_API_KEY", "")

# لود هوشمند مدل‌های فعال Groq
@st.cache_resource
def load_llm():
    if not MY_GROQ_KEY:
        st.error("⚠️ کلید API گراک (GROQ_API_KEY) در Secrets یافت نشد.")
        st.stop()
        
    try:
        client = Groq(api_key=MY_GROQ_KEY)
        all_models = [m.id for m in client.models.list().data]
        
        # ۱. فیلتر کردن مدل‌های غیرچت (مثل گارد، صوتی و تصویر)
        chat_models = [
            m for m in all_models 
            if not any(bad in m.lower() for bad in ["guard", "whisper", "vision", "audio", "embed"])
        ]
        
        # ۲. انتخاب اولویت‌دار از میان مدل‌های چت متنی فعال
        chosen_model = None
        for key in ["3.3", "70b", "3.1", "qwen", "mixtral", "8b"]:
            for m in chat_models:
                if key in m.lower():
                    chosen_model = m
                    break
            if chosen_model:
                break
                
        if not chosen_model:
            chosen_model = chat_models[0] if chat_models else all_models[0]
                
        st.sidebar.caption(f"🤖 مدل فعال: `{chosen_model}`")
        
        return ChatGroq(
            api_key=MY_GROQ_KEY,
            model_name=chosen_model,
            temperature=0
        )
    except Exception as e:
        st.error(f"خطا در دریافت لیست مدل‌های Groq: {e}")
        st.stop()

db = load_and_vectorize_docs()
llm = load_llm()

# سایدبار
st.sidebar.title("📁 انتخاب منابع جستجو")
st.sidebar.write("فایل‌هایی که می‌خواهید هوش مصنوعی در آن‌ها جستجو کند را انتخاب کنید:")

docs_folder = "docs"
available_files = []
if os.path.exists(docs_folder):
    available_files = [f for f in os.listdir(docs_folder) if f.lower().endswith(('.txt', '.pdf', '.docx'))]

selected_files = []
if available_files:
    for file_name in available_files:
        if st.sidebar.checkbox(file_name, value=True, key=file_name):
            selected_files.append(file_name)
else:
    st.sidebar.warning("هیچ فایلی در پوشه docs یافت نشد!")

# چت روم
st.title("دستیار هوشمند الهادی 🤖")
st.write("سوال خود را بپرسید! من پاسخ‌ها را با تحلیل اسناد انتخاب‌شده پیدا می‌کنم.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        render_styled_text(msg["content"])

prompt_template = PromptTemplate.from_template("""
شما یک دستیار هوشمند، دقیق و مستقیم برای شرکت هستید.

دستورالعمل‌ها:
1. زبان پاسخگویی: دقیقاً به همان زبانی که کاربر سوال پرسیده پاسخ دهید.
2. منبع اصلی: از اطلاعات موجود در متن زیر برای پاسخ به سوال استفاده کنید.
3. خروجی مستقیم: مستقیماً پاسخ نهایی را بنویسید. به هیچ عنوان روند فکر کردن (Thinking process)، پیش‌فرضیات یا تحلیل‌های اولیه خود را در خروجی نیاورید.
4. عدم اشاره به ساختار: هرگز از جملاتی مثل "طبق متن داده شده" یا "بر اساس اسناد" استفاده نکنید.

متن اسناد:
{context}

سوال کاربر: {question}

پاسخ نهایی:
""")

user_question = st.chat_input("سوال خود را اینجا بنویسید...")

if user_question:
    if not selected_files:
        st.warning("لطفاً حداقل یک فایل را از سایدبار انتخاب کنید.")
        st.stop()

    if db is None:
        st.error("دیتابیس خالی است یا فایلی خوانده نشده است.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        render_styled_text(user_question)

    with st.chat_message("assistant"):
        with st.spinner("در حال جستجو و تحلیل اسناد..."):
            
            if len(selected_files) == 1:
                filter_dict = {"source": selected_files[0]}
            else:
                filter_dict = {"$or": [{"source": f} for f in selected_files]}

            # بازگرداندن جستجو به ۸ تکه برای پوشش کامل تمام اسناد
            retriever = db.as_retriever(search_kwargs={"k": 8, "filter": filter_dict})
            found_docs = retriever.invoke(user_question)
            
            hidden_text = "\n\n".join([doc.page_content for doc in found_docs])
            final_instructions = prompt_template.format(context=hidden_text, question=user_question)
            
            try:
                raw_response = llm.invoke(final_instructions)
                
                res_content = raw_response.content if hasattr(raw_response, "content") else str(raw_response)
                if isinstance(res_content, list):
                    clean_answer = "".join([item["text"] if isinstance(item, dict) and "text" in item else str(item) for item in res_content])
                else:
                    clean_answer = str(res_content)
                
                # فیلتر اختصاصی جهت حذف هرگونه متن فکر کردن (Thinking Process / <think>)
                clean_answer = re.sub(r'(?i)here\'s a thinking process:[\s\S]*?(?=\n\n|\n[آ-یA-Z]|$)', '', clean_answer)
                clean_answer = re.sub(r'<think>[\s\S]*?</think>', '', clean_answer).strip()
                
                render_styled_text(clean_answer)
                st.session_state.messages.append({"role": "assistant", "content": clean_answer})
            
            except Exception as e:
                st.error(f"❌ خطایی در دریافت پاسخ رخ داد:\n\n{str(e)}")