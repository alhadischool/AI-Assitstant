import warnings
import os
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

from helpers import apply_rtl_styles, render_styled_text
from document_loader import load_and_vectorize_docs

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

# تنظیمات صفحه
st.set_page_config(page_title="دستیار هوشمند الهادی", page_icon="🤖", layout="centered")
apply_rtl_styles()

# کلید API
MY_GEMINI_KEY = st.secrets["GEMINI_API_KEY"]

# لود دیتابیس و مدل هوش مصنوعی
@st.cache_resource
def load_llm():
    if not MY_GEMINI_KEY:
        st.error("⚠️ کلید API جیمینای در Secrets یافت نشد.")
        st.stop()
        
    return ChatGoogleGenerativeAI(
        api_key=MY_GEMINI_KEY, 
        google_api_key=MY_GEMINI_KEY,
        model="gemini-3.7-flash", 
        temperature=0
    )

db = load_and_vectorize_docs()
llm = load_llm()

# سایدبار
st.sidebar.title("📁 انتخاب منابع جستجو")
st.sidebar.write("فایل‌هایی که می‌خواهید هوش مصنوعی در آن‌ها جستجو کند را انتخاب کنید:")

docs_folder = "docs"
available_files = []
if os.path.exists(docs_folder):
    available_files = [f for f in os.listdir(docs_folder) if f.lower().endswith(('.txt', '.pdf', '.docx', '.doc'))]

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
        with st.spinner("در حال جستجو در فایل‌های انتخاب‌شده..."):
            
            if len(selected_files) == 1:
                filter_dict = {"source": selected_files[0]}
            else:
                filter_dict = {"$or": [{"source": f} for f in selected_files]}

            retriever = db.as_retriever(search_kwargs={"k": 20, "filter": filter_dict})
            found_docs = retriever.invoke(user_question)
            
            hidden_text = "\n\n".join([doc.page_content for doc in found_docs])
            final_instructions = prompt_template.format(context=hidden_text, question=user_question)

            # --- مدیریت هوشمند خطاهای API و محدودیت نرخ (Rate Limit) ---   
            try:
                raw_response = llm.invoke(final_instructions)
                
                res_content = raw_response.content if hasattr(raw_response, "content") else str(raw_response)
                if isinstance(res_content, list):
                    clean_answer = "".join([item["text"] if isinstance(item, dict) and "text" in item else str(item) for item in res_content])
                else:
                    clean_answer = str(res_content)
                
                render_styled_text(clean_answer)
                st.session_state.messages.append({"role": "assistant", "content": clean_answer})

            except Exception as e:
                error_str = str(e)
                # بررسی اینکه آیا ارور مربوط به سهمیه یا محدودیت زمان (429 / Rate Limit) است
                if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str or "RateLimit" in error_str:
                    import re
                    # استخراج زمان پیشنهادی برای تلاش مجدد از درون متن ارور (مثلاً 23s)
                    retry_match = re.search(r'retry in ([0-9\.]+)s', error_str)
                    wait_time = "حدود ۱ دقیقه"
                    if retry_match:
                        seconds = float(retry_match.group(1))
                        wait_time = f"حدود {int(seconds)} ثانیه"

                    st.warning(f"⏳ **سهمیه درخواست‌های هوش مصنوعی موقتاً به پایان رسیده است.**\n\nلطفاً **{wait_time}** دیگر مجدداً سوال خود را بپرسید.")
                else:
                    # سایر خطاهای احتمالی
                    st.error(f"❌ خطایی در دریافت پاسخ از هوش مصنوعی رخ داد:\n\n{error_str}")
            
            raw_response = llm.invoke(final_instructions)
            
            res_content = raw_response.content if hasattr(raw_response, "content") else str(raw_response)
            if isinstance(res_content, list):
                clean_answer = "".join([item["text"] if isinstance(item, dict) and "text" in item else str(item) for item in res_content])
            else:
                clean_answer = str(res_content)
            
            render_styled_text(clean_answer)
            st.session_state.messages.append({"role": "assistant", "content": clean_answer})