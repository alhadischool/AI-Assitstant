import warnings
import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

from helpers import apply_rtl_styles, render_styled_text
from document_loader import load_and_vectorize_docs

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

# تنظیمات صفحه
st.set_page_config(page_title="دستیار هوشمند الهادی", page_icon="🤖", layout="centered")
apply_rtl_styles()

# دریافت کلید API گراک
MY_GROQ_KEY = st.secrets.get("GROQ_API_KEY", "")

# لود دیتابیس و مدل Groq
@st.cache_resource
def load_llm():
    if not MY_GROQ_KEY:
        st.error("⚠️ کلید API گراک (GROQ_API_KEY) در Secrets یافت نشد.")
        st.stop()
        
    return ChatGroq(
        api_key=MY_GROQ_KEY,
        model_name="llama-3.1-8b-instant",
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

            # جستجو با 5 تکه متنی برای دقت عالی
            retriever = db.as_retriever(search_kwargs={"k": 5, "filter": filter_dict})
            found_docs = retriever.invoke(user_question)

            with st.expander("🔍 رادیولوژی دیتابیس (چه چیزی پیدا شد؟)"):
                st.write(f"تعداد تکه‌های پیدا شده: {len(found_docs)}")
                for i, doc in enumerate(found_docs):
                    st.info(f"تکه {i+1} (از فایل {doc.metadata.get('source', 'نامشخص')}):\n{doc.page_content}")
            
            hidden_text = "\n\n".join([doc.page_content for doc in found_docs])
            final_instructions = prompt_template.format(context=hidden_text, question=user_question)
            
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
                st.error(f"❌ خطایی در فراخوانی سرویس Groq رخ داد:\n\n{str(e)}")