import streamlit as st
import PyPDF2
import re
import os
from openai import OpenAI
from transformers import PegasusForConditionalGeneration, PegasusTokenizer

client = OpenAI(
    api_key=st.secrets["GEMINI_API_KEY"],
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

st.set_page_config(
    page_title="LegalEase",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown("""
    <style>
    .main-header {
        text-align: center;
        padding: 1.5rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        animation: fadeIn 0.3s ease-in;
    }
    .user-message {
        background-color: #e8f4f8;
        border-left: 4px solid #2196F3;
    }
    .assistant-message {
        background-color: #f5f5f5;
        border-left: 4px solid #764ba2;
    }
    .summary-box {
        background-color: #f0f7ff;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #667eea;
        margin: 1.5rem 0;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        font-weight: 600;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <div class="main-header">
        <h1>⚖️ LegalEase - Legal Document Summarizer</h1>
        <p style="margin: 0; font-size: 1.1rem;">AI-Powered Legal Document Analysis</p>
    </div>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    try:
        model = PegasusForConditionalGeneration.from_pretrained(
         "Shivani0906/LegalEase-Pegasus",
            subfolder="checkpoint-680"
        )

        tokenizer = PegasusTokenizer.from_pretrained(
             "Shivani0906/LegalEase-Pegasus",
                subfolder="checkpoint-680"
        )
        return model, tokenizer
    except Exception as e:
        st.error(f"Model loading failed from Hugging Face: {str(e)}")
        return None, None


def clean_legal_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\s+([.,;:!?])', r'\1', text)
    text = re.sub(r'([.,;:!?])\s+', r'\1 ', text)
    return text.strip()


def legal_chunking(text, max_tokens=384):
    words = text.split()
    total_words = len(words)
    if total_words <= max_tokens:
        return text
    else:
        first_part = words[:int(total_words * 0.3)]
        last_part = words[-int(total_words * 0.6):]
        return " ".join(first_part + last_part)


def generate_summary(text, model, tokenizer):
    cleaned_text = clean_legal_text(text)
    processed_text = legal_chunking(cleaned_text)
    inputs = tokenizer(processed_text, return_tensors="pt", max_length=384, truncation=True)
    summary_ids = model.generate(
        inputs["input_ids"],
        max_length=500,
        min_length=300,
        num_beams=4,
        early_stopping=True,
        length_penalty=1.0,
    )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)


def extract_pdf_text(uploaded_file):
    try:
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + " "
        return text.strip()
    except Exception as e:
        raise Exception(f"PDF extraction error: {str(e)}")


def ask_gemini(document_text, summary_text, question, chat_history):
    try:
        prompt = f"""
        You are a legal assistant. Use the provided legal document and summary to answer questions.

        LEGAL DOCUMENT:
        {document_text[:12000]}

        DOCUMENT SUMMARY:
        {summary_text}

        CHAT HISTORY:
        {chat_history}

        USER QUESTION: {question}

        RULES:
        - Give SHORT answers (2-3 sentences) unless user specifically asks for "detailed", "explain", or "elaborate"
        - Focus on key legal points: parties, judgments, terms, conditions, dates
        - Be precise and factual from the document
        - If information is not in the document, say "This information is not clearly stated in the document"
        - For detailed requests, provide comprehensive but concise explanations

        Answer:
        """
        response = client.chat.completions.create(
            model="gemini-2.0-flash",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"


def main():
    st.markdown("### 📄 Upload Document")
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type="pdf",
        help="Upload a legal document in PDF format",
        label_visibility="collapsed"
    )

    if uploaded_file is not None:
        try:
            with st.spinner(" Extracting text from PDF..."):
                extracted_text = extract_pdf_text(uploaded_file)

            if not extracted_text or len(extracted_text.strip()) < 50:
                st.error("No readable text found in PDF. Please ensure the PDF contains selectable text.")
                return

            st.success("PDF processed successfully!")

            with st.expander("View Document Preview"):
                preview_text = extracted_text[:1500] + "..." if len(extracted_text) > 1500 else extracted_text
                st.text_area("Document Preview", preview_text, height=200, disabled=True)

            st.markdown("---")

            if st.button("Generate Summary", type="primary"):
                with st.spinner("Generating summary..."):
                    model, tokenizer = load_model()
                    if model is None or tokenizer is None:
                        st.error("Failed to load model from Hugging Face.")
                        return
                    summary = generate_summary(extracted_text, model, tokenizer)
                    st.session_state.document_text = extracted_text
                    st.session_state.summary_text = summary
                    st.session_state.summary_generated = True
                    st.rerun()

            if st.session_state.get('summary_generated', False):
                st.markdown("### 📋 Document Summary")
                st.markdown(f"""
                    <div class="summary-box">
                        {st.session_state.summary_text}
                    </div>
                """, unsafe_allow_html=True)

                if 'chat_history' not in st.session_state:
                    st.session_state.chat_history = []

                st.markdown("---")
                st.markdown("### Ask Chatbot if any further questions")

                chat_container = st.container()
                with chat_container:
                    for i, (q, a) in enumerate(st.session_state.chat_history):
                        st.markdown(f"""
                            <div class="chat-message user-message">
                                <strong>🙋 You:</strong><br>{q}
                            </div>
                        """, unsafe_allow_html=True)
                        st.markdown(f"""
                            <div class="chat-message assistant-message">
                                <strong>🤖 Assistant:</strong><br>{a}
                            </div>
                        """, unsafe_allow_html=True)

                col1, col2 = st.columns([5, 1])
                with col1:
                    question = st.text_input(
                        "Type your question here...",
                        key=f"chat_input_{len(st.session_state.chat_history)}",
                        placeholder="e.g., What are the key terms of this agreement?",
                        label_visibility="collapsed"
                    )
                with col2:
                    send_button = st.button("Send", use_container_width=True)

                if (question and send_button) or (question and st.session_state.get('enter_pressed', False)):
                    with st.spinner("Analyzing..."):
                        answer = ask_gemini(
                            st.session_state.document_text,
                            st.session_state.summary_text,
                            question,
                            "\n".join([f"Q: {q}\nA: {a}" for q, a in st.session_state.chat_history[-4:]])
                        )
                        st.session_state.chat_history.append((question, answer))
                        st.session_state.enter_pressed = False
                        st.rerun()

                if st.session_state.chat_history:
                    if st.button("Clear Chat History"):
                        st.session_state.chat_history = []
                        st.rerun()

        except Exception as e:
            st.error(f"Error processing document: {str(e)}")


if __name__ == "__main__":
    main()
