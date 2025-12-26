import streamlit as st
from langchain_community.llms import Ollama


st.sidebar.title("⚙️ Mode")
mode = st.sidebar.radio(
    "Select Mode",
    ["💬 Chat Mode", "🎯 Interview Mode"]
)

# Page config
st.set_page_config(
    page_title="DevOps AI Assistant",
    page_icon="🤖",
    layout="centered"
)

# Load model

@st.cache_resource
def load_llm():
    return Ollama(model="llama3")

llm = load_llm()

# Theme selector
theme = st.toggle("🌙 Dark Mode")

# Custom CSS
if theme:  # DARK MODE
    st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        color: #ffffff;
    }

    .chat-box {
        background-color: #1e1e2f;
        color: #ffffff;
        padding: 14px;
        border-radius: 12px;
        margin-bottom: 10px;
        border-left: 4px solid #6a5acd;
    }

    input {
        color: white !important;
    }
    </style>
    """, unsafe_allow_html=True)

else:  # LIGHT MODE
    st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #f5f7fa, #c3cfe2);
        color: #000000;
    }

    .chat-box {
        background-color: #ffffff;
        color: #000000;
        padding: 14px;
        border-radius: 12px;
        margin-bottom: 10px;
        border-left: 4px solid #6a5acd;
        box-shadow: 0px 4px 10px rgba(0,0,0,0.1);
    }

    input {
        color: black !important;
    }
    </style>
    """, unsafe_allow_html=True)


st.title("🤖 DevOps AI Assistant")
st.caption("Docker • Kubernetes • Jenkins • Terraform • AWS")

# Session state for chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat
for msg in st.session_state.messages:
    with st.container():
        st.markdown(
            f"<div class='chat-box'><b>{msg['role']}:</b> {msg['content']}</div>",
            unsafe_allow_html=True
        )

# Input
user_input = st.text_input("Ask a DevOps question", key="input")

SYSTEM_PROMPT = """
You are a DevOps expert.
Answer only DevOps-related questions.
If question is not DevOps related, say:
'I answer only DevOps questions.'
"""

INTERVIEW_PROMPT = """
You are a senior DevOps interviewer.
Ask one DevOps interview question.
Wait for the candidate answer.
Evaluate the answer.
Give feedback and the correct answer.
"""


if st.button("Send") and user_input:

    st.session_state.messages.append(
        {"role": "You", "content": user_input}
    )

    with st.spinner("🤖 Devy is thinking..."):

        if mode == "💬 Chat Mode":
            response = llm.invoke(SYSTEM_PROMPT + user_input)

        else:  # Interview Mode
            response = llm.invoke(
                INTERVIEW_PROMPT +
                "Question Topic: " + user_input
            )

    st.session_state.messages.append(
        {"role": "Bot", "content": response}
    )

    st.session_state.input = ""
    st.rerun()
