import streamlit as st
from langchain_ollama import OllamaLLM
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
import os
import re
from concurrent.futures import ThreadPoolExecutor

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(
    page_title="DevOps AI Assistant",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------
# INITIALIZE SESSION STATE
# ----------------------------
if 'theme' not in st.session_state:
    st.session_state.theme = "Dark"

if 'mode' not in st.session_state:
    st.session_state.mode = None

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'interview_started' not in st.session_state:
    st.session_state.interview_started = False

if 'current_question' not in st.session_state:
    st.session_state.current_question = 0

if 'score' not in st.session_state:
    st.session_state.score = 0

if 'questions' not in st.session_state:
    st.session_state.questions = []

if 'answers_log' not in st.session_state:
    st.session_state.answers_log = []

# ----------------------------
# THEME STYLES
# ----------------------------
def load_theme():
    if st.session_state.theme == "Dark":
        st.markdown("""
        <style>
        /* Dark Theme */
        .stApp {
            background-color: #0f0f23;
            color: #ffffff;
        }
        
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
        }
        
        .stChatMessage {
            background-color: #1e1e3f !important;
            color: #ffffff !important;
            border-radius: 10px;
            padding: 15px;
            margin: 10px 0;
        }
        
        .stTextInput>div>div>input, .stTextArea>div>div>textarea {
            background-color: #1e1e3f !important;
            color: #ffffff !important;
            border: 2px solid #4a5568 !important;
            border-radius: 8px;
        }
        
        .stButton>button {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            color: white !important;
            border: none;
            border-radius: 8px;
            padding: 12px 24px;
            font-weight: 600;
            transition: all 0.3s;
        }
        
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        }
        
        h1, h2, h3, h4, h5, h6 {
            color: #ffffff !important;
        }
        
        .stMarkdown {
            color: #ffffff !important;
        }
        
        [data-testid="stSidebar"] * {
            color: #ffffff !important;
        }
        
        .stAlert {
            background-color: #1e1e3f !important;
            color: #ffffff !important;
        }
        
        .stInfo {
            background-color: #2d3748 !important;
            color: #ffffff !important;
            border-left: 4px solid #667eea !important;
        }
        
        .stSuccess {
            background-color: #2d4a2c !important;
            color: #ffffff !important;
        }
        
        .stError {
            background-color: #4a2c2c !important;
            color: #ffffff !important;
        }
        
        .stExpander {
            background-color: #1e1e3f !important;
            border: 1px solid #4a5568 !important;
        }
        
        div[data-baseweb="select"]>div {
            background-color: #1e1e3f !important;
            color: #ffffff !important;
        }
        
        .stSlider>div>div>div>div {
            color: #ffffff !important;
        }
        
        footer {
            background-color: #1a1a2e;
            color: #ffffff;
            padding: 20px;
            text-align: center;
            border-top: 2px solid #667eea;
        }
        </style>
        """, unsafe_allow_html=True)
    
    else:  # Light (Purple) Theme
        st.markdown("""
        <style>
        /* Light Purple Theme */
        .stApp {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }
        
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #4f46e5 0%, #7c3aed 100%);
        }
        
        /* Chat messages - WHITE background with BLACK text */
        .stChatMessage {
            background-color: #ffffff !important;
            color: #000000 !important;
            border-radius: 10px;
            padding: 15px;
            margin: 10px 0;
            box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        }
        
        .stChatMessage p, .stChatMessage div, .stChatMessage span {
            color: #000000 !important;
        }
        
        /* Text inputs - WHITE with BLACK text */
        .stTextInput>div>div>input, .stTextArea>div>div>textarea {
            background-color: #ffffff !important;
            color: #000000 !important;
            border: 2px solid #7c3aed !important;
            border-radius: 8px;
        }
        
        .stTextInput label, .stTextArea label {
            color: #ffffff !important;
            font-weight: 600;
        }
        
        .stButton>button {
            background: linear-gradient(90deg, #6366f1 0%, #a855f7 100%);
            color: white !important;
            border: none;
            border-radius: 8px;
            padding: 12px 24px;
            font-weight: 600;
            transition: all 0.3s;
        }
        
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 15px rgba(99, 102, 241, 0.5);
        }
        
        /* Headers - WHITE with shadow */
        h1, h2, h3, h4, h5, h6 {
            color: #ffffff !important;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.4);
            font-weight: 700;
        }
        
        /* Regular text - WHITE */
        .stMarkdown, p, div {
            color: #ffffff !important;
        }
        
        [data-testid="stSidebar"] * {
            color: #ffffff !important;
        }
        
        /* Info box - WHITE with BLACK text */
        .stInfo {
            background-color: #ffffff !important;
            border-left: 4px solid #7c3aed !important;
        }
        
        .stInfo p, .stInfo div, .stInfo strong {
            color: #000000 !important;
        }
        
        .stSuccess {
            background-color: rgba(72, 187, 120, 1) !important;
            color: #ffffff !important;
        }
        
        .stSuccess p, .stSuccess div {
            color: #ffffff !important;
        }
        
        .stError {
            background-color: rgba(245, 101, 101, 1) !important;
            color: #ffffff !important;
        }
        
        .stError p, .stError div {
            color: #ffffff !important;
        }
        
        .stWarning {
            background-color: rgba(237, 137, 54, 1) !important;
            color: #ffffff !important;
        }
        
        .stWarning p, .stWarning div {
            color: #ffffff !important;
        }
        
        /* Expander - WHITE with BLACK text */
        .stExpander {
            background-color: #ffffff !important;
        }
        
        .stExpander p, .stExpander div, .stExpander strong {
            color: #000000 !important;
        }
        
        /* Select boxes - WHITE with BLACK text */
        div[data-baseweb="select"]>div {
            background-color: #ffffff !important;
            color: #000000 !important;
        }
        
        div[data-baseweb="select"] span {
            color: #000000 !important;
        }
        
        /* Slider */
        .stSlider label {
            color: #ffffff !important;
            font-weight: 600;
        }
        
        footer {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            color: #ffffff;
            padding: 20px;
            text-align: center;
            border-top: 2px solid rgba(255,255,255,0.3);
        }
        </style>
        """, unsafe_allow_html=True)

load_theme()

# ----------------------------
# OPTIMIZED LLM SETUP
# ----------------------------
@st.cache_resource
def load_llm():
    # Ultra-fast configuration
    return OllamaLLM(
        model="llama3",
        temperature=0.5,  # Lower = faster
        num_predict=128,  # Max 128 tokens = super fast
        top_k=10,         # Reduced for speed
        top_p=0.8,        # Focused responses
        repeat_penalty=1.1
    )

@st.cache_resource
def load_documents():
    docs = []
    if os.path.exists("devops_data"):
        for file in os.listdir("devops_data"):
            if file.endswith('.txt'):
                try:
                    loader = TextLoader(f"devops_data/{file}")
                    docs.extend(loader.load())
                except:
                    continue
        
        if docs:
            text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=20)
            documents = text_splitter.split_documents(docs)
            
            db = Chroma.from_documents(
                documents,
                embedding=OllamaEmbeddings(model="llama3")
            )
            return db, len(documents)
    return None, 0

llm = load_llm()
db, doc_count = load_documents()

# ----------------------------
# SIDEBAR
# ----------------------------
with st.sidebar:
    st.title("🚀 DevOps AI Assistant")
    st.markdown("---")
    
    # Theme toggle
    theme_option = st.radio(
        "🎨 Theme",
        ["Dark", "Light (Purple)"],
        key="theme_radio"
    )
    
    if theme_option != st.session_state.theme:
        st.session_state.theme = theme_option
        st.rerun()
    
    st.markdown("---")
    
    # Mode selection
    st.subheader("Select Mode")
    
    if st.button("💬 Chat Mode", use_container_width=True):
        st.session_state.mode = "chat"
        st.rerun()
    
    if st.button("🎯 Interview Mode", use_container_width=True):
        st.session_state.mode = "interview"
        st.session_state.interview_started = False
        st.rerun()
    
    st.markdown("---")
    
    # Stats
    st.subheader("📊 Stats")
    if db:
        st.metric("Documents", doc_count)
    st.metric("Model", "Llama3")
    
    st.markdown("---")
    
    with st.expander("ℹ️ About"):
        st.write("""
        **DevOps AI Assistant**
        
        - Ask DevOps questions
        - Take technical interviews
        - Powered by Llama3
        - Fast & optimized
        """)

# ----------------------------
# CHAT MODE (OPTIMIZED)
# ----------------------------
if st.session_state.mode == "chat":
    st.title("💬 DevOps Chat Assistant")
    st.markdown("Ask me anything about DevOps, Cloud, Docker, Kubernetes, CI/CD!")
    
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Chat input
    user_input = st.chat_input("Ask your DevOps question...")
    
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        with st.chat_message("user"):
            st.write(user_input)
        
        with st.chat_message("assistant"):
            with st.spinner("💭"):
                # Ultra-short optimized prompt
                PROMPT = f"Answer briefly: {user_input}"
                
                try:
                    if db:
                        docs = db.similarity_search(user_input, k=1)  # Only 1 doc
                        context = docs[0].page_content[:150] if docs else ""
                        response = llm.invoke(f"{context}\n{PROMPT}")
                    else:
                        response = llm.invoke(PROMPT)
                    
                    st.write(response)
                    st.session_state.chat_history.append({"role": "assistant", "content": response})
                
                except Exception as e:
                    st.error(f"Error: {e}")
    
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

# ----------------------------
# INTERVIEW MODE (OPTIMIZED)
# ----------------------------
elif st.session_state.mode == "interview":
    st.title("🎯 DevOps Interview Simulator")
    
    if not st.session_state.interview_started:
        st.markdown("### Configure Your Interview")
        
        col1, col2 = st.columns(2)
        
        with col1:
            topic = st.selectbox(
                "📚 Topic",
                ["devops", "linux", "aws", "docker", "kubernetes"]
            )
            
            num_questions = st.slider("📝 Questions", 3, 5, 3)
        
        with col2:
            difficulty = st.selectbox(
                "⚡ Difficulty",
                ["beginner", "intermediate", "advanced"]
            )
        
        if st.button("🚀 Start Interview", use_container_width=True):
            with st.spinner("⚡ Generating..."):
                # Super short prompt for speed
                prompt = f"List {num_questions} short {difficulty} {topic} questions. Format: 1. [q] 2. [q]"
                
                try:
                    questions_text = llm.invoke(prompt)
                    questions = re.findall(r'\d+\.\s+(.+?)(?=\d+\.|$)', questions_text, re.DOTALL)
                    questions = [q.strip()[:150] for q in questions if len(q.strip()) > 10][:num_questions]
                    
                    st.session_state.questions = questions
                    st.session_state.interview_started = True
                    st.session_state.current_question = 0
                    st.session_state.score = 0
                    st.session_state.answers_log = []
                    st.rerun()
                
                except Exception as e:
                    st.error(f"Error: {e}")
    
    else:
        if st.session_state.current_question < len(st.session_state.questions):
            q_num = st.session_state.current_question
            
            progress = (q_num) / len(st.session_state.questions)
            st.progress(progress)
            
            st.markdown(f"### Question {q_num + 1} of {len(st.session_state.questions)}")
            
            question = st.session_state.questions[q_num]
            st.info(f"**{question}**")
            
            user_answer = st.text_area("Your Answer:", key=f"answer_{q_num}", height=120)
            
            if st.button("Submit Answer", use_container_width=True):
                if user_answer.strip():
                    with st.spinner("⚡"):
                        # Ultra-short evaluation
                        eval_prompt = f"Is this correct? Say CORRECT or INCORRECT then explain briefly.\nQ: {question}\nA: {user_answer}"
                        
                        try:
                            evaluation = llm.invoke(eval_prompt)
                            
                            is_correct = "CORRECT" in evaluation.upper()
                            
                            if is_correct:
                                st.session_state.score += 1
                                st.success("✅ Correct!")
                            else:
                                st.error("❌ Incorrect")
                            
                            st.markdown("**Feedback:**")
                            st.write(evaluation)
                            
                            st.session_state.answers_log.append({
                                'question': question,
                                'answer': user_answer,
                                'correct': is_correct,
                                'feedback': evaluation
                            })
                            
                            st.session_state.current_question += 1
                            
                            if st.session_state.current_question < len(st.session_state.questions):
                                if st.button("Next Question ➡️"):
                                    st.rerun()
                            else:
                                st.rerun()
                        
                        except Exception as e:
                            st.error(f"Error: {e}")
                else:
                    st.warning("Please provide an answer!")
        
        else:
            st.balloons()
            st.markdown("## 🏁 Interview Complete!")
            
            score = st.session_state.score
            total = len(st.session_state.questions)
            percentage = (score / total) * 100
            pass_score = int(total * 0.6)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Score", f"{score}/{total}")
            
            with col2:
                st.metric("Percentage", f"{percentage:.0f}%")
            
            with col3:
                status = "PASSED ✅" if score >= pass_score else "FAILED ❌"
                st.metric("Status", status)
            
            st.markdown("### 📝 Review")
            
            for idx, log in enumerate(st.session_state.answers_log, 1):
                with st.expander(f"{'✅' if log['correct'] else '❌'} Question {idx}"):
                    st.markdown(f"**Q:** {log['question']}")
                    st.markdown(f"**A:** {log['answer']}")
                    st.markdown(f"**Feedback:** {log['feedback']}")
            
            if st.button("🔄 New Interview", use_container_width=True):
                st.session_state.interview_started = False
                st.session_state.current_question = 0
                st.session_state.score = 0
                st.session_state.questions = []
                st.session_state.answers_log = []
                st.rerun()

# ----------------------------
# HOME PAGE
# ----------------------------
else:
    st.title("🚀 Welcome to DevOps AI Assistant")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 💬 Chat Mode
        
        - Ask DevOps questions
        - Get instant AI answers
        - Context-aware responses
        - Fast & optimized
        """)
        
        if st.button("Start Chatting 💬", use_container_width=True):
            st.session_state.mode = "chat"
            st.rerun()
    
    with col2:
        st.markdown("""
        ### 🎯 Interview Mode
        
        - Test your knowledge
        - Multiple topics
        - Instant feedback
        - Track your progress
        """)
        
        if st.button("Take Interview 🎯", use_container_width=True):
            st.session_state.mode = "interview"
            st.rerun()

# ----------------------------
# FOOTER
# ----------------------------
st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 20px; color: white;'>
    <p style='margin: 10px 0;'><strong>Created by Ansari Mantasha</strong></p>
    <p style='margin: 10px 0;'>
        <a href='https://github.com/mantu0tech' target='_blank' style='color: #a855f7; text-decoration: none; margin: 0 15px;'>
            🔗 GitHub
        </a>
        |
        <a href='https://www.linkedin.com/in/mantasha-ansari-47162b24a/' target='_blank' style='color: #a855f7; text-decoration: none; margin: 0 15px;'>
            🔗 LinkedIn
        </a>
    </p>
    <p style='margin: 10px 0; font-size: 0.9em;'>Powered by Llama3 & Streamlit</p>
</div>
""", unsafe_allow_html=True)