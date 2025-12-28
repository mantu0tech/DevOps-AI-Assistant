import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

import re
from concurrent.futures import ThreadPoolExecutor
import os
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

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
# HELPER FUNCTIONSFload_llm
# ----------------------------
def is_devops_related(query):
    """Check if query is DevOps related"""
    devops_keywords = [
        'devops', 'docker', 'kubernetes', 'k8s', 'jenkins', 'ci/cd', 'cicd',
        'git', 'github', 'gitlab', 'ansible', 'terraform', 'aws', 'azure', 'gcp',
        'cloud', 'linux', 'bash', 'shell', 'container', 'pod', 'deployment',
        'pipeline', 'automation', 'infrastructure', 'monitoring', 'prometheus',
        'grafana', 'elk', 'nginx', 'apache', 'server', 'network', 'security',
        'helm', 'argocd', 'maven', 'gradle', 'nexus', 'artifactory',
        'microservices', 'orchestration', 'scaling', 'load balancer', 'vpc',
        'ec2', 'ecs', 'eks', 'lambda', 's3', 'iam', 'cloudformation',
        'vagrant', 'packer', 'consul', 'vault', 'istio', 'service mesh',
        'ingress', 'egress', 'firewall', 'dns', 'ssl', 'tls', 'https',
        'yaml', 'json', 'api', 'rest', 'webhook', 'cron', 'systemd',
        'daemon', 'process', 'thread', 'cpu', 'memory', 'disk', 'storage',
        'backup', 'disaster recovery', 'high availability', 'redundancy', 'python','aws','azure','networking'
    ]
    
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in devops_keywords)

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
    return ChatGroq(
        model_name="llama-3.1-8b-instant",
        groq_api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.7
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
                embedding=HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2"
                )
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
        
        - Ask DevOps questions only
        - Take technical interviews
        - Powered by Llama3 via Groq
        - Fast & optimized
        """)

# ----------------------------
# CHAT MODE (OPTIMIZED WITH DEVOPS FILTER)
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
            # Check if query is DevOps related
            if not is_devops_related(user_input):
                warning_msg = "⚠️ I'm a DevOps specialist assistant. I can only answer questions related to DevOps, Cloud Computing, Docker, Kubernetes, CI/CD, Linux, Git, Infrastructure, and related technologies. Please ask a DevOps-related question."
                st.warning(warning_msg)
                st.session_state.chat_history.append({"role": "assistant", "content": warning_msg})
            else:
                with st.spinner("💭 Thinking..."):
                    try:
                        if db:
                            # Get relevant context from vector DB
                            docs = db.similarity_search(user_input, k=2)
                            context = "\n".join([doc.page_content[:200] for doc in docs]) if docs else ""
                            
                            prompt = f"""You are a DevOps expert assistant. Only answer questions related to DevOps, Cloud, Docker, Kubernetes, CI/CD, Linux, Git, and related technologies.

Context: {context}

Question: {user_input}

Provide a clear and concise answer focused ONLY on DevOps topics:"""
                            response = llm.invoke(prompt)
                        else:
                            prompt = f"""You are a DevOps expert assistant. Only answer questions related to DevOps, Cloud, Docker, Kubernetes, CI/CD, Linux, Git, and related technologies.

Question: {user_input}

Provide a clear and concise answer focused ONLY on DevOps topics:"""
                            response = llm.invoke(prompt)
                        
                        # Extract content from response
                        response_text = response.content if hasattr(response, 'content') else str(response)
                        
                        st.write(response_text)
                        st.session_state.chat_history.append({"role": "assistant", "content": response_text})
                    
                    except Exception as e:
                        error_msg = f"Error: {str(e)}"
                        st.error(error_msg)
                        st.session_state.chat_history.append({"role": "assistant", "content": error_msg})
    
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

# ----------------------------f
# INTERVIEW MODE (STRICT EVALUATION)
# ----------------------------
elif st.session_state.mode == "interview":
    st.title("🎯 DevOps Interview Simulator")
    
    if not st.session_state.interview_started:
        st.markdown("### Configure Your Interview")
        
        col1, col2 = st.columns(2)
        
        with col1:
            topic = st.selectbox(
                "📚 Topic",
                ["DevOps", "Linux", "AWS", "Docker", "Kubernetes", "CI/CD", "Git"]
            )
            
            num_questions = st.slider("📝 Number of Questions", 3, 10, 5)
        
        with col2:
            difficulty = st.selectbox(
                "⚡ Difficulty",
                ["Beginner", "Intermediate", "Advanced"]
            )
        
        if st.button("🚀 Start Interview", use_container_width=True):
            with st.spinner("⚡ Generating questions..."):
                # Customize prompt based on difficulty
                if difficulty == "Beginner":
                    difficulty_instruction = """BEGINNER LEVEL - Questions should be:
- Basic concepts and definitions
- Simple "What is..." or "Define..." questions
- Fundamental terminology
- Easy to answer for someone just starting
- No complex scenarios or troubleshooting

Examples:
1. What is {topic}?
2. Name three basic commands in {topic}.
3. What is the purpose of {topic}?"""
                
                elif difficulty == "Intermediate":
                    difficulty_instruction = """INTERMEDIATE LEVEL - Questions should be:
- How things work together
- Common use cases and practical scenarios
- Differences between concepts
- Basic troubleshooting
- Real-world applications

Examples:
1. How does {topic} handle [specific feature]?
2. What are the differences between X and Y in {topic}?
3. Explain a common use case for {topic}."""
                
                else:  # Advanced
                    difficulty_instruction = """ADVANCED LEVEL - Questions should be:
- Deep technical details
- Architecture and internal workings
- Complex troubleshooting scenarios
- Best practices and optimization
- Security implications
- Performance tuning

Examples:
1. Explain the internal architecture of {topic}.
2. How would you troubleshoot [complex scenario] in {topic}?
3. What are the security implications of {topic}?"""
                
                prompt = f"""Generate exactly {num_questions} {difficulty} level interview questions about {topic}.

{difficulty_instruction}

IMPORTANT:
- Make questions appropriate for {difficulty} level
- Format: numbered list (1. 2. 3...)
- Each question on a new line
- Keep questions clear and focused

Generate {num_questions} questions now:"""
                
                try:
                    response = llm.invoke(prompt)
                    questions_text = response.content if hasattr(response, 'content') else str(response)
                    
                    # Extract questions using regex
                    questions = re.findall(r'\d+[\.\)]\s*(.+?)(?=\d+[\.\)]|$)', questions_text, re.DOTALL)
                    questions = [q.strip() for q in questions if len(q.strip()) > 10][:num_questions]
                    
                    if len(questions) >= 3:
                        st.session_state.questions = questions
                        st.session_state.interview_started = True
                        st.session_state.current_question = 0
                        st.session_state.score = 0
                        st.session_state.answers_log = []
                        st.rerun()
                    else:
                        st.error("Failed to generate enough questions. Please try again.")
                
                except Exception as e:
                    st.error(f"Error generating questions: {str(e)}")
    
    else:
        if st.session_state.current_question < len(st.session_state.questions):
            q_num = st.session_state.current_question
            
            progress = (q_num) / len(st.session_state.questions)
            st.progress(progress)
            
            st.markdown(f"### Question {q_num + 1} of {len(st.session_state.questions)}")
            
            question = st.session_state.questions[q_num]
            st.info(f"**{question}**")
            
            user_answer = st.text_area("Your Answer:", key=f"answer_{q_num}", height=150)
            
            if st.button("Submit Answer", use_container_width=True):
                if user_answer.strip():
                    with st.spinner("⚡ Evaluating your answer..."):
                        eval_prompt = f"""You are a fair but thorough technical interviewer evaluating a candidate's answer.

Question: {question}
Candidate's Answer: {user_answer}

EVALUATION CRITERIA:
Mark as CORRECT if the answer:
✓ Demonstrates understanding of core concepts
✓ Provides technical explanation (not just 1-2 words)
✓ Is factually accurate
✓ Addresses the main points of the question
✓ Shows practical knowledge

Mark as INCORRECT if the answer:
✗ Is just 1-2 words without explanation
✗ Is vague (like "idk", "not sure", "don't know")
✗ Is factually wrong or completely off-topic
✗ Shows no understanding of the concept

SCORING GUIDE:
- 0-3: Wrong, vague, or no real answer (INCORRECT)
- 4-6: Partially correct but missing key concepts (INCORRECT)
- 7-8: Good answer with proper explanation (CORRECT)
- 9-10: Excellent comprehensive answer (CORRECT)

Evaluation format:
Line 1: ONLY write "CORRECT" or "INCORRECT"
Line 2: Score: X/10
Line 3+: Brief explanation of the score
Line 4+: Key points covered/missing

Evaluate now:"""
                        
                        try:
                            # First check for obviously wrong answers
                            answer_lower = user_answer.lower().strip()
                            word_count = len(user_answer.split())
                            
                            # Auto-fail conditions - only for extremely bad answers
                            bad_answers = ['idk', "i don't know", "dont know", "no idea", "not sure", 
                                         "dunno", "dk", "?", "...", "na", "n/a"]
                            
                            # Only auto-fail if answer is in bad list OR is less than 5 words
                            if answer_lower in bad_answers or word_count < 5:
                                is_correct = False
                                evaluation = f"""INCORRECT

Score: 0/10

Your answer "{user_answer}" is not acceptable because:
- Too short or vague to demonstrate understanding
- Lacks technical explanation or details
- Does not properly address the question

A good answer should:
- Explain the core concepts clearly
- Provide technical details
- Show practical understanding
- Address all parts of the question"""
                            else:
                                # Send to LLM for evaluation
                                response = llm.invoke(eval_prompt)
                                evaluation = response.content if hasattr(response, 'content') else str(response)
                                
                                # Extract first line to check correctness
                                first_line = evaluation.strip().split('\n')[0].upper()
                                is_correct = "CORRECT" in first_line and "INCORRECT" not in first_line
                                
                                # Verify with score - but be more lenient
                                score_match = re.search(r'(\d+)/10', evaluation)
                                if score_match:
                                    score = int(score_match.group(1))
                                    # Override: score >= 7 means correct, < 7 means incorrect
                                    if score >= 5:
                                        is_correct = True
                                    elif score < 5:
                                        is_correct = False
                            
                            if is_correct:
                                st.session_state.score += 1
                                st.success("✅ Correct!")
                            else:
                                st.error("❌ Incorrect")
                            
                            st.markdown("**Detailed Feedback:**")
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
                            st.error(f"Error evaluating answer: {str(e)}")
                else:
                    st.warning("⚠️ Please provide an answer!")
        
        else:
            st.balloons()
            st.markdown("## 🏁 Interview Complete!")
            
            score = st.session_state.score
            total = len(st.session_state.questions)
            percentage = (score / total) * 100
            pass_score = int(total * 0.5)  # 70% passing threshold
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Score", f"{score}/{total}")
            
            with col2:
                st.metric("Percentage", f"{percentage:.0f}%")
            
            with col3:
                status = "PASSED ✅" if score >= pass_score else "FAILED ❌"
                st.metric("Status", status)
            
            # Performance message
            if percentage >= 80:
                st.success("🌟 Excellent performance! You have strong DevOps knowledge.")
            elif percentage >= 70:
                st.success("👍 Good job! You passed the interview.")
            elif percentage >= 50:
                st.warning("⚠️ You need more practice. Review the feedback below.")
            else:
                st.error("📚 Please study more and try again. Review the concepts thoroughly.")
            
            st.markdown("### 📝 Answer Review")
            
            for idx, log in enumerate(st.session_state.answers_log, 1):
                with st.expander(f"{'✅' if log['correct'] else '❌'} Question {idx}"):
                    st.markdown(f"**Q:** {log['question']}")
                    st.markdown(f"**Your Answer:** {log['answer']}")
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
    
    st.info("💡 **Note:** This assistant specializes in DevOps, Cloud, Docker, Kubernetes, CI/CD, Linux, Git, and related technologies only.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 💬 Chat Mode
        
        - Ask DevOps questions
        - Get instant AI answers
        - Context-aware responses
        - Fast & optimized
        - **DevOps topics only**
        """)
        
        if st.button("Start Chatting 💬", use_container_width=True):
            st.session_state.mode = "chat"
            st.rerun()
    
    with col2:
        st.markdown("""
        ### 🎯 Interview Mode
        
        - Test your knowledge
        - Multiple topics
        - **Strict evaluation**
        - Detailed feedback
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
    <p style='margin: 10px 0; font-size: 0.9em;'>Powered by Llama3 (Groq API) & Streamlit</p>
</div>
""", unsafe_allow_html=True)