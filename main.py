import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

import re
import os
import bcrypt
import mysql.connector
from mysql.connector import Error
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(
    page_title="DevOps AI Assistant",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# DATABASE CONFIG & FUNCTIONS
# ============================================================

DB_CONFIG = {
    "host":     os.getenv("DB_HOST", "localhost"),
    "port":     int(os.getenv("DB_PORT", 3306)),
    "user":     os.getenv("DB_USER", "root"),
    "password": os.getenv("DB_PASSWORD", ""),
    "database": os.getenv("DB_NAME", "devops_assistant"),
}

def get_connection():
    return mysql.connector.connect(**DB_CONFIG)

def initialize_database():
    try:
        cfg_no_db = {k: v for k, v in DB_CONFIG.items() if k != "database"}
        conn   = mysql.connector.connect(**cfg_no_db)
        cursor = conn.cursor()
        cursor.execute(
            f"CREATE DATABASE IF NOT EXISTS `{DB_CONFIG['database']}` "
            f"CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;"
        )
        cursor.execute(f"USE `{DB_CONFIG['database']}`;")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id            INT AUTO_INCREMENT PRIMARY KEY,
                full_name     VARCHAR(150)  NOT NULL,
                username      VARCHAR(80)   NOT NULL UNIQUE,
                email         VARCHAR(150)  NOT NULL UNIQUE,
                password_hash VARCHAR(255)  NOT NULL,
                created_at    DATETIME      DEFAULT CURRENT_TIMESTAMP,
                last_login    DATETIME      NULL,
                is_active     TINYINT(1)    DEFAULT 1
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """)
        conn.commit()
        cursor.close()
        conn.close()
        return True, "Database initialised successfully."
    except Error as e:
        return False, f"DB init error: {e}"

def register_user(full_name, username, email, password):
    try:
        pw_hash = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt(rounds=12)).decode("utf-8")
        conn   = get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO users (full_name, username, email, password_hash) VALUES (%s, %s, %s, %s)",
            (full_name.strip(), username.strip().lower(), email.strip().lower(), pw_hash),
        )
        conn.commit()
        cursor.close()
        conn.close()
        return True, "Registration successful! You can now log in."
    except mysql.connector.IntegrityError as e:
        msg = str(e)
        if "username" in msg:
            return False, "Username already taken. Please choose another."
        if "email" in msg:
            return False, "An account with this email already exists."
        return False, f"Registration failed: {e}"
    except Error as e:
        return False, f"Database error: {e}"

def login_user(username_or_email, password):
    try:
        conn   = get_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute(
            """
            SELECT id, full_name, username, email, password_hash, is_active
            FROM   users
            WHERE  (username = %s OR email = %s)
            LIMIT  1
            """,
            (username_or_email.strip().lower(), username_or_email.strip().lower()),
        )
        user = cursor.fetchone()
        if not user:
            cursor.close(); conn.close()
            return False, "No account found with that username/email.", None
        if not user["is_active"]:
            cursor.close(); conn.close()
            return False, "Your account has been deactivated. Contact support.", None
        if not bcrypt.checkpw(password.encode("utf-8"), user["password_hash"].encode("utf-8")):
            cursor.close(); conn.close()
            return False, "Incorrect password. Please try again.", None
        cursor.execute("UPDATE users SET last_login = NOW() WHERE id = %s", (user["id"],))
        conn.commit()
        cursor.close()
        conn.close()
        user.pop("password_hash", None)
        return True, f"Welcome back, {user['full_name']}!", user
    except Error as e:
        return False, f"Database error: {e}", None

# ============================================================
# DB BOOTSTRAP
# ============================================================
if "db_initialised" not in st.session_state:
    ok, msg = initialize_database()
    if not ok:
        st.error(f"❌ Could not connect to MySQL: {msg}")
        st.info("Check your .env for DB_HOST, DB_USER, DB_PASSWORD, DB_NAME and ensure MySQL is running.")
        st.stop()
    st.session_state.db_initialised = True

# ----------------------------
# INITIALIZE SESSION STATE
# ----------------------------
defaults = {
    "theme":             "Dark",
    "mode":              None,
    "chat_history":      [],
    "interview_started": False,
    "current_question":  0,
    "score":             0,
    "questions":         [],
    "answers_log":       [],
    "logged_in":         False,
    "current_user":      None,
    "auth_page":         "login",
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ----------------------------
# HELPER FUNCTIONS
# ----------------------------
def is_devops_related(query):
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
        'backup', 'disaster recovery', 'high availability', 'redundancy', 'python', 'aws', 'azure', 'networking'
    ]
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in devops_keywords)

def _valid_email(email):
    return bool(re.match(r"^[\w\.\+\-]+@[\w\-]+\.[a-zA-Z]{2,}$", email))

def _valid_password(pw):
    if len(pw) < 8:
        return False, "Password must be at least 8 characters."
    if not re.search(r"[A-Z]", pw):
        return False, "Password must contain at least one uppercase letter."
    if not re.search(r"\d", pw):
        return False, "Password must contain at least one digit."
    return True, ""

def _password_strength_html(pw):
    score = 0
    if len(pw) >= 8:                          score += 1
    if re.search(r"[A-Z]", pw):              score += 1
    if re.search(r"\d", pw):                 score += 1
    if re.search(r"[^A-Za-z0-9]", pw):      score += 1
    colours = ["#ff4b6e", "#ff8c42", "#f5c518", "#00f5a0"]
    labels  = ["Weak", "Fair", "Good", "Strong"]
    pct     = (score / 4) * 100
    colour  = colours[max(score - 1, 0)]
    label   = labels[max(score - 1, 0)]
    return (
        f'<div style="margin:6px 0 14px;">'
        f'<div style="width:100%;height:4px;background:#1a1a2e;border-radius:2px;">'
        f'<div style="width:{pct}%;height:4px;background:{colour};border-radius:2px;'
        f'box-shadow:0 0 8px {colour};transition:width .4s ease;"></div></div>'
        f'<div style="font-size:.75rem;margin-top:5px;color:{colour};font-family:\'JetBrains Mono\',monospace;letter-spacing:.05em;">'
        f'▸ {label} password</div>'
        f'</div>'
    )

# ----------------------------
# GLOBAL STYLES
# ----------------------------
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@300;400;500;700&family=Syne:wght@400;600;700;800&display=swap" rel="stylesheet">

<style>
/* ── ROOT VARIABLES ── */
:root {
    --bg-base:      #030712;
    --bg-surface:   #080f1e;
    --bg-card:      #0d1829;
    --bg-elevated:  #111d33;
    --cyan:         #00e5ff;
    --cyan-dim:     #00b8cc;
    --green:        #00f5a0;
    --green-dim:    #00c27e;
    --amber:        #ffb700;
    --pink:         #ff4b8b;
    --text-primary: #e8f4fd;
    --text-secondary:#7a9ab5;
    --text-muted:   #3d5a73;
    --border:       rgba(0,229,255,0.15);
    --border-bright:rgba(0,229,255,0.5);
    --glow-cyan:    0 0 20px rgba(0,229,255,0.3), 0 0 40px rgba(0,229,255,0.1);
    --glow-green:   0 0 20px rgba(0,245,160,0.3), 0 0 40px rgba(0,245,160,0.1);
    --radius:       12px;
}

/* ── BASE ── */
.stApp {
    background-color: var(--bg-base) !important;
    background-image:
        radial-gradient(ellipse 80% 50% at 50% -10%, rgba(0,229,255,0.08) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 90% 80%, rgba(0,245,160,0.05) 0%, transparent 50%),
        repeating-linear-gradient(0deg, transparent, transparent 39px, rgba(0,229,255,0.03) 39px, rgba(0,229,255,0.03) 40px),
        repeating-linear-gradient(90deg, transparent, transparent 39px, rgba(0,229,255,0.02) 39px, rgba(0,229,255,0.02) 40px);
    font-family: 'Space Grotesk', sans-serif !important;
    color: var(--text-primary) !important;
}

/* ── SIDEBAR ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #060e1c 0%, #030a15 100%) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"]::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--cyan), transparent);
}
[data-testid="stSidebar"] * {
    color: var(--text-primary) !important;
    font-family: 'Space Grotesk', sans-serif !important;
}

/* ── TYPOGRAPHY ── */
h1 {
    font-family: 'Syne', sans-serif !important;
    font-weight: 800 !important;
    font-size: 2.2rem !important;
    color: var(--text-primary) !important;
    letter-spacing: -0.02em !important;
}
h2, h3 {
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    color: var(--text-primary) !important;
}
h4, h5, h6 { color: var(--text-primary) !important; }
p, div, span, label { color: var(--text-primary) !important; }

/* ── GLOWING TITLE ACCENT ── */
.neon-title {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 2.4rem;
    background: linear-gradient(135deg, var(--cyan) 0%, var(--green) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    filter: drop-shadow(0 0 20px rgba(0,229,255,0.4));
    letter-spacing: -0.02em;
    line-height: 1.1;
}
.neon-subtitle {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    color: var(--cyan-dim) !important;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    opacity: 0.8;
}

/* ── BUTTONS ── */
.stButton > button {
    background: transparent !important;
    border: 1px solid var(--cyan) !important;
    color: var(--cyan) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.08em !important;
    padding: 10px 20px !important;
    border-radius: 6px !important;
    transition: all 0.25s ease !important;
    text-transform: uppercase !important;
    position: relative !important;
    overflow: hidden !important;
}
.stButton > button::before {
    content: '';
    position: absolute;
    top: 0; left: -100%; right: 0; bottom: 0;
    background: linear-gradient(90deg, transparent, rgba(0,229,255,0.12), transparent);
    transition: left 0.4s ease;
}
.stButton > button:hover {
    background: rgba(0,229,255,0.08) !important;
    box-shadow: var(--glow-cyan) !important;
    transform: translateY(-1px) !important;
}
.stButton > button:hover::before { left: 100%; }

/* Primary action buttons (start, submit) */
.stButton > button[kind="primary"],
.stButton > button:contains("Start"),
.stButton > button:contains("Submit"),
.stButton > button:contains("Login"),
.stButton > button:contains("Register") {
    background: linear-gradient(135deg, rgba(0,229,255,0.15), rgba(0,245,160,0.1)) !important;
    border-color: var(--green) !important;
    color: var(--green) !important;
    box-shadow: 0 0 15px rgba(0,245,160,0.2) !important;
}

/* ── INPUTS ── */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text-primary) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.88rem !important;
    padding: 10px 14px !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
    border-color: var(--cyan) !important;
    box-shadow: 0 0 0 2px rgba(0,229,255,0.15), var(--glow-cyan) !important;
    outline: none !important;
}
.stTextInput > label,
.stTextArea > label {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: var(--cyan-dim) !important;
}

/* ── CHAT MESSAGES ── */
.stChatMessage {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    padding: 16px 20px !important;
    margin: 8px 0 !important;
}
.stChatMessage[data-testid*="user"] {
    border-left: 3px solid var(--cyan) !important;
    background: rgba(0,229,255,0.04) !important;
}
.stChatMessage[data-testid*="assistant"] {
    border-left: 3px solid var(--green) !important;
    background: rgba(0,245,160,0.03) !important;
}
.stChatMessage * { color: var(--text-primary) !important; }

/* ── CHAT INPUT ── */
.stChatInput > div {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
}
.stChatInput input {
    background: transparent !important;
    color: var(--text-primary) !important;
    font-family: 'JetBrains Mono', monospace !important;
}
.stChatInput input::placeholder { color: var(--text-muted) !important; }

/* ── ALERTS & CALLOUTS ── */
.stInfo {
    background: rgba(0,229,255,0.06) !important;
    border: 1px solid rgba(0,229,255,0.25) !important;
    border-left: 3px solid var(--cyan) !important;
    border-radius: 8px !important;
    color: var(--text-primary) !important;
}
.stSuccess {
    background: rgba(0,245,160,0.07) !important;
    border: 1px solid rgba(0,245,160,0.3) !important;
    border-left: 3px solid var(--green) !important;
    border-radius: 8px !important;
}
.stError {
    background: rgba(255,75,110,0.07) !important;
    border: 1px solid rgba(255,75,110,0.3) !important;
    border-left: 3px solid var(--pink) !important;
    border-radius: 8px !important;
}
.stWarning {
    background: rgba(255,183,0,0.06) !important;
    border: 1px solid rgba(255,183,0,0.25) !important;
    border-left: 3px solid var(--amber) !important;
    border-radius: 8px !important;
}
.stInfo *, .stSuccess *, .stError *, .stWarning * {
    color: var(--text-primary) !important;
}

/* ── SELECTBOX ── */
div[data-baseweb="select"] > div {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text-primary) !important;
    font-family: 'JetBrains Mono', monospace !important;
}
div[data-baseweb="select"] span { color: var(--text-primary) !important; }
div[data-baseweb="popover"] { background: var(--bg-elevated) !important; border: 1px solid var(--border) !important; }
li[role="option"] { color: var(--text-primary) !important; }
li[role="option"]:hover { background: rgba(0,229,255,0.08) !important; }

/* ── SLIDER ── */
.stSlider > div > div > div > div { color: var(--text-primary) !important; }
[data-testid="stSlider"] > div > div > div {
    background: var(--bg-elevated) !important;
}
[data-testid="stSlider"] > div > div > div > div {
    background: linear-gradient(90deg, var(--cyan), var(--green)) !important;
}

/* ── PROGRESS BAR ── */
.stProgress > div > div > div {
    background: linear-gradient(90deg, var(--cyan), var(--green)) !important;
    box-shadow: var(--glow-cyan) !important;
    border-radius: 4px !important;
}
.stProgress > div > div {
    background: var(--bg-elevated) !important;
    border-radius: 4px !important;
}

/* ── METRIC ── */
[data-testid="stMetric"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    padding: 16px !important;
}
[data-testid="stMetricLabel"] > div { color: var(--text-secondary) !important; font-size: 0.72rem !important; letter-spacing: 0.1em !important; text-transform: uppercase !important; font-family: 'JetBrains Mono', monospace !important; }
[data-testid="stMetricValue"] > div { color: var(--cyan) !important; font-family: 'Syne', sans-serif !important; font-size: 1.8rem !important; font-weight: 700 !important; }

/* ── EXPANDER ── */
.stExpander {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
}
.stExpander > div > div > div > div { color: var(--text-primary) !important; }
.stExpander summary { font-family: 'JetBrains Mono', monospace !important; color: var(--cyan) !important; font-size: 0.85rem !important; }

/* ── RADIO ── */
.stRadio > label { color: var(--text-secondary) !important; font-family: 'JetBrains Mono', monospace !important; font-size: 0.75rem !important; letter-spacing: 0.08em !important; text-transform: uppercase !important; }
.stRadio div[role="radiogroup"] label { color: var(--text-primary) !important; font-size: 0.88rem !important; }

/* ── DIVIDER ── */
hr { border-color: var(--border) !important; margin: 16px 0 !important; }

/* ── SCROLLBAR ── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: var(--bg-base); }
::-webkit-scrollbar-thumb { background: var(--border-bright); border-radius: 2px; }
::-webkit-scrollbar-thumb:hover { background: var(--cyan); }

/* ── SPINNER ── */
.stSpinner > div { border-top-color: var(--cyan) !important; }

/* ── FORMS ── */
[data-testid="stForm"] {
    background: transparent !important;
    border: none !important;
}

/* ── MARKDOWN ── */
.stMarkdown { color: var(--text-primary) !important; }
.stMarkdown code {
    background: var(--bg-elevated) !important;
    color: var(--cyan) !important;
    border: 1px solid var(--border) !important;
    border-radius: 4px !important;
    font-family: 'JetBrains Mono', monospace !important;
    padding: 2px 6px !important;
    font-size: 0.85em !important;
}
.stMarkdown pre {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-left: 3px solid var(--cyan) !important;
    border-radius: 8px !important;
    padding: 16px !important;
}

/* ── CUSTOM CARDS ── */
.glass-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 24px;
    position: relative;
    overflow: hidden;
    transition: border-color 0.3s, box-shadow 0.3s;
}
.glass-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--cyan), transparent);
    opacity: 0.6;
}
.glass-card:hover {
    border-color: rgba(0,229,255,0.35);
    box-shadow: var(--glow-cyan);
}

.tag-badge {
    display: inline-block;
    padding: 3px 10px;
    background: rgba(0,229,255,0.1);
    border: 1px solid rgba(0,229,255,0.3);
    border-radius: 20px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    color: var(--cyan) !important;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin: 2px;
}

.stat-pill {
    background: var(--bg-elevated);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 6px 14px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    color: var(--text-secondary) !important;
    display: inline-flex;
    align-items: center;
    gap: 6px;
}

/* ── AUTH CARD ── */
.auth-wrap {
    max-width: 460px;
    margin: 2rem auto;
}
.auth-header {
    text-align: center;
    margin-bottom: 2rem;
    padding-bottom: 1.5rem;
    border-bottom: 1px solid var(--border);
}
.auth-header .logo {
    font-family: 'Syne', sans-serif;
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(135deg, var(--cyan), var(--green));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    filter: drop-shadow(0 0 16px rgba(0,229,255,0.5));
    display: block;
    line-height: 1;
    margin-bottom: 8px;
}
.auth-header .tagline {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    color: var(--text-muted) !important;
    letter-spacing: 0.2em;
    text-transform: uppercase;
}
.auth-divider {
    text-align: center;
    color: var(--text-muted) !important;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.15em;
    margin: 1rem 0;
    position: relative;
}
.auth-divider::before, .auth-divider::after {
    content: '';
    position: absolute;
    top: 50%;
    width: 35%;
    height: 1px;
    background: var(--border);
}
.auth-divider::before { left: 0; }
.auth-divider::after { right: 0; }

/* ── USER PILL (SIDEBAR) ── */
.user-pill {
    background: var(--bg-elevated);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 12px 16px;
    margin-bottom: 4px;
    position: relative;
    overflow: hidden;
}
.user-pill::before {
    content: '';
    position: absolute;
    left: 0; top: 0; bottom: 0;
    width: 3px;
    background: linear-gradient(180deg, var(--cyan), var(--green));
}
.user-pill .name {
    font-weight: 600;
    font-size: 0.9rem;
    color: var(--text-primary) !important;
    font-family: 'Space Grotesk', sans-serif;
}
.user-pill .handle {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: var(--cyan-dim) !important;
    margin-top: 2px;
}

/* ── ONLINE INDICATOR ── */
.online-dot {
    display: inline-block;
    width: 7px;
    height: 7px;
    background: var(--green);
    border-radius: 50%;
    box-shadow: 0 0 8px var(--green);
    margin-right: 6px;
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.6; transform: scale(0.85); }
}

/* ── HOME FEATURE CARDS ── */
.feature-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 28px 24px;
    height: 100%;
    position: relative;
    overflow: hidden;
    transition: all 0.3s ease;
}
.feature-card::after {
    content: '';
    position: absolute;
    top: -50%; left: -50%;
    width: 200%; height: 200%;
    background: radial-gradient(circle at center, rgba(0,229,255,0.04) 0%, transparent 60%);
    pointer-events: none;
}
.feature-card.green-tint { border-color: rgba(0,245,160,0.2); }
.feature-card.green-tint::after { background: radial-gradient(circle at center, rgba(0,245,160,0.04) 0%, transparent 60%); }

.feature-icon {
    font-size: 2.2rem;
    margin-bottom: 12px;
    filter: drop-shadow(0 0 12px rgba(0,229,255,0.5));
    display: block;
}
.feature-title {
    font-family: 'Syne', sans-serif !important;
    font-size: 1.3rem !important;
    font-weight: 700 !important;
    color: var(--text-primary) !important;
    margin-bottom: 12px !important;
}
.feature-item {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 6px 0;
    font-size: 0.87rem;
    color: var(--text-secondary) !important;
    border-bottom: 1px solid rgba(0,229,255,0.05);
    font-family: 'Space Grotesk', sans-serif;
}
.feature-item::before {
    content: '›';
    color: var(--cyan);
    font-weight: 700;
    font-size: 1rem;
}

/* ── INTERVIEW QUESTION CARD ── */
.question-card {
    background: var(--bg-card);
    border: 1px solid rgba(0,229,255,0.2);
    border-radius: 12px;
    padding: 20px 24px;
    margin: 12px 0;
    position: relative;
}
.question-card::before {
    content: 'QUESTION';
    position: absolute;
    top: -10px; left: 20px;
    background: var(--bg-base);
    padding: 0 8px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    color: var(--cyan) !important;
    letter-spacing: 0.15em;
}
.q-number {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: var(--cyan-dim) !important;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 8px;
}
.q-text {
    font-size: 1.05rem !important;
    font-weight: 500 !important;
    color: var(--text-primary) !important;
    line-height: 1.5 !important;
}

/* ── RESULTS ── */
.result-banner {
    background: linear-gradient(135deg, rgba(0,229,255,0.08), rgba(0,245,160,0.06));
    border: 1px solid rgba(0,229,255,0.25);
    border-radius: 16px;
    padding: 32px;
    text-align: center;
    margin: 16px 0;
    position: relative;
    overflow: hidden;
}
.result-score {
    font-family: 'Syne', sans-serif;
    font-size: 4rem;
    font-weight: 800;
    background: linear-gradient(135deg, var(--cyan), var(--green));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    filter: drop-shadow(0 0 20px rgba(0,229,255,0.4));
    line-height: 1;
}
.result-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    color: var(--text-secondary) !important;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-top: 4px;
}

/* ── FOOTER ── */
.custom-footer {
    margin-top: 48px;
    padding: 24px 0;
    border-top: 1px solid var(--border);
    text-align: center;
}
.custom-footer .creator {
    font-family: 'Syne', sans-serif;
    font-size: 0.9rem;
    font-weight: 600;
    color: var(--text-secondary) !important;
    margin-bottom: 10px;
}
.custom-footer a {
    color: var(--cyan) !important;
    text-decoration: none;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    letter-spacing: 0.05em;
    margin: 0 14px;
    transition: color 0.2s;
}
.custom-footer a:hover { color: var(--green) !important; }
.custom-footer .stack {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    color: var(--text-muted) !important;
    letter-spacing: 0.1em;
    margin-top: 8px;
}

/* ── SIDEBAR SECTION HEADERS ── */
.sidebar-section {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    color: var(--text-muted) !important;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    margin: 8px 0 4px;
    padding-left: 2px;
}

/* ── MODE BUTTONS (sidebar) ── */
.mode-btn {
    width: 100%;
    background: var(--bg-elevated) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-secondary) !important;
    border-radius: 8px !important;
    padding: 10px 16px !important;
    text-align: left !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 0.88rem !important;
    cursor: pointer !important;
    transition: all 0.2s !important;
}
.mode-btn:hover {
    border-color: var(--cyan) !important;
    color: var(--cyan) !important;
    background: rgba(0,229,255,0.06) !important;
}

/* Spinner text */
.stSpinner p { color: var(--text-secondary) !important; font-family: 'JetBrains Mono', monospace !important; font-size: 0.8rem !important; }

/* Form submit button override */
button[type="submit"] {
    background: linear-gradient(135deg, rgba(0,229,255,0.15), rgba(0,245,160,0.1)) !important;
    border-color: var(--cyan) !important;
    color: var(--cyan) !important;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# AUTH PAGES
# ============================================================

def render_login():
    st.markdown("""
    <div class="auth-wrap">
        <div class="auth-header">
            <span class="logo">⚡ DEVOPS AI</span>
            <div class="tagline">Intelligent Infrastructure Assistant</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_l, col_c, col_r = st.columns([1, 2, 1])
    with col_c:
        st.markdown("""
        <div class="glass-card" style="margin-bottom:8px;">
            <div style="font-family:'JetBrains Mono',monospace;font-size:0.65rem;color:var(--cyan);letter-spacing:.18em;text-transform:uppercase;margin-bottom:20px;opacity:0.8;">
                ▸ Authentication Required
            </div>
        """, unsafe_allow_html=True)

        with st.form("login_form"):
            identifier = st.text_input("Identifier", placeholder="username or email")
            password   = st.text_input("Password",   type="password", placeholder="••••••••")
            submitted  = st.form_submit_button("⚡ AUTHENTICATE", use_container_width=True)

            if submitted:
                if not identifier or not password:
                    st.error("All fields are required.")
                else:
                    with st.spinner("Verifying credentials…"):
                        ok, msg, user = login_user(identifier, password)
                    if ok:
                        st.success(f"✓ {msg}")
                        st.session_state.logged_in    = True
                        st.session_state.current_user = user
                        st.session_state.auth_page    = None
                        st.rerun()
                    else:
                        st.error(msg)

        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="auth-divider">OR</div>', unsafe_allow_html=True)

        if st.button("→ CREATE NEW ACCOUNT", use_container_width=True, key="go_register"):
            st.session_state.auth_page = "register"
            st.rerun()


def render_register():
    st.markdown("""
    <div class="auth-wrap">
        <div class="auth-header">
            <span class="logo">⚡ DEVOPS AI</span>
            <div class="tagline">Create Your Account</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_l, col_c, col_r = st.columns([1, 2, 1])
    with col_c:
        st.markdown("""
        <div class="glass-card">
            <div style="font-family:'JetBrains Mono',monospace;font-size:0.65rem;color:var(--green);letter-spacing:.18em;text-transform:uppercase;margin-bottom:20px;opacity:0.8;">
                ▸ New User Registration
            </div>
        """, unsafe_allow_html=True)

        with st.form("register_form"):
            full_name = st.text_input("Full Name",        placeholder="Jane Smith")
            username  = st.text_input("Username",         placeholder="jane_smith")
            email     = st.text_input("Email Address",    placeholder="jane@example.com")
            password  = st.text_input("Password",         type="password", placeholder="Min 8 chars · 1 uppercase · 1 digit")
            confirm   = st.text_input("Confirm Password", type="password", placeholder="Re-enter password")

            if password:
                st.markdown(_password_strength_html(password), unsafe_allow_html=True)

            submitted = st.form_submit_button("✓ CREATE ACCOUNT", use_container_width=True)

            if submitted:
                errors = []
                if not full_name.strip():             errors.append("Full name is required.")
                if not username.strip() or " " in username: errors.append("Username must have no spaces.")
                if not _valid_email(email):           errors.append("Enter a valid email address.")
                pw_ok, pw_msg = _valid_password(password)
                if not pw_ok:                         errors.append(pw_msg)
                if password != confirm:               errors.append("Passwords do not match.")

                if errors:
                    for err in errors: st.error(err)
                else:
                    with st.spinner("Creating account…"):
                        ok, msg = register_user(full_name, username, email, password)
                    if ok:
                        st.success(f"✓ {msg}")
                        st.balloons()
                        st.session_state.auth_page = "login"
                        st.rerun()
                    else:
                        st.error(msg)

        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('<div class="auth-divider">OR</div>', unsafe_allow_html=True)

        if st.button("← BACK TO LOGIN", use_container_width=True, key="go_login"):
            st.session_state.auth_page = "login"
            st.rerun()


# ============================================================
# AUTH GATE
# ============================================================
if not st.session_state.logged_in:
    if st.session_state.auth_page == "register":
        render_register()
    else:
        render_login()
    st.stop()

# ----------------------------
# LLM SETUP
# ----------------------------
@st.cache_resource
def load_llm():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        st.error("❌ GROQ_API_KEY not found in .env file!")
        st.stop()
    return ChatGroq(
        model_name="llama-3.1-8b-instant",
        groq_api_key=api_key,
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
                embedding=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            )
            return db, len(documents)
    return None, 0

llm = load_llm()
db, doc_count = load_documents()

# ----------------------------
# SIDEBAR
# ----------------------------
with st.sidebar:
    st.markdown("""
    <div style="padding:16px 0 8px;">
        <div class="neon-title" style="font-size:1.5rem;">⚡ DevOps AI</div>
        <div class="neon-subtitle" style="margin-top:4px;">v2.0 · LLAMA3 ENGINE</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    user = st.session_state.current_user
    st.markdown(f"""
    <div class="user-pill">
        <div class="name"><span class="online-dot"></span>{user['full_name']}</div>
        <div class="handle">@{user['username']}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown('<div class="sidebar-section">Interface</div>', unsafe_allow_html=True)
    theme_option = st.radio("", ["Dark", "Light (Purple)"], key="theme_radio", label_visibility="collapsed")
    if theme_option != st.session_state.theme:
        st.session_state.theme = theme_option
        st.rerun()

    st.markdown("---")
    st.markdown('<div class="sidebar-section">Navigation</div>', unsafe_allow_html=True)

    if st.button("💬  Chat Mode", use_container_width=True):
        st.session_state.mode = "chat"
        st.rerun()
    if st.button("🎯  Interview Mode", use_container_width=True):
        st.session_state.mode = "interview"
        st.session_state.interview_started = False
        st.rerun()
    if st.button("🏠  Home", use_container_width=True):
        st.session_state.mode = None
        st.rerun()

    st.markdown("---")
    st.markdown('<div class="sidebar-section">System Status</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Docs", doc_count if db else 0)
    with col2:
        st.metric("Model", "L3.1")

    st.markdown("---")

    if st.button("⏻  LOGOUT", use_container_width=True):
        for key in ["logged_in", "current_user", "mode", "chat_history",
                    "interview_started", "current_question", "score", "questions", "answers_log"]:
            st.session_state.pop(key, None)
        st.session_state.auth_page = "login"
        st.rerun()

    st.markdown("---")
    with st.expander("ℹ️ About"):
        st.markdown("""
        <div style="font-family:'JetBrains Mono',monospace;font-size:0.75rem;color:var(--text-secondary);line-height:1.8;">
        DevOps AI Assistant<br>
        ▸ Chat with AI expert<br>
        ▸ Mock interviews<br>
        ▸ Groq · LLaMA 3.1<br>
        ▸ MySQL auth backend
        </div>
        """, unsafe_allow_html=True)

# ----------------------------
# CHAT MODE
# ----------------------------
if st.session_state.mode == "chat":
    st.markdown("""
    <div style="display:flex;align-items:center;gap:14px;margin-bottom:6px;">
        <div>
            <div class="neon-title">Chat Assistant</div>
            <div class="neon-subtitle" style="margin-top:4px;">DevOps knowledge base · RAG-powered</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:20px;">
        <span class="tag-badge">Docker</span>
        <span class="tag-badge">Kubernetes</span>
        <span class="tag-badge">CI/CD</span>
        <span class="tag-badge">Cloud</span>
        <span class="tag-badge">Linux</span>
        <span class="tag-badge">Terraform</span>
        <span class="tag-badge">Git</span>
    </div>
    """, unsafe_allow_html=True)

    # Chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    user_input = st.chat_input("Ask a DevOps question…")

    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        with st.chat_message("assistant"):
            if not is_devops_related(user_input):
                warning_msg = "⚠️ I specialise exclusively in DevOps, Cloud, Docker, Kubernetes, CI/CD, Linux, Git, and related infrastructure topics. Please ask something within that domain."
                st.warning(warning_msg)
                st.session_state.chat_history.append({"role": "assistant", "content": warning_msg})
            else:
                with st.spinner("Processing…"):
                    try:
                        if db:
                            docs    = db.similarity_search(user_input, k=2)
                            context = "\n".join([doc.page_content[:200] for doc in docs]) if docs else ""
                            prompt  = f"""You are a DevOps expert assistant. Only answer questions related to DevOps, Cloud, Docker, Kubernetes, CI/CD, Linux, Git, and related technologies.

Context: {context}

Question: {user_input}

Provide a clear and concise answer focused ONLY on DevOps topics:"""
                        else:
                            prompt = f"""You are a DevOps expert assistant. Only answer questions related to DevOps, Cloud, Docker, Kubernetes, CI/CD, Linux, Git, and related technologies.

Question: {user_input}

Provide a clear and concise answer focused ONLY on DevOps topics:"""

                        response      = llm.invoke(prompt)
                        response_text = response.content if hasattr(response, 'content') else str(response)
                        st.write(response_text)
                        st.session_state.chat_history.append({"role": "assistant", "content": response_text})

                    except Exception as e:
                        error_msg = f"Error: {str(e)}"
                        st.error(error_msg)
                        st.session_state.chat_history.append({"role": "assistant", "content": error_msg})

    if st.session_state.chat_history:
        if st.button("⌫  Clear History", use_container_width=False):
            st.session_state.chat_history = []
            st.rerun()

# ----------------------------
# INTERVIEW MODE
# ----------------------------
elif st.session_state.mode == "interview":
    st.markdown("""
    <div style="margin-bottom:8px;">
        <div class="neon-title">Interview Simulator</div>
        <div class="neon-subtitle" style="margin-top:4px;">AI-evaluated · Real-time feedback</div>
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.interview_started:
        st.markdown("---")
        st.markdown("""
        <div style="font-family:'JetBrains Mono',monospace;font-size:0.72rem;color:var(--cyan);letter-spacing:.14em;text-transform:uppercase;margin-bottom:16px;">
            ▸ Configure Session
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            topic         = st.selectbox("Topic", ["DevOps", "Linux", "AWS", "Docker", "Kubernetes", "CI/CD", "Git"])
            num_questions = st.slider("Questions", 3, 10, 5)
        with col2:
            difficulty = st.selectbox("Difficulty", ["Beginner", "Intermediate", "Advanced"])

            # Visual difficulty indicator
            diff_colors = {"Beginner": "#00f5a0", "Intermediate": "#ffb700", "Advanced": "#ff4b8b"}
            diff_color  = diff_colors[difficulty]
            st.markdown(f"""
            <div style="margin-top:12px;background:var(--bg-elevated);border:1px solid {diff_color}33;
                        border-left:3px solid {diff_color};border-radius:8px;padding:12px 16px;">
                <div style="font-family:'JetBrains Mono',monospace;font-size:0.68rem;color:{diff_color};letter-spacing:.12em;text-transform:uppercase;">
                    {difficulty} Level
                </div>
                <div style="font-size:0.8rem;color:var(--text-secondary);margin-top:4px;font-family:'Space Grotesk',sans-serif;">
                    {"Foundational concepts & definitions" if difficulty=="Beginner" else "Practical scenarios & use-cases" if difficulty=="Intermediate" else "Architecture, internals & edge cases"}
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("⚡ LAUNCH INTERVIEW", use_container_width=True):
            with st.spinner("Generating questions…"):
                if difficulty == "Beginner":
                    difficulty_instruction = """BEGINNER LEVEL - Questions should be:
- Basic concepts and definitions
- Simple "What is..." or "Define..." questions
- Fundamental terminology
- Easy to answer for someone just starting
- No complex scenarios or troubleshooting"""
                elif difficulty == "Intermediate":
                    difficulty_instruction = """INTERMEDIATE LEVEL - Questions should be:
- How things work together
- Common use cases and practical scenarios
- Differences between concepts
- Basic troubleshooting
- Real-world applications"""
                else:
                    difficulty_instruction = """ADVANCED LEVEL - Questions should be:
- Deep technical details
- Architecture and internal workings
- Complex troubleshooting scenarios
- Best practices and optimization
- Security implications
- Performance tuning"""

                prompt = f"""Generate exactly {num_questions} {difficulty} level interview questions about {topic}.

{difficulty_instruction}

IMPORTANT:
- Make questions appropriate for {difficulty} level
- Format: numbered list (1. 2. 3...)
- Each question on a new line
- Keep questions clear and focused

Generate {num_questions} questions now:"""

                try:
                    response       = llm.invoke(prompt)
                    questions_text = response.content if hasattr(response, 'content') else str(response)
                    questions      = re.findall(r'\d+[\.\)]\s*(.+?)(?=\d+[\.\)]|$)', questions_text, re.DOTALL)
                    questions      = [q.strip() for q in questions if len(q.strip()) > 10][:num_questions]

                    if len(questions) >= 3:
                        st.session_state.questions         = questions
                        st.session_state.interview_started = True
                        st.session_state.current_question  = 0
                        st.session_state.score             = 0
                        st.session_state.answers_log       = []
                        st.rerun()
                    else:
                        st.error("Failed to generate enough questions. Please try again.")
                except Exception as e:
                    st.error(f"Error generating questions: {str(e)}")

    else:
        if st.session_state.current_question < len(st.session_state.questions):
            q_num    = st.session_state.current_question
            total_q  = len(st.session_state.questions)
            progress = q_num / total_q

            # Progress bar with label
            col_prog, col_score = st.columns([3, 1])
            with col_prog:
                st.progress(progress)
            with col_score:
                st.markdown(f"""
                <div class="stat-pill" style="float:right;">
                    ✓ {st.session_state.score} correct
                </div>
                """, unsafe_allow_html=True)

            question = st.session_state.questions[q_num]

            st.markdown(f"""
            <div class="question-card">
                <div class="q-number">Question {q_num + 1} of {total_q}</div>
                <div class="q-text">{question}</div>
            </div>
            """, unsafe_allow_html=True)

            user_answer = st.text_area(
                "Your Answer",
                key=f"answer_{q_num}",
                height=130,
                placeholder="Type your answer here…"
            )

            if st.button("→ SUBMIT ANSWER", use_container_width=True):
                if user_answer.strip():
                    with st.spinner("Evaluating…"):
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
                            answer_lower = user_answer.lower().strip()
                            word_count   = len(user_answer.split())
                            bad_answers  = ['idk', "i don't know", "dont know", "no idea", "not sure",
                                            "dunno", "dk", "?", "...", "na", "n/a"]

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
                                response    = llm.invoke(eval_prompt)
                                evaluation  = response.content if hasattr(response, 'content') else str(response)
                                first_line  = evaluation.strip().split('\n')[0].upper()
                                is_correct  = "CORRECT" in first_line and "INCORRECT" not in first_line

                                score_match = re.search(r'(\d+)/10', evaluation)
                                if score_match:
                                    score_val  = int(score_match.group(1))
                                    is_correct = score_val >= 5

                            if is_correct:
                                st.session_state.score += 1
                                st.success("✓ Correct!")
                            else:
                                st.error("✗ Incorrect")

                            st.markdown("**Feedback:**")
                            st.write(evaluation)

                            st.session_state.answers_log.append({
                                'question': question,
                                'answer':   user_answer,
                                'correct':  is_correct,
                                'feedback': evaluation
                            })

                            st.session_state.current_question += 1

                            if st.session_state.current_question < total_q:
                                if st.button("Next →"):
                                    st.rerun()
                            else:
                                st.rerun()

                        except Exception as e:
                            st.error(f"Evaluation error: {str(e)}")
                else:
                    st.warning("Please enter an answer before submitting.")

        else:
            st.balloons()

            score      = st.session_state.score
            total      = len(st.session_state.questions)
            percentage = (score / total) * 100
            pass_score = int(total * 0.5)
            passed     = score >= pass_score

            st.markdown(f"""
            <div class="result-banner">
                <div style="font-family:'JetBrains Mono',monospace;font-size:0.65rem;color:var(--cyan);letter-spacing:.2em;text-transform:uppercase;margin-bottom:12px;">
                    Interview Complete
                </div>
                <div class="result-score">{percentage:.0f}%</div>
                <div class="result-label">{score} of {total} correct · {"PASSED" if passed else "FAILED"}</div>
                <div style="margin-top:16px;display:inline-block;padding:6px 18px;
                    background:{"rgba(0,245,160,0.15)" if passed else "rgba(255,75,110,0.12)"};
                    border:1px solid {"var(--green)" if passed else "var(--pink)"};
                    border-radius:20px;font-family:'JetBrains Mono',monospace;font-size:0.8rem;
                    color:{"var(--green)" if passed else "var(--pink)"} !important;">
                    {"✓ CERTIFIED PASS" if passed else "✗ NEEDS IMPROVEMENT"}
                </div>
            </div>
            """, unsafe_allow_html=True)

            if percentage >= 80:   st.success("🌟 Excellent — strong DevOps proficiency demonstrated.")
            elif percentage >= 70: st.success("👍 Good performance — you passed the assessment.")
            elif percentage >= 50: st.warning("⚠️ Borderline pass — review the feedback below.")
            else:                  st.error("📚 Below passing threshold — study the concepts and retry.")

            st.markdown("---")
            st.markdown("""
            <div style="font-family:'JetBrains Mono',monospace;font-size:0.7rem;color:var(--cyan);
                        letter-spacing:.14em;text-transform:uppercase;margin-bottom:12px;">
                ▸ Answer Review
            </div>
            """, unsafe_allow_html=True)

            for idx, log in enumerate(st.session_state.answers_log, 1):
                icon = "✓" if log['correct'] else "✗"
                color = "var(--green)" if log['correct'] else "var(--pink)"
                with st.expander(f"{icon}  Question {idx}"):
                    st.markdown(f"**Question:** {log['question']}")
                    st.markdown(f"**Your Answer:** {log['answer']}")
                    st.markdown("**Feedback:**")
                    st.write(log['feedback'])

            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("↺ NEW INTERVIEW", use_container_width=True):
                st.session_state.interview_started = False
                st.session_state.current_question  = 0
                st.session_state.score             = 0
                st.session_state.questions         = []
                st.session_state.answers_log       = []
                st.rerun()

# ----------------------------
# HOME PAGE
# ----------------------------
else:
    # Hero section
    st.markdown(f"""
    <div style="padding:32px 0 24px;">
        <div style="font-family:'JetBrains Mono',monospace;font-size:0.7rem;color:var(--cyan);
                    letter-spacing:.2em;text-transform:uppercase;margin-bottom:12px;opacity:.8;">
            <span class="online-dot"></span> System Online · Welcome back
        </div>
        <div class="neon-title" style="font-size:3rem;">
            Hello, {st.session_state.current_user['full_name'].split()[0]}
        </div>
        <div style="font-family:'Space Grotesk',sans-serif;font-size:1.05rem;color:var(--text-secondary);
                    margin-top:8px;max-width:540px;line-height:1.6;">
            Your AI-powered DevOps companion. Ask questions or test your skills with an interview simulation.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="background:rgba(0,229,255,0.05);border:1px solid rgba(0,229,255,0.18);border-radius:10px;
                padding:12px 18px;margin-bottom:28px;display:flex;align-items:center;gap:10px;">
        <span style="color:var(--cyan);font-size:1.1rem;">ℹ</span>
        <span style="font-family:'Space Grotesk',sans-serif;font-size:0.88rem;color:var(--text-secondary);">
            This assistant is scoped to <strong style="color:var(--text-primary);">DevOps, Cloud, Docker, Kubernetes, CI/CD, Linux, Git</strong> and related topics only.
        </span>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("""
        <div class="feature-card">
            <span class="feature-icon">💬</span>
            <div class="feature-title">Chat Mode</div>
            <div class="feature-item">Instant AI-powered answers</div>
            <div class="feature-item">RAG-enhanced knowledge base</div>
            <div class="feature-item">Context-aware responses</div>
            <div class="feature-item">DevOps topics only</div>
            <div style="margin-top:20px;height:2px;background:linear-gradient(90deg,var(--cyan),transparent);border-radius:1px;"></div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("OPEN CHAT  →", use_container_width=True, key="home_chat"):
            st.session_state.mode = "chat"
            st.rerun()

    with col2:
        st.markdown("""
        <div class="feature-card green-tint">
            <span class="feature-icon" style="filter:drop-shadow(0 0 12px rgba(0,245,160,.5));">🎯</span>
            <div class="feature-title">Interview Mode</div>
            <div class="feature-item">AI-generated questions</div>
            <div class="feature-item">Beginner → Advanced difficulty</div>
            <div class="feature-item">Strict real-time evaluation</div>
            <div class="feature-item">Detailed feedback per answer</div>
            <div style="margin-top:20px;height:2px;background:linear-gradient(90deg,var(--green),transparent);border-radius:1px;"></div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("START INTERVIEW  →", use_container_width=True, key="home_interview"):
            st.session_state.mode = "interview"
            st.rerun()

    # Tech stack badges
    st.markdown("---")
    st.markdown("""
    <div style="display:flex;flex-wrap:wrap;gap:8px;align-items:center;">
        <span style="font-family:'JetBrains Mono',monospace;font-size:0.68rem;color:var(--text-muted);letter-spacing:.1em;text-transform:uppercase;margin-right:6px;">Stack</span>
        <span class="tag-badge">Llama 3.1</span>
        <span class="tag-badge">Groq API</span>
        <span class="tag-badge">Streamlit</span>
        <span class="tag-badge">LangChain</span>
        <span class="tag-badge">ChromaDB</span>
        <span class="tag-badge">MySQL</span>
    </div>
    """, unsafe_allow_html=True)

# ----------------------------
# FOOTER
# ----------------------------
st.markdown("""
<div class="custom-footer">
    <div class="creator">Created by Ansari Mantasha</div>
    <div>
        <a href="https://github.com/mantu0tech" target="_blank">GitHub</a>
        <a href="https://www.linkedin.com/in/mantasha-ansari-47162b24a/" target="_blank">LinkedIn</a>
    </div>
    <div class="stack">Powered by Llama3 · Groq API · Streamlit · MySQL</div>
</div>
""", unsafe_allow_html=True)