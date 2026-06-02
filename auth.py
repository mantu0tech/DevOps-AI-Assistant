"""
auth_pages.py - Login & Register UI components for DevOps AI Assistant
Author: Extended by Senior Python Developer
"""

import streamlit as st
import re
from database import register_user, login_user


# ----------------------------
# VALIDATORS
# ----------------------------
def _valid_email(email: str) -> bool:
    return bool(re.match(r"^[\w\.\+\-]+@[\w\-]+\.[a-zA-Z]{2,}$", email))

def _valid_password(pw: str) -> tuple[bool, str]:
    if len(pw) < 8:
        return False, "Password must be at least 8 characters."
    if not re.search(r"[A-Z]", pw):
        return False, "Password must contain at least one uppercase letter."
    if not re.search(r"\d", pw):
        return False, "Password must contain at least one digit."
    return True, ""


# ----------------------------
# SHARED STYLES (injected once)
# ----------------------------
AUTH_CSS = """
<style>
/* ---- Auth card ---- */
.auth-card {
    background: rgba(30, 30, 63, 0.92);
    border: 1px solid #4a5568;
    border-radius: 16px;
    padding: 2.5rem 2rem;
    max-width: 480px;
    margin: 2rem auto;
    box-shadow: 0 8px 32px rgba(102,126,234,0.18);
}
.auth-card h2 {
    text-align: center;
    color: #a78bfa !important;
    margin-bottom: 1.5rem;
    font-size: 1.8rem;
}
.auth-card .subtitle {
    text-align: center;
    color: #94a3b8;
    margin-bottom: 1.5rem;
    font-size: 0.95rem;
}
/* ---- Input labels inside card ---- */
.auth-card label { color: #e2e8f0 !important; }

/* ---- Divider ---- */
.auth-divider {
    text-align: center;
    color: #64748b;
    margin: 1rem 0;
    font-size: 0.85rem;
}

/* ---- Strength bar ---- */
.strength-bar-wrap { margin: 4px 0 12px; }
.strength-bar-bg {
    width: 100%; height: 6px;
    background: #2d3748; border-radius: 3px;
}
.strength-bar-fill {
    height: 6px; border-radius: 3px;
    transition: width .3s, background .3s;
}
.strength-label { font-size: 0.78rem; margin-top: 3px; }
</style>
"""


def _password_strength_html(pw: str) -> str:
    score = 0
    if len(pw) >= 8:       score += 1
    if re.search(r"[A-Z]", pw): score += 1
    if re.search(r"\d", pw):    score += 1
    if re.search(r"[^A-Za-z0-9]", pw): score += 1

    colours = ["#ef4444", "#f97316", "#eab308", "#22c55e"]
    labels  = ["Weak", "Fair", "Good", "Strong"]
    pct     = (score / 4) * 100
    colour  = colours[max(score - 1, 0)]
    label   = labels[max(score - 1, 0)]

    return f"""
<div class="strength-bar-wrap">
  <div class="strength-bar-bg">
    <div class="strength-bar-fill" style="width:{pct}%; background:{colour};"></div>
  </div>
  <div class="strength-label" style="color:{colour};">{label} password</div>
</div>
"""


# ----------------------------
# LOGIN PAGE
# ----------------------------
def render_login():
    st.markdown(AUTH_CSS, unsafe_allow_html=True)
    st.markdown('<div class="auth-card">', unsafe_allow_html=True)
    st.markdown("## 🔐 Sign In")
    st.markdown('<p class="subtitle">Welcome back! Log in to continue.</p>', unsafe_allow_html=True)

    with st.form("login_form", clear_on_submit=False):
        identifier = st.text_input(
            "Username or Email",
            placeholder="john_doe  or  john@example.com",
        )
        password = st.text_input(
            "Password",
            type="password",
            placeholder="••••••••",
        )

        col_submit, col_forgot = st.columns([2, 1])
        with col_submit:
            submitted = st.form_submit_button("🚀 Login", use_container_width=True)
        with col_forgot:
            st.markdown(
                "<div style='text-align:right;padding-top:8px;'>"
                "<small style='color:#a78bfa;'>Forgot password?</small></div>",
                unsafe_allow_html=True,
            )

        if submitted:
            if not identifier or not password:
                st.error("Please fill in all fields.")
            else:
                with st.spinner("Authenticating…"):
                    ok, msg, user = login_user(identifier, password)
                if ok:
                    st.success(msg)
                    st.session_state.logged_in   = True
                    st.session_state.current_user = user
                    st.session_state.auth_page    = None
                    st.rerun()
                else:
                    st.error(msg)

    st.markdown('<div class="auth-divider">— or —</div>', unsafe_allow_html=True)

    if st.button("✨ Create a new account", use_container_width=True, key="go_register"):
        st.session_state.auth_page = "register"
        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


# ----------------------------
# REGISTER PAGE
# ----------------------------
def render_register():
    st.markdown(AUTH_CSS, unsafe_allow_html=True)
    st.markdown('<div class="auth-card">', unsafe_allow_html=True)
    st.markdown("## 🚀 Create Account")
    st.markdown('<p class="subtitle">Join the DevOps AI Assistant platform.</p>', unsafe_allow_html=True)

    with st.form("register_form", clear_on_submit=False):
        full_name = st.text_input("Full Name", placeholder="John Doe")
        username  = st.text_input("Username",  placeholder="john_doe (no spaces)")
        email     = st.text_input("Email",     placeholder="john@example.com")
        password  = st.text_input("Password",  type="password", placeholder="Min 8 chars, 1 uppercase, 1 digit")
        confirm   = st.text_input("Confirm Password", type="password", placeholder="Re-enter password")

        submitted = st.form_submit_button("✅ Register", use_container_width=True)

        # Live strength bar (renders every keystroke via form re-draw)
        if password:
            st.markdown(_password_strength_html(password), unsafe_allow_html=True)

        if submitted:
            # ---- Client-side validation ----
            errors = []
            if not full_name.strip():
                errors.append("Full name is required.")
            if not username.strip() or " " in username:
                errors.append("Username is required and must not contain spaces.")
            if not _valid_email(email):
                errors.append("Enter a valid email address.")
            pw_ok, pw_msg = _valid_password(password)
            if not pw_ok:
                errors.append(pw_msg)
            if password != confirm:
                errors.append("Passwords do not match.")

            if errors:
                for err in errors:
                    st.error(err)
            else:
                with st.spinner("Creating your account…"):
                    ok, msg = register_user(full_name, username, email, password)
                if ok:
                    st.success(msg)
                    st.balloons()
                    st.session_state.auth_page = "login"
                    st.rerun()
                else:
                    st.error(msg)

    st.markdown('<div class="auth-divider">— or —</div>', unsafe_allow_html=True)

    if st.button("🔐 Already have an account? Log in", use_container_width=True, key="go_login"):
        st.session_state.auth_page = "login"
        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)