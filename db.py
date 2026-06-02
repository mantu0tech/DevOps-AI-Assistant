"""
database.py - MySQL Database Handler for DevOps AI Assistant
Author: Extended by Senior Python Developer
"""

import mysql.connector
from mysql.connector import Error
import bcrypt
import os
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))


# ----------------------------
# DB CONFIG (reads from .env)
# ----------------------------
DB_CONFIG = {
    "host":     os.getenv("DB_HOST", "localhost"),
    "port":     int(os.getenv("DB_PORT", 3306)),
    "user":     os.getenv("DB_USER", "root"),
    "password": os.getenv("DB_PASSWORD", ""),
    "database": os.getenv("DB_NAME", "devops_assistant"),
}


# ----------------------------
# CONNECTION HELPER
# ----------------------------
def get_connection():
    """Return a live MySQL connection, or raise on failure."""
    return mysql.connector.connect(**DB_CONFIG)


# ----------------------------
# SCHEMA BOOTSTRAP
# ----------------------------
def initialize_database():
    """
    Create the database and users table if they don't exist.
    Safe to call on every app start.
    """
    try:
        # Connect WITHOUT specifying the database so we can CREATE it
        cfg_no_db = {k: v for k, v in DB_CONFIG.items() if k != "database"}
        conn = mysql.connector.connect(**cfg_no_db)
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


# ----------------------------
# USER OPERATIONS
# ----------------------------
def register_user(full_name: str, username: str, email: str, password: str):
    """
    Hash password with bcrypt and insert a new user row.
    Returns (success: bool, message: str)
    """
    try:
        # bcrypt hash  (work factor 12 is a sensible default)
        pw_hash = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt(rounds=12)).decode("utf-8")

        conn   = get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO users (full_name, username, email, password_hash)
            VALUES (%s, %s, %s, %s)
            """,
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


def login_user(username_or_email: str, password: str):
    """
    Verify credentials and update last_login timestamp.
    Returns (success: bool, message: str, user_dict | None)
    """
    try:
        conn   = get_connection()
        cursor = conn.cursor(dictionary=True)

        # Allow login with either username or email
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

        # Update last_login
        cursor.execute(
            "UPDATE users SET last_login = NOW() WHERE id = %s",
            (user["id"],),
        )
        conn.commit()
        cursor.close()
        conn.close()

        # Don't expose the hash downstream
        user.pop("password_hash", None)
        return True, f"Welcome back, {user['full_name']}! 🚀", user

    except Error as e:
        return False, f"Database error: {e}", None


def username_exists(username: str) -> bool:
    """Quick uniqueness check used for real-time feedback during registration."""
    try:
        conn   = get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT 1 FROM users WHERE username = %s LIMIT 1", (username.lower(),))
        exists = cursor.fetchone() is not None
        cursor.close(); conn.close()
        return exists
    except Error:
        return False


def email_exists(email: str) -> bool:
    """Quick uniqueness check used for real-time feedback during registration."""
    try:
        conn   = get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT 1 FROM users WHERE email = %s LIMIT 1", (email.lower(),))
        exists = cursor.fetchone() is not None
        cursor.close(); conn.close()
        return exists
    except Error:
        return False