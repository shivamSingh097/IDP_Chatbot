# db.py
import sqlite3
from typing import Optional, Dict

DB_PATH = "app.db"

def init_db():
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        email TEXT UNIQUE,
        password_hash TEXT,
        name TEXT,
        created_at TEXT
    );
    """)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        role TEXT,
        text TEXT,
        created_at TEXT
    );
    """)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS uploads (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        filename TEXT,
        content TEXT,
        created_at TEXT
    );
    """)
    con.commit()
    con.close()

def get_user_by_email(email:str) -> Optional[Dict]:
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("SELECT id, email, password_hash, name FROM users WHERE email = ?", (email,))
    row = cur.fetchone()
    con.close()
    if not row:
        return None
    return {"id":row[0], "email":row[1], "password_hash":row[2], "name":row[3]}

def create_user(email:str, password_hash:str, name:str):
    from datetime import datetime
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("INSERT INTO users (email,password_hash,name,created_at) VALUES (?,?,?,?)",
                (email, password_hash, name, datetime.utcnow().isoformat()))
    con.commit()
    con.close()

def save_message(user_id:int, role:str, text:str):
    from datetime import datetime
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("INSERT INTO messages (user_id,role,text,created_at) VALUES (?,?,?,?)",
                (user_id, role, text, datetime.utcnow().isoformat()))
    con.commit()
    con.close()

def save_upload(user_id:int, filename:str, content:str):
    from datetime import datetime
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("INSERT INTO uploads (user_id,filename,content,created_at) VALUES (?,?,?,?)",
                (user_id, filename, content, datetime.utcnow().isoformat()))
    con.commit()
    con.close()
