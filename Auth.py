# auth.py
import hashlib
import os
from dotenv import load_dotenv
load_dotenv()
SECRET_SALT = os.getenv("SECRET_SALT", "CHANGE_ME")

def hash_password(password: str) -> str:
    s = (SECRET_SALT + password).encode('utf-8')
    return hashlib.sha256(s).hexdigest()

def check_password(password: str, password_hash: str) -> bool:
    return hash_password(password) == password_hash
