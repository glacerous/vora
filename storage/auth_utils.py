import hashlib
import secrets

ITERATIONS = 600000

def hash_password(password: str) -> tuple[str, str]:
    """
    Hashes a password using hashlib.pbkdf2_hmac with SHA-256.
    Returns a tuple of (hex_hash, hex_salt).
    """
    salt = secrets.token_hex(16)
    salt_bytes = salt.encode('utf-8')
    pwd_bytes = password.encode('utf-8')
    
    dk = hashlib.pbkdf2_hmac('sha256', pwd_bytes, salt_bytes, ITERATIONS)
    password_hash = dk.hex()
    return password_hash, salt

def verify_password(password: str, password_hash: str, password_salt: str) -> bool:
    """
    Verifies a password against its stored hash and salt using constant-time comparison.
    """
    salt_bytes = password_salt.encode('utf-8')
    pwd_bytes = password.encode('utf-8')
    
    dk = hashlib.pbkdf2_hmac('sha256', pwd_bytes, salt_bytes, ITERATIONS)
    computed_hash = dk.hex()
    
    return secrets.compare_digest(computed_hash.encode('utf-8'), password_hash.encode('utf-8'))
