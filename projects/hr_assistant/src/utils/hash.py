import hashlib

def get_chunk_id(content: str) -> str:
    return hashlib.sha256(content.encode('utf-8')).hexdigest()