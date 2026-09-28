import hashlib
import json

def encoded(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()

def digest(data):
    return hashlib.sha256(data).hexdigest()
