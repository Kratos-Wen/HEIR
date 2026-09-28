import hashlib
from pathlib import Path

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()

def noun_order(vocabulary):
    names = [r['id'] for r in vocabulary['nouns']]
    if len(set(names)) != len(names) or 'person' not in names:
        raise ValueError('Noun IDs must be unique and include person')
    return ['person'] + [n for n in names if n != 'person']
