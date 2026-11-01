"""Load the fixed action--noun--role support inventory."""
import hashlib
import json


def encoded(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def check_hash(document):
    payload = {k: v for k, v in document.items() if k != 'content_sha256'}
    if document.get('content_sha256') != digest(encoded(payload)):
        raise ValueError('Content hash mismatch')

class Compatibility:
    """Validated action--noun--role support; a table entry is a role bitset."""
    def __init__(self, document, expected_vocab_sha=None):
        check_hash(document)
        if document.get('status') != 'HUMAN_APPROVED' or document.get('null_state_allowed') is not True:
            raise ValueError('Not an approved null-preserving table')
        if (document.get('policy', {}).get('unresolved_at_freeze') == 'block'
                and document.get('review', {}).get('unresolved_explicitly_allowed') != 0):
            raise ValueError('Strict ontology cannot contain implicitly permitted unresolved states')
        if expected_vocab_sha and document['vocabulary_sha256'] != expected_vocab_sha:
            raise ValueError('Compatibility table vocabulary hash differs')
        self.document = document
        self.axes = document['vocabulary']
        if set(self.axes) != {'verbs', 'nouns', 'roles'} or any(not a or len(a) != len(set(a)) for a in self.axes.values()):
            raise ValueError('Invalid class order')
        self.bits = document['role_bits_verb_noun']
        nr = len(self.axes['roles'])
        if len(self.bits) != len(self.axes['verbs']) or any(len(row) != len(self.axes['nouns']) for row in self.bits):
            raise ValueError('Compatibility shape mismatch')
        if any(type(b) is not int or not 0 <= b < 1 << nr for row in self.bits for b in row):
            raise ValueError('Invalid role bitset')
        self.indices = {k: {name: i for i, name in enumerate(a)} for k, a in self.axes.items()}

    def allows(self, verb, noun, role):
        v, n, r = (self.indices[k][name] for k, name in zip(('verbs', 'nouns', 'roles'), (verb, noun, role)))
        return bool(self.bits[v][n] & (1 << r))

    def binding(self):
        return {'content_sha256': self.document['content_sha256'],
                'vocabulary_sha256': self.document['vocabulary_sha256'],
                'class_order_sha256': digest(encoded(self.axes)), 'scope': 'HEIR'}
