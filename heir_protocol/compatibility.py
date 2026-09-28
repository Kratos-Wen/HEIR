import json
from pathlib import Path
from .closed import digest, encoded

FROZEN = 'heir_semantic_compatibility_v1'

def check_hash(document, schema):
    if document.get('schema') != schema:
        raise ValueError(f'Expected {schema}')
    payload = {k: v for k, v in document.items() if k != 'content_sha256'}
    if document.get('content_sha256') != digest(encoded(payload)):
        raise ValueError('Content hash mismatch')

class Compatibility:
    """Framework-neutral shared loader. Projection masks do not replace the 3D mask."""
    def __init__(self, document, expected_vocab_sha=None):
        check_hash(document, FROZEN)
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

    @classmethod
    def load(cls, path, expected_vocab_sha=None):
        return cls(json.loads(Path(path).read_bytes()), expected_vocab_sha)

    def allows(self, verb, noun, role):
        v, n, r = (self.indices[k][name] for k, name in zip(('verbs', 'nouns', 'roles'), (verb, noun, role)))
        return bool(self.bits[v][n] & (1 << r))

    def pair_role_mask(self, nouns):
        return [[[self.allows(v, n, r) for r in self.axes['roles']] for v in self.axes['verbs']] for n in nouns]

    def object_action_mask(self):
        return [[self.bits[v][n] != 0 for v in range(len(self.bits))] for n in range(len(self.axes['nouns']))]

    def verb_role_mask(self):
        return [[any(bits & (1 << r) for bits in row) for r in range(len(self.axes['roles']))] for row in self.bits]

    def binding(self):
        return {'schema': FROZEN, 'content_sha256': self.document['content_sha256'],
                'vocabulary_sha256': self.document['vocabulary_sha256'],
                'class_order_sha256': digest(encoded(self.axes)), 'scope': 'HEIR'}
