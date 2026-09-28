"""Dataset-specific constraints; likelihood and quotient remain the frozen core."""
from scipy.optimize import linear_sum_assignment
import torch
from torchvision.ops import box_iou

from . import environment
from corisp.grounded_role_set import RoleFillerTargetGroup


class EventTargets:
    def __init__(self, proposal, agent_entities, pairs, target, cached=False):
        self.proposal, self.pairs, self.target = proposal, pairs, target
        self.missing_entity_events = 0
        boxes = target['person_boxes']
        overlap = box_iou(proposal['boxes'][agent_entities], boxes)
        self.overlap = overlap
        self.assignment = {}
        ambiguous = torch.zeros(len(boxes), dtype=torch.bool, device=boxes.device)
        if len(boxes):
            equal = torch.isclose(boxes[:, None], boxes[None], atol=1e-4, rtol=0).all(-1)
            equal.fill_diagonal_(False)
            ambiguous = equal.any(-1)
        self.ignored = ((overlap >= .5) & ambiguous[None]).any(-1)
        valid = (overlap >= .5) & ~ambiguous[None] & ~self.ignored[:, None]
        reward = valid * (float(min(overlap.shape)+1) + overlap)
        if reward.numel():
            rows, cols = linear_sum_assignment(reward.detach().cpu().numpy(), maximize=True)
            self.assignment = {int(r): int(c) for r, c in zip(rows, cols) if bool(valid[r, c])}
        self.identity = {int(value): i for i, value in enumerate(target['instance_ids'])}
        self.cached = cached
        if cached:
            # Matching metadata has no gradients. Copy once, not once per
            # person/action event; all IoU and noun matching below is unchanged.
            self.cpu_ignored = self.ignored.tolist()
            self.cpu_overlap = self.overlap.detach().cpu()
            self.cpu_observed = target['action_observed'].tolist()
            self.cpu_person_ids = target['person_ids'].tolist()
            self.cpu_relations = {}
            for subject, entity, action, role in target['relations'].tolist():
                self.cpu_relations.setdefault((subject, action), []).append((entity, role))

    def groups(self, agent, action, pair_indices):
        if self.cached:
            return self.cached_groups(agent, action, pair_indices)
        if bool(self.ignored[agent]):
            return None
        person = self.assignment.get(agent)
        if person is None:
            if self.target['partial']:
                # A duplicate of an observed positive is suppressed; unrelated
                # unreviewed people/actions are not made into negative events.
                if self.overlap.shape[1] == 0:
                    return None
                quality, nearest = self.overlap[agent].max(0)
                return [] if quality >= .5 and self.target['action_observed'][nearest, action] >= 0 else None
            if self.overlap.shape[1] and bool((self.overlap[agent] >= .5).any()):
                nearest = self.overlap[agent].argmax()
                if self.target['action_observed'][nearest, action] < 0:
                    return None
            return []
        state = int(self.target['action_observed'][person, action])
        if state < 0:
            return None
        if state == 0:
            return []
        subject = self.target['person_ids'][person]
        relations = self.target['relations']
        selected = relations[(relations[:, 0] == subject) & (relations[:, 2] == action)]
        if not len(selected):
            raise ValueError('Positive event has no visible annotated members')
        entities = self.pairs[pair_indices, 1].long()
        groups = []
        for _, identity, _, role in selected:
            gt = self.identity[int(identity)]
            quality = box_iou(self.proposal['boxes'][entities], self.target['boxes'][gt:gt+1]).flatten()
            compatible = self.proposal['labels'][entities] == self.target['categories'][gt]
            rows = torch.where((quality >= .5) & compatible)[0]
            if not len(rows):
                # Frozen detector cannot represent this complete set: ignore
                # its training likelihood, but never omit it from test recall.
                self.missing_entity_events += 1
                return None
            groups.append(RoleFillerTargetGroup(int(role), rows))
        return groups

    def cached_groups(self, agent, action, pair_indices):
        if self.cpu_ignored[agent]:
            return None
        person = self.assignment.get(agent)
        if person is None:
            overlap = self.cpu_overlap[agent]
            if self.target['partial']:
                if not len(overlap):
                    return None
                quality, nearest = overlap.max(0)
                return [] if quality >= .5 and self.cpu_observed[int(nearest)][action] >= 0 else None
            if len(overlap) and bool((overlap >= .5).any()):
                if self.cpu_observed[int(overlap.argmax())][action] < 0:
                    return None
            return []
        state = self.cpu_observed[person][action]
        if state < 0:
            return None
        if state == 0:
            return []
        selected = self.cpu_relations.get((self.cpu_person_ids[person], action), [])
        if not selected:
            raise ValueError('Positive event has no visible annotated members')
        entities = self.pairs[pair_indices, 1].long()
        groups = []
        for identity, role in selected:
            gt = self.identity[identity]
            quality = box_iou(self.proposal['boxes'][entities], self.target['boxes'][gt:gt+1]).flatten()
            compatible = self.proposal['labels'][entities] == self.target['categories'][gt]
            rows = torch.where((quality >= .5) & compatible)[0]
            if not len(rows):
                self.missing_entity_events += 1
                return None
            groups.append(RoleFillerTargetGroup(role, rows))
        return groups
