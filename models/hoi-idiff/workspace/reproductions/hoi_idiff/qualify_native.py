"""Real-image GPU gate before independent diffusion training."""

import argparse
import json
from pathlib import Path
import time

import torch

from .detector import build_detector, extract, digest, ROOT
from .vcoco import annotations, pair_targets, DATA
from .run import image_tensor, train_image_path, training_batch, write_json
from .diffusion import ground_truth, initialize, ForwardDiffusion
from .model import SliceDiT
from .transition import exact_transition
from .vcoco import regression_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve qualification evidence')
    torch.manual_seed(42)
    device = torch.device('cuda', 0)
    rows, channels = annotations()
    detector, loading = build_detector(device)
    torch.cuda.reset_peak_memory_stats()
    cases = []
    model = SliceDiT(81, 29, 777).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    process = ForwardDiffusion().to(device)
    candidates = [r for r in rows if sum(1 in p['labels'] for p in r['people']) > 1]
    for row in candidates[:2]:
        start = time.monotonic()
        tensor, w, h = image_tensor(train_image_path(row['file_name']), device)
        cache = extract(detector, tensor)
        cache['labels'], cache['target_nouns'] = pair_targets(cache, row, w, h, channels)
        cache['image_id'] = row['image_id']
        app, prior, label, known, noun = training_batch(cache, 0, device, 29)
        if not known.any() or not (label == 1).any():
            raise ValueError('Real qualification image has no localized positive training pair')
        app, prior, label, known, noun = [x.repeat_interleave(10, 0) for x in (app, prior, label, known, noun)]
        clean = ground_truth(noun, label, 81)
        prior = initialize(prior, 29)
        clean = torch.where(known[:, None, :, None], clean, prior)
        steps = torch.randint(1, 51, (len(clean),), device=device)
        current, previous = exact_transition(clean, prior, steps, process)
        prediction = model(current, steps, app)
        summed, count = regression_loss(prediction, previous, known)
        loss = summed / count
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        if not torch.isfinite(loss) or not torch.isfinite(norm) or norm <= 0:
            raise ValueError('Invalid native full-model gradient')
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise ValueError('Missing/nonfinite diffusion parameter gradient')
        if any(p.grad is not None for p in detector.parameters()):
            raise ValueError('Frozen detector received gradients')
        optimizer.step()
        cases.append({'image_id': row['image_id'], 'candidate_pairs': len(cache['appearance']),
            'model_examples': len(app), 'positive_assignments': int((cache['labels'] == 1).sum()),
            'loss': float(loss), 'gradient_norm': float(norm), 'seconds': time.monotonic() - start})
    output = {'status': 'PASS', 'scope': 'independent transition-regression adaptation, not full author equivalence',
        'detector': loading, 'cases': cases, 'peak_mib': torch.cuda.max_memory_allocated() / 1024**2,
        'test_labels_read': False,
        'source_sha256': {str(p): digest(p) for p in Path(__file__).parent.glob('*.py')}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, output)
    print(json.dumps(output), flush=True)


if __name__ == '__main__':
    main()
