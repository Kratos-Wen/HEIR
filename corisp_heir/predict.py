"""Predict HEIR relations and native sets from a fixed CoRISP checkpoint."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('data', 'detector', 'prototypes', 'compatibility', 'checkpoint', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--split', choices=('val', 'test'), required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    import torch
    from torch.utils.data import DataLoader
    from .environment import restore_execution, verify_core
    from .checkpoint import load_adaptation
    from .data import HEIREvents, collate
    from .model import build
    from evaluation.export import native_prediction
    from heir_protocol.compatibility import Compatibility
    from heir_training.data import sha256

    if not torch.cuda.is_available():
        raise RuntimeError('Prediction requires CUDA')
    torch.set_num_threads(1)
    verify_core()
    # Only load checkpoints supplied by a trusted source.
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    protocol = state['protocol']
    if protocol['schema'] != 'corisp_heir_training_v1':
        raise ValueError('Expected a checkpoint from corisp_heir.train')
    restore_execution(protocol['execution'])
    data = HEIREvents(args.data, training=False, split=args.split)
    support = json.loads(args.compatibility.read_text())
    if Compatibility(support).binding() != protocol['compatibility_binding']:
        raise ValueError('Checkpoint support does not match the supplied inventory')
    for name, path in [('train.json', args.data / 'annotations/train.json'),
                       ('vocabulary.json', args.data / 'vocabulary.json'),
                       ('detector', args.detector), ('prototypes', args.prototypes)]:
        if sha256(path) != protocol['inputs'][name]:
            raise ValueError(f'Checkpoint input mismatch: {name}')
    model = build(data, args.detector, args.prototypes, support, protocol['ablation'])
    load_adaptation(model, state['model'])
    del state
    model = model.cuda().eval()
    loader = DataLoader(data, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as handle, torch.no_grad():
        for images, targets in loader:
            with torch.autocast('cuda', dtype=torch.bfloat16):
                outputs = model([image.cuda() for image in images])
            for output, target in zip(outputs, targets):
                record = native_prediction(output, target['image_id'], target['orig_size'],
                                           data.nouns, data.verbs, data.roles)
                handle.write(json.dumps(record, allow_nan=False) + '\n')
    print(f'Predicted {len(data)} images with native set decoding', flush=True)


if __name__ == '__main__':
    main()
