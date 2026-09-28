import random
import numpy as np
import torch
import torch.distributed as dist

def rng_state():
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state()}

def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    torch.cuda.set_rng_state(state['cuda'])

def trainable_state(model):
    # Save every non-detector/non-semantic tensor, including ontology buffers.
    return {k: v.detach().cpu() for k, v in model.state_dict().items()
            if not k.startswith(('detector.', 'semantic_backbone.'))}

def load_adaptation(model, state):
    expected = set(trainable_state(model))
    if set(state) != expected:
        raise ValueError('Adaptation checkpoint keys do not match the frozen architecture')
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.unexpected_keys or any(not k.startswith(('detector.', 'semantic_backbone.'))
                                           for k in incompatible.missing_keys):
        raise ValueError('Unexpected checkpoint omissions')

def save(path, model, optimizer, scheduler, epoch, protocol):
    states = [None]*dist.get_world_size()
    dist.all_gather_object(states, rng_state())
    if dist.get_rank() == 0:
        temp = path.with_suffix('.tmp')
        torch.save({'model': trainable_state(model), 'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(), 'epoch': epoch,
                    'rng_states': states, 'protocol': protocol}, temp)
        temp.replace(path)
    dist.barrier()

def move(batch, device):
    images, targets = batch
    return [x.to(device) for x in images], [
        {k: v.to(device) if torch.is_tensor(v) else v for k, v in t.items()} for t in targets]
