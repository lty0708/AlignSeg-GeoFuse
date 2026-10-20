from pathlib import Path
import argparse
import json
import platform
import sys
import tempfile
sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))
import torch
import torch.nn.functional as F
from nets import AlignedMultiViewResUNet2D, replace_batch_norm_with_group_norm
from mva_multitask_s2r1 import MultiTaskR1, TARGET_SPECS

def flatten_tensors(value):
    if isinstance(value, torch.Tensor):
        return [value]
    return [t for item in value.values() for t in flatten_tensors(item)]

def update_and_reload(model, inputs, loss_fn, factory):
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    output = model(*inputs)
    loss = loss_fn(output)
    assert torch.isfinite(loss), 'Non-finite loss'
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients and all((torch.isfinite(g).all() for g in gradients))
    assert any((torch.count_nonzero(g).item() for g in gradients)), 'No nonzero gradient'
    optimizer.step()
    changed = sum((not torch.equal(before[n], p.detach()) for n, p in model.named_parameters()))
    assert changed > 0, 'Optimizer did not update parameters'
    model.eval()
    with torch.inference_mode():
        expected = flatten_tensors(model(*inputs))
    with tempfile.TemporaryDirectory(prefix='alignseg_smoke_') as temporary:
        checkpoint = Path(temporary) / 'test_state.pt'
        torch.save(model.state_dict(), checkpoint)
        restored = factory().to(inputs[0].device).eval()
        restored.load_state_dict(torch.load(checkpoint, map_location=inputs[0].device, weights_only=True))
        with torch.inference_mode():
            actual = flatten_tensors(restored(*inputs))
        assert len(actual) == len(expected)
        max_error = max(((a - b).abs().max().item() for a, b in zip(actual, expected)))
        assert max_error == 0, f'Reload changed outputs: {max_error}'
    return {'status': 'PASS', 'loss': loss.item(), 'updated_parameter_tensors': changed, 'reload_max_abs_error': max_error, 'output_shapes': [list(t.shape) for t in expected]}

def make_alignseg():
    model = AlignedMultiViewResUNet2D(out_channels=3, base_channels=8)
    replace_batch_norm_with_group_norm(model)
    return model

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.device == 'cuda' and (not torch.cuda.is_available()):
        parser.error('CUDA is unavailable; use --device cpu or install a compatible CUDA build')
    torch.set_num_threads(4)
    torch.manual_seed(2026)
    device = torch.device(args.device)
    images = torch.rand(2, 5, 64, 64, device=device, dtype=torch.float32)
    targets = torch.randint(0, 2, (2, 3, 64, 64), device=device).float()
    seg = update_and_reload(make_alignseg().to(device), (images,), lambda logits: F.binary_cross_entropy_with_logits(logits, targets), make_alignseg)
    crops = torch.rand(2, 3, 1, 64, 64, device=device)
    geometry = torch.zeros(2, 32, device=device)
    labels = {name: torch.tensor([0, len(spec.classes) - 1], device=device) for name, spec in TARGET_SPECS.items()}

    def task_loss(output):
        return sum((F.cross_entropy(output['logits'][n], y) + 0.25 * F.cross_entropy(output['image_logits'][n], y) for n, y in labels.items()))
    factory = lambda: MultiTaskR1(32, 16, 96, 128, 0.3)
    geo = update_and_reload(factory().to(device), (crops, geometry), task_loss, factory)
    report = {'scope': 'generated_tensor_software_smoke_only', 'python': platform.python_version(), 'torch': torch.__version__, 'device': args.device, 'alignseg': seg, 'geofuse': geo}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report))
if __name__ == '__main__':
    main()
