from pathlib import Path
import argparse
import importlib
import json
import sys
from unittest.mock import patch
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
import torch
from nets import AlignedMultiViewResUNet2D, replace_batch_norm_with_group_norm
from knosp_r_experiments import GEOMETRY_FEATURES, GeometryPreprocessor, case_geometry_and_quality
from mva_multitask_s2r1 import MultiTaskR1, TARGET_SPECS

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(2026)
    report = {'scope': 'synthetic_interface_only', 'tests': {}}
    for path in sorted((ROOT / 'src').glob('*.py')):
        importlib.import_module(path.stem)
    report['tests']['all_source_imports'] = 'PASS'
    aligned = AlignedMultiViewResUNet2D(out_channels=3, base_channels=8).eval()
    no_warp = AlignedMultiViewResUNet2D(out_channels=3, base_channels=8, enable_alignment=False).eval()
    replace_batch_norm_with_group_norm(aligned)
    replace_batch_norm_with_group_norm(no_warp)
    no_warp.load_state_dict(aligned.state_dict(), strict=True)
    assert {k: v.shape for k, v in aligned.state_dict().items()} == {k: v.shape for k, v in no_warp.state_dict().items()}
    assert [f.max_offset for f in aligned.alignment_fusions] == [4, 2, 1, 0.5, 0.25]
    x = torch.randn(1, 5, 64, 64)
    for name, model, expected in [('aligned', aligned, 10), ('no_warp', no_warp, 0)]:
        called = {'sag': 0, 'ax': 0, 'film': 0}

        def hook(key):

            def record(*_):
                called[key] += 1
            return record
        handles = [model.b_encoder.register_forward_hook(hook('sag')), model.v_encoder.register_forward_hook(hook('ax'))]
        handles += [m.register_forward_hook(hook('film')) for m in model.context_gates]
        with torch.inference_mode(), patch('torch.nn.functional.grid_sample', wraps=torch.nn.functional.grid_sample) as warp:
            y = model(x)
        assert tuple(y.shape) == (1, 3, 64, 64) and y.dtype == torch.float32 and torch.isfinite(y).all()
        assert warp.call_count == expected and called == {'sag': 1, 'ax': 1, 'film': 5}
        for h in handles:
            h.remove()
        report['tests'][name] = {'status': 'PASS', 'warp_calls': warp.call_count, 'hooks': called}
    names = list(GEOMETRY_FEATURES)
    pre = GeometryPreprocessor(names, {'pitnet': {'medians': [0.0] * 16, 'means': [0.0] * 16, 'stds': [1.0] * 16}})
    grid_y, grid_x = np.ogrid[:64, :64]
    tumor = (grid_x - 32) ** 2 + (grid_y - 32) ** 2 < 12 ** 2
    right = (grid_x - 16) ** 2 + (grid_y - 32) ** 2 < 5 ** 2
    left = (grid_x - 48) ** 2 + (grid_y - 32) ** 2 < 5 ** 2
    empty = np.zeros_like(tumor)
    cases = {'normal': (tumor, right, left), 'empty_pa': (empty, right, left), 'missing_ica': (tumor, empty, left), 'coincident_ica': (tumor, right, right), 'overlap': (tumor | right, right, left)}
    with patch.object(GeometryPreprocessor, 'fit', side_effect=AssertionError('Inference refit prohibited')) as fit:
        for name, (t, r, l) in cases.items():
            features = case_geometry_and_quality({'m': {'tumor': t, 'right_ica': r, 'left_ica': l}})
            vec = pre.transform(features, 'pitnet')
            assert vec.shape == (32,) and np.isfinite(vec).all()
            assert np.array_equal(vec, GeometryPreprocessor.from_dict(json.loads(json.dumps(pre.to_dict()))).transform(features, 'pitnet'))
            assert set(np.unique(vec[16:])).issubset({0.0, 1.0})
            if name in ('empty_pa', 'missing_ica', 'coincident_ica'):
                assert np.all(vec[16:] == 1)
        assert fit.call_count == 0
    report['tests']['geometry_boundaries_and_no_refit'] = {'status': 'PASS', 'cases': list(cases), 'fit_calls': 0}
    geo = MultiTaskR1(32, 16, 96, 128, 0.3).eval()
    with torch.inference_mode():
        out = geo(torch.rand(1, 3, 1, 64, 64), torch.zeros(1, 32))
    assert geo.fusion_encoder[0].in_features == 224
    for target, spec in TARGET_SPECS.items():
        assert out['logits'][target].shape == (1, len(spec.classes))
        assert out['image_logits'][target].shape == (1, len(spec.classes))
    report['tests']['geofuse_dimensions_and_heads'] = 'PASS'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report), flush=True)
if __name__ == '__main__':
    main()
