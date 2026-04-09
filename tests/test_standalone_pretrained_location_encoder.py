import importlib.util
from pathlib import Path

import torch

from satclip.location_encoder_only import (
    build_location_encoder,
    load_location_encoder_checkpoint,
)


def _load_standalone_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "load_pretrained_location_encoder_standalone.py"
    spec = importlib.util.spec_from_file_location("standalone_pretrained_loader", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _analytic_metadata():
    return {
        "capacity": 32,
        "embed_dim": 8,
        "harmonics_calculation": "analytic",
        "legendre_polys": 4,
        "le_type": "sphericalharmonics",
        "num_hidden_layers": 2,
        "pe_type": "siren",
    }


def test_standalone_loader_matches_package_loader(tmp_path):
    metadata = _analytic_metadata()
    reference_model = build_location_encoder(metadata)
    checkpoint_path = tmp_path / "analytic-full.ckpt"

    torch.save(
        {
            "hyper_parameters": metadata,
            "state_dict": {
                f"model.nnet.{key}": value.clone()
                for key, value in reference_model.nnet.state_dict().items()
            },
        },
        checkpoint_path,
    )

    standalone = _load_standalone_module()
    package_model = load_location_encoder_checkpoint(checkpoint_path)
    standalone_model = standalone.load_pretrained_location_encoder(checkpoint_path)

    coords = torch.tensor(
        [
            [-87.6298, 41.8781],
            [151.2093, -33.8688],
            [2.3522, 48.8566],
        ],
        dtype=torch.float64,
    )

    torch.testing.assert_close(standalone_model(coords), package_model(coords))


def test_standalone_loader_compiles_without_graph_breaks(tmp_path):
    metadata = _analytic_metadata()
    reference_model = build_location_encoder(metadata, nnet_dtype=torch.float32)
    checkpoint_path = tmp_path / "analytic-full.ckpt"

    torch.save(
        {
            "hyper_parameters": metadata,
            "state_dict": {
                f"model.nnet.{key}": value.clone()
                for key, value in reference_model.nnet.state_dict().items()
            },
        },
        checkpoint_path,
    )

    standalone = _load_standalone_module()
    model = standalone.load_pretrained_location_encoder(
        checkpoint_path,
        dtype=torch.float32,
        posenc_compute_dtype=torch.float32,
    )
    coords = torch.tensor(
        [
            [-87.6298, 41.8781],
            [151.2093, -33.8688],
            [2.3522, 48.8566],
            [0.0, 0.0],
        ],
        dtype=torch.float32,
    )

    explanation = torch._dynamo.explain(model)(coords)
    assert getattr(explanation, "graph_break_count", 0) == 0
    assert not getattr(explanation, "break_reasons", [])

    compiled_model = torch.compile(model)
    eager_output = model(coords)
    compiled_output = compiled_model(coords)
    torch.testing.assert_close(compiled_output, eager_output)
