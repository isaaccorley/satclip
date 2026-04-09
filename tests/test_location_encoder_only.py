import pytest
import torch
import satclip.positional_encoding.spherical_harmonics as sh_mod
from satclip.positional_encoding.spherical_harmonics_ylm import SH as legacy_sh_analytic

from satclip.location_encoder_only import (
    SphericalHarmonics,
    build_location_encoder,
    export_location_encoder_checkpoint,
    load_location_encoder_checkpoint,
)


def _metadata():
    return {
        "capacity": 32,
        "embed_dim": 8,
        "harmonics_calculation": "closed-form",
        "legendre_polys": 4,
        "le_type": "sphericalharmonics",
        "num_hidden_layers": 2,
        "pe_type": "siren",
    }


def _coordinate_suite(dtype):
    return torch.tensor(
        [
            [-87.6298, 41.8781],
            [151.2093, -33.8688],
            [2.3522, 48.8566],
            [0.0, 0.0],
            [179.999, 89.999],
            [-179.999, -89.999],
            [179.999, 0.001],
            [-179.999, -0.001],
            [77.5946, 12.9716],
            [-58.3816, -34.6037],
            [139.6917, 35.6895],
            [37.6173, 55.7558],
        ],
        dtype=dtype,
    )


def _analytic_metadata():
    return {
        "capacity": 32,
        "embed_dim": 8,
        "harmonics_calculation": "analytic",
        "legendre_polys": 40,
        "le_type": "sphericalharmonics",
        "num_hidden_layers": 2,
        "pe_type": "siren",
    }


def test_location_only_loader_round_trip(tmp_path):
    metadata = _metadata()
    reference_model = build_location_encoder(metadata)

    full_checkpoint = tmp_path / "full.ckpt"
    compact_checkpoint = tmp_path / "location-only.ckpt"

    torch.save(
        {
            "hyper_parameters": metadata,
            "state_dict": {
                f"model.nnet.{key}": value.clone()
                for key, value in reference_model.nnet.state_dict().items()
            },
        },
        full_checkpoint,
    )

    export_location_encoder_checkpoint(full_checkpoint, compact_checkpoint)

    loaded_from_full = load_location_encoder_checkpoint(full_checkpoint)
    loaded_from_compact = load_location_encoder_checkpoint(compact_checkpoint)

    coords = torch.tensor(
        [[-87.6298, 41.8781], [151.2093, -33.8688]],
        dtype=torch.float64,
    )

    expected = reference_model(coords)
    full_output = loaded_from_full(coords)
    compact_output = loaded_from_compact(coords)

    torch.testing.assert_close(full_output, expected)
    torch.testing.assert_close(compact_output, expected)


def test_spherical_harmonics_hybrid_precision_avoids_float32_nans():
    coords32 = _coordinate_suite(torch.float32)
    coords64 = coords32.to(torch.float64)

    float32_posenc = SphericalHarmonics(
        legendre_polys=40,
        harmonics_calculation="analytic",
        compute_dtype=torch.float32,
        output_dtype=torch.float32,
    )
    hybrid_posenc = SphericalHarmonics(
        legendre_polys=40,
        harmonics_calculation="analytic",
        compute_dtype=torch.float64,
        output_dtype=torch.float32,
    )
    reference_posenc = SphericalHarmonics(
        legendre_polys=40,
        harmonics_calculation="analytic",
        compute_dtype=torch.float64,
        output_dtype=torch.float64,
    )

    float32_output = float32_posenc(coords32)
    hybrid_output = hybrid_posenc(coords32)
    reference_output = reference_posenc(coords64).to(torch.float32)

    assert not torch.isnan(float32_output).any()
    assert not torch.isinf(float32_output).any()
    assert not torch.isnan(hybrid_output).any()
    assert not torch.isinf(hybrid_output).any()
    torch.testing.assert_close(hybrid_output, reference_output, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(float32_output, reference_output, rtol=1e-3, atol=2e-3)


def test_stable_analytic_tracks_legacy_generated_basis_away_from_poles():
    coords = _coordinate_suite(torch.float64)[:4]
    posenc = SphericalHarmonics(
        legendre_polys=40,
        harmonics_calculation="analytic",
        compute_dtype=torch.float64,
        output_dtype=torch.float64,
    )

    lon = coords[:, 0]
    lat = coords[:, 1]
    phi = torch.deg2rad(lon + 180)
    theta = torch.deg2rad(lat + 90)

    legacy_terms = []
    for degree in range(40):
        for order in range(-degree, degree + 1):
            value = legacy_sh_analytic(order, degree, phi, theta)
            if isinstance(value, float):
                value = value * torch.ones_like(phi)
            legacy_terms.append(value)
    legacy_output = torch.stack(legacy_terms, dim=-1)
    stable_output = posenc(coords)

    torch.testing.assert_close(stable_output, legacy_output, rtol=1e-6, atol=3e-4)


def test_chunked_forward_matches_unchunked():
    metadata = _metadata()
    model = build_location_encoder(metadata)
    coords = torch.tensor(
        [
            [-87.6298, 41.8781],
            [151.2093, -33.8688],
            [2.3522, 48.8566],
            [0.0, 0.0],
        ],
        dtype=torch.float64,
    )

    full_output = model(coords)
    chunked_output = model(coords, chunk_size=2)

    torch.testing.assert_close(chunked_output, full_output)


def test_stable_analytic_backward_is_finite_away_from_poles():
    posenc = SphericalHarmonics(
        legendre_polys=40,
        harmonics_calculation="analytic",
        compute_dtype=torch.float32,
        output_dtype=torch.float32,
    )
    coords = torch.tensor(
        [
            [-87.6298, 41.8781],
            [151.2093, -33.8688],
            [0.0, 0.0],
            [179.5, -0.25],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )

    output = posenc(coords)
    loss = output.square().mean() + 0.1 * output.abs().mean()
    loss.backward()

    assert torch.isfinite(output).all()
    assert torch.isfinite(coords.grad).all()


def test_stable_analytic_backward_is_finite_near_poles():
    posenc = SphericalHarmonics(
        legendre_polys=40,
        harmonics_calculation="analytic",
        compute_dtype=torch.float32,
        output_dtype=torch.float32,
    )
    coords = torch.tensor(
        [
            [179.999, 89.999],
            [-179.999, -89.999],
            [179.0, 89.9],
            [-179.0, -89.9],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )

    output = posenc(coords)
    loss = output.square().mean() + 0.1 * output.abs().mean()
    loss.backward()

    assert torch.isfinite(output).all()
    assert torch.isfinite(coords.grad).all()


def test_stable_analytic_backward_is_finite_on_random_batches():
    posenc = SphericalHarmonics(
        legendre_polys=40,
        harmonics_calculation="analytic",
        compute_dtype=torch.float32,
        output_dtype=torch.float32,
    )

    torch.manual_seed(0)
    for batch_size in (16, 64, 128):
        coords = torch.rand(batch_size, 2, dtype=torch.float32)
        coords[:, 0] = coords[:, 0] * 360.0 - 180.0
        coords[:, 1] = coords[:, 1] * 179.8 - 89.9
        coords.requires_grad_(True)

        output = posenc(coords)
        loss = output.square().mean() + 0.1 * output.abs().mean() + 0.01 * output.mean()
        loss.backward()

        assert torch.isfinite(output).all()
        assert torch.isfinite(coords.grad).all()


def test_location_encoder_backward_is_finite_on_clamp_band():
    model = build_location_encoder(
        _analytic_metadata(),
        nnet_dtype=torch.float32,
        posenc_compute_dtype=torch.float32,
        output_dtype=torch.float32,
    )
    coords = torch.tensor(
        [
            [0.0, 89.94],
            [45.0, 89.95],
            [90.0, 89.96],
            [135.0, 89.99],
            [180.0, -89.94],
            [-45.0, -89.95],
            [-90.0, -89.96],
            [-135.0, -89.99],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )

    model.zero_grad(set_to_none=True)
    output = model(coords)
    loss = output.square().mean() + 0.1 * output.abs().mean() + 0.01 * output.mean()
    loss.backward()

    assert torch.isfinite(output).all()
    assert torch.isfinite(coords.grad).all()
    assert all(torch.isfinite(param.grad).all() for param in model.parameters() if param.grad is not None)


def test_pole_clamp_does_not_change_outputs_away_from_polar_cap():
    posenc = SphericalHarmonics(
        legendre_polys=40,
        harmonics_calculation="analytic",
        compute_dtype=torch.float32,
        output_dtype=torch.float32,
    )
    coords = torch.tensor(
        [
            [-87.6298, 41.8781],
            [151.2093, -33.8688],
            [2.3522, 48.8566],
            [0.0, 0.0],
            [179.5, -0.25],
            [-120.0, -70.0],
            [45.0, 80.0],
            [179.999, 0.001],
        ],
        dtype=torch.float32,
    )

    sh_mod.POLE_LATITUDE_EPS_DEGREES = 0.0
    unclamped = posenc(coords)
    sh_mod.POLE_LATITUDE_EPS_DEGREES = 0.05
    clamped = posenc(coords)
    sh_mod.POLE_LATITUDE_EPS_DEGREES = 0.05

    torch.testing.assert_close(clamped, unclamped, rtol=0.0, atol=0.0)


def test_pole_clamp_only_perturbs_outputs_slightly_near_poles():
    posenc = SphericalHarmonics(
        legendre_polys=40,
        harmonics_calculation="analytic",
        compute_dtype=torch.float32,
        output_dtype=torch.float32,
    )
    coords = torch.tensor(
        [
            [0.0, 89.94],
            [45.0, 89.95],
            [90.0, 89.96],
            [135.0, 89.99],
            [180.0, -89.94],
            [-45.0, -89.95],
            [-90.0, -89.96],
            [-135.0, -89.99],
        ],
        dtype=torch.float32,
    )

    sh_mod.POLE_LATITUDE_EPS_DEGREES = 0.0
    unclamped = posenc(coords)
    sh_mod.POLE_LATITUDE_EPS_DEGREES = 0.05
    clamped = posenc(coords)
    sh_mod.POLE_LATITUDE_EPS_DEGREES = 0.05

    diff = (clamped - unclamped).abs()
    cosine = torch.nn.functional.cosine_similarity(clamped, unclamped, dim=1)
    assert float(diff.max()) < 0.1
    assert float(cosine.min()) > 0.9998


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_stable_analytic_backward_is_finite_on_mps_away_from_poles():
    posenc = SphericalHarmonics(
        legendre_polys=40,
        harmonics_calculation="analytic",
        compute_dtype=torch.float32,
        output_dtype=torch.float32,
    ).to(device="mps", dtype=torch.float32)
    coords = torch.tensor(
        [
            [-87.6298, 41.8781],
            [151.2093, -33.8688],
            [0.0, 0.0],
            [179.5, -0.25],
        ],
        dtype=torch.float32,
        device="mps",
        requires_grad=True,
    )

    output = posenc(coords)
    loss = output.square().mean() + 0.1 * output.abs().mean()
    loss.backward()

    assert torch.isfinite(output).all()
    assert torch.isfinite(coords.grad).all()


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_stable_analytic_backward_is_finite_on_mps_random_batches():
    posenc = SphericalHarmonics(
        legendre_polys=40,
        harmonics_calculation="analytic",
        compute_dtype=torch.float32,
        output_dtype=torch.float32,
    ).to(device="mps", dtype=torch.float32)

    torch.manual_seed(0)
    for batch_size in (16, 64, 128):
        coords = torch.rand(batch_size, 2, dtype=torch.float32)
        coords[:, 0] = coords[:, 0] * 360.0 - 180.0
        coords[:, 1] = coords[:, 1] * 179.8 - 89.9
        coords = coords.to(device="mps").requires_grad_(True)

        output = posenc(coords)
        loss = output.square().mean() + 0.1 * output.abs().mean() + 0.01 * output.mean()
        loss.backward()

        assert torch.isfinite(output).all()
        assert torch.isfinite(coords.grad).all()


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_stable_analytic_backward_is_finite_on_mps_near_poles():
    posenc = SphericalHarmonics(
        legendre_polys=40,
        harmonics_calculation="analytic",
        compute_dtype=torch.float32,
        output_dtype=torch.float32,
    ).to(device="mps", dtype=torch.float32)
    coords = torch.tensor(
        [
            [179.999, 89.999],
            [-179.999, -89.999],
            [179.0, 89.9],
            [-179.0, -89.9],
        ],
        dtype=torch.float32,
        device="mps",
        requires_grad=True,
    )

    output = posenc(coords)
    loss = output.square().mean() + 0.1 * output.abs().mean()
    loss.backward()

    assert torch.isfinite(output).all()
    assert torch.isfinite(coords.grad).all()


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_location_encoder_backward_matches_cpu_on_training_batches():
    cpu_model = build_location_encoder(
        _analytic_metadata(),
        nnet_dtype=torch.float32,
        posenc_compute_dtype=torch.float32,
        output_dtype=torch.float32,
    )
    mps_model = build_location_encoder(
        _analytic_metadata(),
        nnet_dtype=torch.float32,
        posenc_compute_dtype=torch.float32,
        output_dtype=torch.float32,
    ).to(device="mps", dtype=torch.float32)
    mps_model.load_state_dict(cpu_model.state_dict())

    coords_cpu = torch.tensor(
        [
            [-87.6298, 41.8781],
            [151.2093, -33.8688],
            [0.0, 0.0],
            [179.5, -0.25],
            [45.0, 89.95],
            [-45.0, -89.95],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )
    coords_mps = coords_cpu.detach().to("mps").requires_grad_(True)

    cpu_out = cpu_model(coords_cpu)
    mps_out = mps_model(coords_mps)
    cpu_loss = cpu_out.square().mean() + 0.1 * cpu_out.abs().mean() + 0.01 * cpu_out.mean()
    mps_loss = mps_out.square().mean() + 0.1 * mps_out.abs().mean() + 0.01 * mps_out.mean()
    cpu_loss.backward()
    mps_loss.backward()

    torch.testing.assert_close(mps_out.detach().cpu(), cpu_out.detach(), rtol=2e-3, atol=3e-4)
    grad_cpu = coords_cpu.grad.detach()
    grad_mps = coords_mps.grad.detach().cpu()
    grad_diff = (grad_mps - grad_cpu).abs()
    assert float(grad_diff.max()) < 5e-4
    assert (
        float(
            torch.nn.functional.cosine_similarity(
                grad_mps.reshape(-1), grad_cpu.reshape(-1), dim=0
            )
        )
        > 0.999
    )
