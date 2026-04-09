import math

import torch
from torch import nn
import torch.nn.functional as F

from .positional_encoding.spherical_harmonics import SphericalHarmonics


LOCATION_ENCODER_METADATA_KEYS = (
    "capacity",
    "embed_dim",
    "harmonics_calculation",
    "legendre_polys",
    "le_type",
    "num_hidden_layers",
    "pe_type",
)
LOW_PRECISION_DTYPES = (torch.float16, torch.float32, torch.bfloat16)

def _extract_location_encoder_metadata(hparams):
    metadata = {key: hparams[key] for key in LOCATION_ENCODER_METADATA_KEYS}
    if metadata["le_type"] != "sphericalharmonics":
        raise NotImplementedError(
            "location-only extraction currently supports le_type='sphericalharmonics' only"
        )
    if metadata["pe_type"] != "siren":
        raise NotImplementedError(
            "location-only extraction currently supports pe_type='siren' only"
        )
    return metadata


def _extract_location_encoder_state_dict(state_dict):
    location_state_dict = {
        key.split("nnet.", 1)[1]: value for key, value in state_dict.items() if "nnet." in key
    }
    if not location_state_dict:
        raise KeyError("could not find any location encoder weights under an 'nnet.' prefix")
    return location_state_dict


def _load_location_encoder_payload(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if checkpoint.get("format") == "satclip-location-only":
        return checkpoint["metadata"], checkpoint["state_dict"]
    return (
        _extract_location_encoder_metadata(checkpoint["hyper_parameters"]),
        _extract_location_encoder_state_dict(checkpoint["state_dict"]),
    )


def _resolve_posenc_compute_dtype(harmonics_calculation, dtype, posenc_compute_dtype):
    if posenc_compute_dtype is not None:
        return posenc_compute_dtype
    if harmonics_calculation == "analytic" and dtype in LOW_PRECISION_DTYPES:
        return torch.float32
    if dtype in (torch.float16, torch.bfloat16):
        return torch.float32
    return dtype


def _normalize_nnet_state_dict(state_dict):
    if any(key.startswith("nnet.") for key in state_dict):
        return state_dict
    return {f"nnet.{key}": value for key, value in state_dict.items()}


class Sine(nn.Module):
    def __init__(self, w0=1.0):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)


class Siren(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        w0=1.0,
        c=6.0,
        is_first=False,
        use_bias=True,
        activation=None,
        dropout=False,
    ):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.is_first = is_first
        self.dropout = dropout

        weight = torch.zeros(dim_out, dim_in)
        bias = torch.zeros(dim_out) if use_bias else None
        self._init_parameters(weight, bias, c=c, w0=w0)

        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias) if use_bias else None
        self.activation = Sine(w0) if activation is None else activation

    def _init_parameters(self, weight, bias, c, w0):
        scale = (1 / self.dim_in) if self.is_first else (math.sqrt(c / self.dim_in) / w0)
        weight.uniform_(-scale, scale)
        if bias is not None:
            bias.uniform_(-scale, scale)

    def forward(self, x):
        out = F.linear(x, self.weight, self.bias)
        if self.dropout:
            out = F.dropout(out, training=self.training)
        return self.activation(out)


class SirenNet(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_hidden,
        dim_out,
        num_layers,
        w0=1.0,
        w0_initial=30.0,
        use_bias=True,
        final_activation=None,
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        for index in range(num_layers):
            is_first = index == 0
            self.layers.append(
                Siren(
                    dim_in=dim_in if is_first else dim_hidden,
                    dim_out=dim_hidden,
                    w0=w0_initial if is_first else w0,
                    use_bias=use_bias,
                    is_first=is_first,
                    dropout=True,
                )
            )
        self.last_layer = Siren(
            dim_in=dim_hidden,
            dim_out=dim_out,
            w0=w0,
            use_bias=use_bias,
            activation=nn.Identity() if final_activation is None else final_activation,
            dropout=False,
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.last_layer(x)


class LocationEncoder(nn.Module):
    def __init__(self, posenc, nnet, output_dtype=None):
        super().__init__()
        self.posenc = posenc
        self.nnet = nnet
        self.output_dtype = output_dtype

    @property
    def device(self):
        return next(self.nnet.parameters()).device

    @property
    def nnet_dtype(self):
        return next(self.nnet.parameters()).dtype

    def _forward_chunk(self, coords):
        coords = coords.to(device=self.device)
        encoded = self.posenc(coords)
        encoded = encoded.to(device=self.device, dtype=self.nnet_dtype)
        output = self.nnet(encoded)
        if self.output_dtype is not None and output.dtype != self.output_dtype:
            output = output.to(dtype=self.output_dtype)
        return output

    def forward(self, coords, chunk_size=None):
        if chunk_size is None:
            return self._forward_chunk(coords)
        return torch.cat([self._forward_chunk(chunk) for chunk in coords.split(chunk_size)], dim=0)


def build_location_encoder(
    metadata,
    nnet_dtype=torch.float64,
    posenc_compute_dtype=torch.float64,
    output_dtype=None,
):
    posenc = SphericalHarmonics(
        legendre_polys=metadata["legendre_polys"],
        harmonics_calculation=metadata["harmonics_calculation"],
        compute_dtype=posenc_compute_dtype,
        output_dtype=nnet_dtype,
    )
    nnet = SirenNet(
        dim_in=posenc.embedding_dim,
        dim_hidden=metadata["capacity"],
        dim_out=metadata["embed_dim"],
        num_layers=metadata["num_hidden_layers"],
    )
    return LocationEncoder(posenc, nnet, output_dtype=output_dtype or nnet_dtype).to(
        dtype=nnet_dtype
    ).eval()


def load_location_encoder_checkpoint(
    checkpoint_path,
    device="cpu",
    dtype=torch.float64,
    posenc_compute_dtype=None,
):
    metadata, state_dict = _load_location_encoder_payload(checkpoint_path)
    posenc_compute_dtype = _resolve_posenc_compute_dtype(
        metadata["harmonics_calculation"], dtype, posenc_compute_dtype
    )

    model = build_location_encoder(
        metadata,
        nnet_dtype=dtype,
        posenc_compute_dtype=posenc_compute_dtype,
        output_dtype=dtype,
    )
    model.load_state_dict(_normalize_nnet_state_dict(state_dict))
    model = model.to(device=device)
    model.eval()
    return model


def export_location_encoder_checkpoint(source_checkpoint_path, output_checkpoint_path):
    metadata, state_dict = _load_location_encoder_payload(source_checkpoint_path)

    torch.save(
        {
            "format": "satclip-location-only",
            "metadata": metadata,
            "state_dict": state_dict,
        },
        output_checkpoint_path,
    )
