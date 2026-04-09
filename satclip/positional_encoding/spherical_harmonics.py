import math

import torch
from torch import nn
from .spherical_harmonics_closed_form import SH as SH_closed_form

"""
Spherical Harmonics locaiton encoder
"""
POLE_LATITUDE_EPS_DEGREES = 0.05


class SphericalHarmonics(nn.Module):
    def __init__(
        self,
        legendre_polys: int = 10,
        harmonics_calculation="analytic",
        compute_dtype=torch.float64,
        output_dtype=None,
    ):
        """
        legendre_polys: determines the number of legendre polynomials.
                        more polynomials lead more fine-grained resolutions
        calculation of spherical harmonics:
            analytic uses pre-computed equations. This is exact, but works only up to degree 50,
            closed-form uses one equation but is computationally slower (especially for high degrees)
        """
        super(SphericalHarmonics, self).__init__()
        self.L = int(legendre_polys)
        self.embedding_dim = self.L * self.L
        self.compute_dtype = compute_dtype
        self.output_dtype = output_dtype
        self.harmonics_calculation = harmonics_calculation
        self._init_analytic_constants()

        if harmonics_calculation == "closed-form":
            return
        if harmonics_calculation != "analytic":
            raise ValueError(
                "harmonics_calculation must be either 'analytic' or 'closed-form'"
            )

    def _init_analytic_constants(self):
        orders = torch.arange(self.L, dtype=torch.int64)
        center_indices = orders * orders + orders

        diag_coeffs = torch.ones(self.L, dtype=torch.float64)
        if self.L > 1:
            diag_terms = torch.sqrt(1.0 + 1.0 / (2.0 * orders[1:].to(torch.float64)))
            diag_coeffs[1:] = torch.cumprod(diag_terms, dim=0)

        subdiag_coeffs = torch.sqrt(2.0 * orders[:-1].to(torch.float64) + 3.0)
        alpha = torch.zeros((self.L, self.L), dtype=torch.float64)
        beta = torch.zeros((self.L, self.L), dtype=torch.float64)
        flat_degree = []
        flat_m = []
        neg_index = []
        pos_index = []

        for degree in range(2, self.L):
            m = torch.arange(degree - 1, dtype=torch.float64)
            alpha[degree, : degree - 1] = torch.sqrt(
                ((2.0 * degree + 1.0) / (2.0 * degree - 3.0))
                * ((4.0 * (degree - 1) * (degree - 1) - 1.0) / (degree * degree - m * m))
            )
            beta[degree, : degree - 1] = torch.sqrt(
                ((2.0 * degree + 1.0) / (2.0 * degree - 3.0))
                * ((((degree - 1) * (degree - 1)) - m * m) / (degree * degree - m * m))
            )

        for degree in range(self.L):
            center_index = degree * degree + degree
            for m in range(1, degree + 1):
                flat_degree.append(degree)
                flat_m.append(m)
                neg_index.append(center_index - m)
                pos_index.append(center_index + m)

        self.register_buffer("orders", orders, persistent=False)
        self.register_buffer("center_indices", center_indices, persistent=False)
        self.register_buffer("diag_coeffs", diag_coeffs, persistent=False)
        self.register_buffer("subdiag_coeffs", subdiag_coeffs, persistent=False)
        self.register_buffer("alpha", alpha, persistent=False)
        self.register_buffer("beta", beta, persistent=False)
        self.register_buffer(
            "flat_degree",
            torch.tensor(flat_degree, dtype=torch.int64),
            persistent=False,
        )
        self.register_buffer(
            "flat_m",
            torch.tensor(flat_m, dtype=torch.int64),
            persistent=False,
        )
        self.register_buffer(
            "negative_output_indices",
            torch.tensor(neg_index, dtype=torch.int64),
            persistent=False,
        )
        self.register_buffer(
            "positive_output_indices",
            torch.tensor(pos_index, dtype=torch.int64),
            persistent=False,
        )

    def _forward_stable_analytic(self, lonlat):
        lon = lonlat[:, 0]
        lat = lonlat[:, 1].clamp(
            min=-90.0 + POLE_LATITUDE_EPS_DEGREES,
            max=90.0 - POLE_LATITUDE_EPS_DEGREES,
        )

        phi = torch.deg2rad(lon + 180)
        theta = torch.deg2rad(lat + 90)
        x = torch.cos(theta)
        u = torch.sqrt(torch.clamp(1 - x * x, min=0))

        dtype = lonlat.dtype
        device = lonlat.device
        batch_size = lonlat.shape[0]
        orders = self.orders.to(device=device)
        order_values = orders.to(dtype=dtype)

        p00 = lonlat.new_full((batch_size,), math.sqrt(1.0 / (4.0 * math.pi)))
        diag_terms = [p00]
        if self.L > 1:
            diag_coeffs = self.diag_coeffs.to(device=device, dtype=dtype)
            diag_values = (
                p00.unsqueeze(1)
                * diag_coeffs[1:].unsqueeze(0)
                * u.unsqueeze(1).pow(order_values[1:].unsqueeze(0))
            )
            diag_terms.extend(diag_values.unbind(dim=1))

        subdiag_coeffs = self.subdiag_coeffs.to(device=device, dtype=dtype)
        alpha = self.alpha.to(device=device, dtype=dtype)
        beta = self.beta.to(device=device, dtype=dtype)
        rows = [
            torch.cat(
                [diag_terms[0].unsqueeze(1), lonlat.new_zeros((batch_size, self.L - 1))],
                dim=1,
            )
        ]
        if self.L > 1:
            rows.append(
                torch.cat(
                    [
                        (subdiag_coeffs[0] * x * diag_terms[0]).unsqueeze(1),
                        diag_terms[1].unsqueeze(1),
                        lonlat.new_zeros((batch_size, self.L - 2)),
                    ],
                    dim=1,
                )
            )

        for degree in range(2, self.L):
            recurrence = (
                alpha[degree, : degree - 1].unsqueeze(0)
                * x.unsqueeze(1)
                * rows[degree - 1][:, : degree - 1]
                - beta[degree, : degree - 1].unsqueeze(0) * rows[degree - 2][:, : degree - 1]
            )
            row_terms = [
                recurrence,
                (subdiag_coeffs[degree - 1] * x * diag_terms[degree - 1]).unsqueeze(1),
                diag_terms[degree].unsqueeze(1),
            ]
            if degree < self.L - 1:
                row_terms.append(lonlat.new_zeros((batch_size, self.L - degree - 1)))
            rows.append(torch.cat(row_terms, dim=1))

        plm = torch.stack(rows, dim=1)
        phases = phi.unsqueeze(1) * order_values.unsqueeze(0)
        sin_terms = torch.sin(phases)
        cos_terms = torch.cos(phases)
        degree_blocks = []
        for degree in range(self.L):
            center = (math.pi * plm[:, degree, 0]).unsqueeze(1)
            if degree == 0:
                degree_blocks.append(center)
                continue
            m = orders[1 : degree + 1]
            base = math.sqrt(2.0) * plm[:, degree, 1 : degree + 1]
            negative = torch.flip(base * sin_terms[:, m], dims=(1,))
            positive = base * cos_terms[:, m]
            degree_blocks.append(torch.cat([negative, center, positive], dim=1))

        return torch.cat(degree_blocks, dim=1)

    def forward(self, lonlat):
        work_dtype = self.compute_dtype or lonlat.dtype
        lonlat = lonlat.to(dtype=work_dtype)

        if self.harmonics_calculation == "analytic":
            output = self._forward_stable_analytic(lonlat)
        else:
            lon, lat = lonlat[:, 0], lonlat[:, 1]
            phi = torch.deg2rad(lon + 180)
            theta = torch.deg2rad(lat + 90)

            Y = []
            for l in range(self.L):
                for m in range(-l, l + 1):
                    y = SH_closed_form(m, l, phi, theta)
                    if isinstance(y, float):
                        y = y * torch.ones_like(phi)
                    Y.append(y)

            output = torch.stack(Y, dim=-1)

        if self.output_dtype is not None:
            output = output.to(dtype=self.output_dtype)
        return output
