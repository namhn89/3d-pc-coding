import warnings
from typing import Any
from typing import Callable
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchac
from torch import Tensor


def lower_bound_fwd(x: Tensor, bound: Tensor) -> Tensor:
    return torch.max(x, bound)


def lower_bound_bwd(x: Tensor, bound: Tensor, grad_output: Tensor):
    pass_through_if = (x >= bound) | (grad_output < 0)
    return pass_through_if * grad_output, None


class RoundNoGradientFunction(torch.autograd.Function):
    """Autograd function for the "Round" operator."""
    @staticmethod
    def forward(ctx, x):
        return x.round()

    @staticmethod
    def backward(ctx, g):
        return g


class LowerBoundFunction(torch.autograd.Function):
    """Autograd function for the `LowerBound` operator."""

    @staticmethod
    def forward(ctx, x, bound):
        ctx.save_for_backward(x, bound)
        return lower_bound_fwd(x, bound)

    @staticmethod
    def backward(ctx, grad_output):
        x, bound = ctx.saved_tensors
        return lower_bound_bwd(x, bound, grad_output)


class LowerBound(nn.Module):
    """Lower bound operator, computes `torch.max(x, bound)` with a custom
    gradient.
    The derivative is replaced by the identity function when `x` is moved
    towards the `bound`, otherwise the gradient is kept to zero.
    """

    bound: Tensor

    def __init__(self, bound: float):
        super().__init__()
        self.register_buffer("bound", torch.Tensor([float(bound)]))

    @torch.jit.unused
    def lower_bound(self, x):
        return LowerBoundFunction.apply(x, self.bound)

    def forward(self, x):
        if torch.jit.is_scripting():
            return torch.max(x, self.bound)
        return self.lower_bound(x)


class EntropyBottleneck(nn.Module):
    """Entropy bottleneck layer, introduced by J. Ballé, D. Minnen, S. Singh,
    S. J. Hwang, N. Johnston, in `"Variational image compression with a scale
    hyperprior" <https://arxiv.org/abs/1802.01436>`_.
    """
    def __init__(
        self,
        channels: int,
        likelihood_bound: float = 1e-9,
        tail_mass: float = 1e-9,
        init_scale: float = 8,
        filters: Tuple[int, ...] = (3, 3, 3),
    ):
        super().__init__()
        self.likelihood_bound = likelihood_bound
        self.use_likelihood_bound = likelihood_bound > 0
        if self.use_likelihood_bound:
            self.likelihood_lower_bound = LowerBound(likelihood_bound)

        self.init_scale = float(init_scale)
        self.tail_mass = float(tail_mass)
        self.channels = int(channels)
        self.filters = tuple(int(f) for f in filters)

        # Create parameters
        filters = (1,) + self.filters + (1,)
        scale = self.init_scale ** (1 / (len(self.filters) + 1))
        channels = self.channels

        for i in range(len(self.filters) + 1):
            init = np.log(np.expm1(1 / scale / filters[i + 1]))
            matrix = torch.Tensor(channels, filters[i + 1], filters[i])
            matrix.data.fill_(init)
            self.register_parameter(f"_matrix{i:d}", nn.Parameter(matrix))

            bias = torch.Tensor(channels, filters[i + 1], 1)
            # nn.init.uniform_(bias, -0.5, 0.5)
            bias = torch.FloatTensor(np.random.uniform(-0.5, 0.5, bias.size()))
            self.register_parameter(f"_bias{i:d}", nn.Parameter(bias))

            if i < len(self.filters):
                factor = torch.Tensor(channels, filters[i + 1], 1)
                nn.init.zeros_(factor)
                self.register_parameter(f"_factor{i:d}", nn.Parameter(factor))

        self.quantiles = nn.Parameter(torch.Tensor(channels, 1, 3))
        init = torch.Tensor([-self.init_scale, 0, self.init_scale])
        self.quantiles.data = init.repeat(self.quantiles.size(0), 1, 1)

        target = np.log(2 / self.tail_mass - 1)
        self.register_buffer("target", torch.Tensor([-target, 0, target]))

    def _logits_cumulative(self, inputs: Tensor, stop_gradient: Optional[bool] = False) -> Tensor:
        logits = inputs
        for i in range(len(self.filters) + 1):
            matrix = getattr(self, f"_matrix{i:d}")
            if stop_gradient:
                matrix = matrix.detach()
            logits = torch.matmul(F.softplus(matrix), logits)

            bias = getattr(self, f"_bias{i:d}")
            if stop_gradient:
                bias = bias.detach()
            logits += bias

            if i < len(self.filters):
                factor = getattr(self, f"_factor{i:d}")
                if stop_gradient:
                    factor = factor.detach()
                logits += torch.tanh(factor) * torch.tanh(logits)
        return logits

    def _quantize(self, inputs: Tensor, mode: str) -> Tensor:
        if mode not in ("noise", "symbols"):
            raise ValueError(f'Invalid quantization mode: "{mode}"')

        if mode == "noise":
            half = float(0.5)
            noise = torch.empty_like(inputs).uniform_(-half, half)
            inputs = inputs + noise
            return inputs

        if mode == "symbols":
            outputs = RoundNoGradientFunction.apply(inputs)
            return outputs

    def _likelihood(self, inputs: Tensor) -> Tensor:
        """Estimate the likelihood.
        """
        perm = np.arange(len(inputs.shape))
        perm[0], perm[1] = perm[1], perm[0]
        inv_perm = np.arange(len(inputs.shape))[np.argsort(perm)]

        inputs = inputs.permute(*perm).contiguous()
        shape = inputs.size()
        inputs = inputs.reshape(inputs.size(0), 1, -1)

        half = float(0.5)
        v0 = inputs - half
        v1 = inputs + half
        lower = self._logits_cumulative(v0)
        upper = self._logits_cumulative(v1)
        sign = -torch.sign(lower + upper)
        sign = sign.detach()
        likelihood = torch.abs(
            torch.sigmoid(sign * upper) - torch.sigmoid(sign * lower)
        )
        likelihood = likelihood.reshape(shape)
        likelihood = likelihood.permute(*inv_perm).contiguous()

        return likelihood

    def forward(
        self, x: Tensor, training: Optional[bool] = None
    ) -> Tuple[Tensor, Tensor]:
        if training is None:
            training = self.training

        outputs = self._quantize(
            x, mode="noise" if training else "symbols"
        )

        likelihood = self._likelihood(outputs)
        if self.use_likelihood_bound:
            likelihood = self.likelihood_lower_bound(likelihood)

        return outputs, likelihood

    def _pmf_to_cdf(self, pmf: Tensor) -> Tensor:
        cdf = pmf.cumsum(dim=-1)
        spatial_dimensions = pmf.shape[:-1] + (1,)
        zeros = torch.zeros(spatial_dimensions, dtype=pmf.dtype, device=pmf.device)
        cdf_with_0 = torch.cat([zeros, cdf], dim=-1)
        cdf_with_0 = cdf_with_0.clamp(max=1.)

        return cdf_with_0

    @torch.no_grad()
    def compress(self, inputs):
        inputs = inputs.permute(0, 2, 3, 4, 1)

        # quantize
        values = self._quantize(inputs, mode="symbols")

        # get symbols
        min_v = values.min().detach().float()
        max_v = values.max().detach().float()
        symbols = torch.arange(min_v, max_v + 1)
        symbols = symbols.reshape(1, 1, -1).repeat(values.shape[-1], 1, 1).float()  # (channels, 1, num_symbols)
        # Get normalized values
        values_norm = values - min_v
        min_v, max_v = torch.tensor([min_v]), torch.tensor([max_v])
        values_norm = values_norm.to(torch.int16)

        # Get pmf
        lower = self._logits_cumulative(symbols - 0.5)
        upper = self._logits_cumulative(symbols + 0.5)
        sign = -torch.sign(torch.add(lower, upper))
        likelihood = torch.abs(torch.sigmoid(sign * upper) - torch.sigmoid(sign * lower))
        pmf = torch.clamp(likelihood, min=self.likelihood_bound)  # (channels, 1, num_symbols)
        pmf = pmf.reshape(values.shape[-1], -1)
        cdf = self._pmf_to_cdf(pmf)

        # arithmetic encoding
        values_norm = values_norm.reshape(-1, values.shape[-1])
        out_cdf = cdf.unsqueeze(0).repeat(values_norm.shape[0], 1, 1).detach().cpu()
        strings = torchac.encode_float_cdf(out_cdf, values_norm.cpu(), check_input_bounds=True)

        return strings, min_v.cpu().numpy(), max_v.cpu().numpy()

    @torch.no_grad()
    def decompress(self, strings, min_v, max_v, shape):
        # Get symbols
        symbols = torch.arange(min_v, max_v + 1)
        channels = int(shape[1])
        symbols = symbols.reshape(1, 1, -1).repeat(channels, 1, 1).float()

        # Get pmf
        lower = self._logits_cumulative(symbols - 0.5)
        upper = self._logits_cumulative(symbols + 0.5)
        sign = -torch.sign(torch.add(lower, upper))
        likelihood = torch.abs(torch.sigmoid(sign * upper) - torch.sigmoid(sign * lower))
        pmf = torch.clamp(likelihood, min=self.likelihood_bound)  # (channels, 1, num_symbols)
        pmf = pmf.reshape(channels, -1)

        # Get cdf
        cdf = self._pmf_to_cdf(pmf)

        # Arithmetic decoding
        out_cdf = cdf.unsqueeze(0).repeat(torch.prod(torch.tensor(shape)).item() // channels, 1, 1).cpu()
        values = torchac.decode_float_cdf(out_cdf, strings)
        values = values.float()
        values += min_v
        values = torch.reshape(values, (shape[0], shape[2], shape[3], shape[4], -1))
        values = values.permute(0, 4, 1, 2, 3)

        return values
