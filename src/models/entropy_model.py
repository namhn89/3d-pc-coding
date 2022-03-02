from typing import Any
from typing import Callable
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torchac
from torch.nn.parameter import Parameter


class RoundNoGradient(torch.autograd.Function):
    """TODO: check."""

    @staticmethod
    def forward(ctx, x):
        return x.round()

    @staticmethod
    def backward(ctx, g):
        return g


class Low_bound(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        x = torch.clamp(x, min=1e-9)
        return x

    @staticmethod
    def backward(ctx, g):
        (x,) = ctx.saved_tensors
        grad1 = g.clone()
        try:
            grad1[x < 1e-9] = 0
        except RuntimeError:
            print("ERROR!")
            grad1 = g.clone()
        pass_through_if = np.logical_or(
            x.cpu().detach().numpy() >= 1e-9, g.cpu().detach().numpy() < 0.0
        )
        t = torch.Tensor(pass_through_if + 0.0).to(grad1.device)

        return grad1 * t


class EntropyBottleneck(nn.Module):
    """The layer implements a flexible probability density model to estimate
    entropy of its input tensor, which is described in this paper:
    >"Variational image compression with a scale hyperprior"
    > J. Balle, D. Minnen, S. Singh, S. J. Hwang, N. Johnston
    > https://arxiv.org/abs/1802.01436"""

    def __init__(
        self,
        channels: int,
        likelihood_bound: float = 1e-9,
        init_scale: float = 8,
        filters: Tuple[int, ...] = (3, 3, 3)
    ):

        """create parameters"""
        super(EntropyBottleneck, self).__init__()
        self._likelihood_bound = likelihood_bound
        self._init_scale = float(init_scale)
        self._filters = tuple(int(f) for f in filters)
        # self._channels = channels
        self.ASSERT = False
        # build.
        filters = (1,) + self._filters + (1,)
        scale = self._init_scale ** (1 / (len(self._filters) + 1))
        # Create variables.
        self._matrices = nn.ParameterList([])
        self._biases = nn.ParameterList([])
        self._factors = nn.ParameterList([])

        for i in range(len(self._filters) + 1):
            self.matrix = Parameter(
                torch.FloatTensor(channels, filters[i + 1], filters[i])
            )
            init_matrix = np.log(np.expm1(1.0 / scale / filters[i + 1]))
            self.matrix.data.fill_(init_matrix)
            self._matrices.append(self.matrix)
            #
            self.bias = Parameter(
                torch.FloatTensor(channels, filters[i + 1], 1)
            )
            init_bias = torch.FloatTensor(
                np.random.uniform(-0.5, 0.5, self.bias.size())
            )
            self.bias.data.copy_(init_bias)
            self._biases.append(self.bias)
            #
            self.factor = Parameter(
                torch.FloatTensor(channels, filters[i + 1], 1)
            )
            self.factor.data.fill_(0.0)
            self._factors.append(self.factor)

    def _logits_cumulative(self, inputs: torch.Tensor):
        """Evaluate logits of the cumulative densities.

        Args:
            inputs (torch.Tensor): The values at which to evaluate the cumulative densities,
            expected to have shape `(channels, 1, batch)`.

        Returns:
            _type_: A tensor of the same shape as inputs, containing the logits of the
                cumulative densities evaluated at the the given inputs.
        """
        logits = inputs
        for i in range(len(self._filters) + 1):
            matrix = torch.nn.functional.softplus(self._matrices[i])
            logits = torch.matmul(matrix, logits)
            logits += self._biases[i]
            factor = torch.tanh(self._factors[i])
            logits += factor * torch.tanh(logits)

        return logits

    def _quantize(self, inputs, mode):
        """Add noise or quantize."""
        if mode == "noise":
            noise = np.random.uniform(-0.5, 0.5, inputs.size())
            noise = torch.Tensor(noise).to(inputs.device)
            return inputs + noise
        if mode == "symbols":
            return RoundNoGradient.apply(inputs)

    def _likelihood(self, inputs):
        """
        Estimate the likelihood.
        """
        inputs = inputs.permute(1, 0, 2, 3, 4).contiguous()
        shape = inputs.size()
        inputs = inputs.view(shape[0], 1, -1)
        inputs = inputs.to(self.matrix.device)
        # Evaluate densities.
        lower = self._logits_cumulative(inputs - 0.5)
        upper = self._logits_cumulative(inputs + 0.5)
        sign = -torch.sign(torch.add(lower, upper)).detach()
        likelihood = torch.abs(
            torch.sigmoid(sign * upper) - torch.sigmoid(sign * lower)
        )
        # reshape to (points, channels)
        likelihood = likelihood.view(shape)
        likelihood = likelihood.permute(1, 0, 2, 3, 4)

        return likelihood

    def forward(self, inputs, quantize_mode="noise"):
        """Pass a tensor through the bottleneck."""
        if quantize_mode is None:
            outputs = inputs
        else:
            outputs = self._quantize(inputs, mode=quantize_mode)
        likelihood = self._likelihood(outputs)
        likelihood = Low_bound.apply(likelihood)

        return outputs, likelihood

    def _pmf_to_cdf(self, pmf):
        cdf = pmf.cumsum(dim=-1)
        spatial_dimensions = pmf.shape[:-1] + (1,)
        zeros = torch.zeros(
            spatial_dimensions, dtype=pmf.dtype, device=pmf.device
        )
        cdf_with_0 = torch.cat([zeros, cdf], dim=-1)
        cdf_with_0 = cdf_with_0.clamp(max=1.)

        return cdf_with_0

    @torch.no_grad()
    def compress(self, inputs):
        inputs = inputs.permute(
            0, 2, 3, 4, 1
        )
        # quantize
        values = self._quantize(inputs, mode="symbols")
        # get symbols
        min_v = values.min().detach().float()  # -17
        max_v = values.max().detach().float()  # 18
        symbols = torch.arange(min_v, max_v + 1)
        symbols = (
            symbols.reshape(1, 1, -1).repeat(values.shape[-1], 1, 1).float()
        )
        symbols = symbols.to(self.matrix.device)

        # get normalized values
        values_norm = values - min_v
        min_v, max_v = torch.tensor([min_v]), torch.tensor([max_v])
        values_norm = values_norm.to(torch.int16)

        # get pmf
        lower = self._logits_cumulative(symbols - 0.5)
        upper = self._logits_cumulative(symbols + 0.5)
        sign = -torch.sign(torch.add(lower, upper))
        likelihood = torch.abs(
            torch.sigmoid(sign * upper) - torch.sigmoid(sign * lower)
        )
        pmf = torch.clamp(
            likelihood, min=self._likelihood_bound
        )  # (channels, 1, num_symbols)
        pmf = pmf.reshape(values.shape[-1], -1)  # (8,N)

        # get cdf
        cdf = self._pmf_to_cdf(pmf)  # cdf
        # arithmetic encoding
        values_norm = values_norm.reshape(-1, values.shape[-1])
        out_cdf = (
            cdf.unsqueeze(0).repeat(values_norm.shape[0], 1, 1).detach().cpu()
        )
        strings = torchac.encode_float_cdf(
            out_cdf, values_norm.cpu(), check_input_bounds=True
        )

        return strings, min_v.cpu().numpy(), max_v.cpu().numpy()

    @torch.no_grad()
    def decompress(self, strings, min_v, max_v, shape):  # shape:(N,C,D,H,W)
        # get symbols
        symbols = torch.arange(min_v, max_v + 1)
        channels = int(shape[1])
        symbols = (
            symbols.reshape(1, 1, -1).repeat(channels, 1, 1).float()
        )  # (channels=8,1,num_symbols)
        symbols = symbols.to(self.matrix.device)

        # get pmf
        lower = self._logits_cumulative(symbols - 0.5)
        upper = self._logits_cumulative(symbols + 0.5)
        sign = -torch.sign(torch.add(lower, upper))
        likelihood = torch.abs(
            torch.sigmoid(sign * upper) - torch.sigmoid(sign * lower)
        )
        pmf = torch.clamp(likelihood, min=self._likelihood_bound)  # (channels, 1, num_symbols)
        pmf = pmf.reshape(channels, -1)  # (8,N)
        # get cdf
        cdf = self._pmf_to_cdf(pmf)
        print(cdf)

        # arithmetic decoding
        out_cdf = (
            cdf.unsqueeze(0)
            .repeat(torch.prod(torch.tensor(shape)).item() // channels, 1, 1)
            .cpu()
        )
        values = torchac.decode_float_cdf(out_cdf, strings)
        values = values.float()
        values += min_v
        values = torch.reshape(
            values, (shape[0], shape[2], shape[3], shape[4], -1)
        )
        values = values.permute(0, 4, 1, 2, 3)
        return values


# if __name__ == "__main__":
#     device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
#     np.random.seed(108)
#     training = False
#     y = np.random.rand(2, 8, 8, 8, 8).astype("float32")  # 0-1均匀分布
#     y = np.round(y * 20 - 10)
#     y_gpu = torch.from_numpy(y).to(device)
#     print("y_gpu[0,0,0,0]:", y_gpu[0, 0, 0, 0])
#     entropy_bottleneck = EntropyBottleneck(channels=8)
#     entropy_bottleneck = entropy_bottleneck.to(device)
#     out, _ = entropy_bottleneck(y_gpu)
#     print(out)
#     y_strings, y_min_v, y_max_v = entropy_bottleneck.compress(y_gpu)  # encode
#     print("y_min_v:", y_min_v)
#     print("y_max_v:", y_max_v)

#     # decode
#     y_decoded = entropy_bottleneck.decompress(
#         y_strings, y_min_v.item(), y_max_v.item(), y_gpu.shape
#     )
#     compare = torch.eq(torch.from_numpy(y).int(), y_decoded.int())
#     compare = compare.float()
#     print(
#         "compare=False:",
#         torch.nonzero(compare < 0.1),
#         len(torch.nonzero(compare < 0.1)),
#     )  # len(torch.nonzero(compare<0.1))=0
#     print("y_decoded[0,0,0,0]:", y_decoded[0, 0, 0, 0])
