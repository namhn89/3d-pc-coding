import models.model_voxception as model
import numpy as np
import torch
from models.autoencoder import AnalysisTransform
from models.autoencoder import HyperAnalysisTransform
from models.autoencoder import HyperSynthesisTransform
from models.autoencoder import SynthesisTransform
from models.conditional_entropy_model import SymmetricConditional
from models.entropy_model import EntropyBottleneck


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# class PCCModel(torch.nn.Module):
#     def __init__(self, lower_bound: float = 1e-9, channels: int = 8):
#         super().__init__()
#         self.lower_bound = lower_bound
#         self.encoder = AnalysisTransform()
#         self.decoder = SynthesisTransform()
#         self.hyper_encoder = HyperAnalysisTransform()
#         self.hyper_decoder = HyperSynthesisTransform()
#         self.entropy_bottleneck = EntropyBottleneck(channels)
#         self.conditional_entropy_model = SymmetricConditional()
#     def forward(self, x, training=True):
#         y = self.encoder(x)
#         z = self.hyper_encoder(y)
#         z_tilde, likelihoods_hyper = self.entropy_bottleneck(
#             z, quantize_mode="noise" if training else "symbols")
#         loc, scale = self.hyper_decoder(z_tilde)
#         scale = torch.clamp(scale, min=self.lower_bound)
#         y_tilde, likelihoods = self.conditional_entropy_model(
#             y, loc, scale, quantize_mode="noise" if training else "symbols")
#         x_tilde = self.decoder(y_tilde)
#         return {
#             'likelihoods': likelihoods,
#             'likelihoods_hyper': likelihoods_hyper,
#             'x_tilde': x_tilde
#         }


class PCCModel(torch.nn.Module):
    def __init__(self, lower_bound=1e-9):
        super().__init__()
        self.analysis_transform = model.AnalysisTransform()
        self.synthesis_transform = model.SynthesisTransform()
        self.hyper_encoder = model.HyperEncoder()
        self.hyper_decoder = model.HyperDecoder()
        self.entropy_bottleneck = EntropyBottleneck(channels=8)
        self.conditional_entropy_model = SymmetricConditional()
        self.lower_bound = lower_bound

    def forward(self, x, training=True):
        y = self.analysis_transform(x)
        z = self.hyper_encoder(y)
        z_tilde, likelihoods_hyper = self.entropy_bottleneck(
            z, quantize_mode="noise" if training else "symbols")
        loc, scale = self.hyper_decoder(z_tilde)
        scale = torch.clamp(scale, min=self.lower_bound)
        y_tilde, likelihoods = self.conditional_entropy_model(
            y, loc, scale, quantize_mode="noise" if training else "symbols")
        x_tilde = self.synthesis_transform(y_tilde)

        return {
            'likelihoods': likelihoods,
            'likelihoods_hyper': likelihoods_hyper,
            'x_tilde': x_tilde
        }


# if __name__ == '__main__':
#     torch.manual_seed(3)
#     np.random.seed(3)
#     x = torch.randn(1, 1, 64, 64, 64)
#     model = PCCModel(lower_bound=1e-9)
#     print(count_parameters(model))
#     y = model(x)
#     print(y['x_tilde'][0, 0, 0, 0, 0])
