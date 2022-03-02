import numpy as np
import torch
from autoencoder import AnalysisTransform
from autoencoder import HyperAnalysisTransform
from autoencoder import HyperSynthesisTransform
from autoencoder import SynthesisTransform
from conditional_entropy_model import SymmetricConditional
from entropy_model import EntropyBottleneck


def test_model():
    training = False
    x = torch.randn(8, 1, 64, 64, 64)
    entropy_bottleneck = EntropyBottleneck(8)
    conditional_entropy_model = SymmetricConditional()
    encoder = AnalysisTransform()
    decoder = SynthesisTransform()
    hyper_encoder = HyperAnalysisTransform()
    hyper_decoder = HyperSynthesisTransform()
    y = encoder(x)
    print(y.shape)
    z = hyper_encoder(y)
    print(z.shape)
    z_tilde, likelihoods_hyper = entropy_bottleneck(z, quantize_mode="noise" if training else "symbols")
    loc, scale = hyper_decoder(z)
    scale = torch.clamp(scale, min=1e-9)
    y_tilde, likelihoods = conditional_entropy_model(y, loc, scale, quantize_mode="noise" if training else "symbols")
    x_tilde = decoder(y_tilde)
    print(x_tilde.shape)
    assert x_tilde.shape == x.shape


if __name__ == "__main__":
    test_model()
