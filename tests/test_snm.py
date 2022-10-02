import torch
from model import SwapNoiseMasker, TransformerAutoEncoder, TransformerEncoder


def test_tf_encoder():
    m = TransformerEncoder(4, 2, .1, 16)
    x = torch.rand((32, 8))
    x = x.reshape((32, 2, 4)).permute((1, 0, 2))
    o = m(x)
    assert o.shape == torch.Size([2, 32, 4])


def test_dae_model():
    m = TransformerAutoEncoder(5, 2, 3, 16, 4, 4, 2, .1, 4, .75)
    x = torch.cat([torch.randint(0, 2, (5, 2)), torch.rand((5, 3))], dim=1)
    f = m.feature(x)
    assert f.shape == torch.Size([5, 16 * 3])
    # loss = m.loss(x, x, (x > .2).float())


def test_swap_noise():
    probas = [.2, .5, .8]
    m = SwapNoiseMasker(probas)
    diffs = []
    for i in range(1000):
        x = torch.rand((32, 3))
        noisy_x, _ = m.apply(x)
        diffs.append((x != noisy_x).float().mean(0).unsqueeze(0)) 

    print('specified : ', probas, ' - actual : ', torch.cat(diffs, 0).mean(0))