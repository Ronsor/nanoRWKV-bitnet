import torch
import torch.nn as nn
import torch.nn.functional as F


class BitLinear158(nn.Module):
    def __init__(self, in_features, out_features, bias=True, scale_eps=1e-5, device=None, dtype=None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scale_eps = scale_eps

        self.weight = nn.Parameter(torch.randn(out_features, in_features, **factory_kwargs))
        self.bias = nn.Parameter(torch.zeros(out_features, **factory_kwargs)) if bias else None

    @property
    def qweight(self):
        scale = self.scale_eps + torch.mean(self.weight.abs()).detach()
        quant = torch.round(self.weight / scale).clamp_(-1, 1)
        return (quant - self.weight).detach() + self.weight

    def forward(self, x):
        return F.linear(x, self.qweight, self.bias)

# We're only doing ternary
BitLinear = BitLinear158
