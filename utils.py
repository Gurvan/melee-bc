import torch
from torch import nn


def expand_but_last(x, shape):
    return x.expand(*shape[:-1], x.shape[-1])


def add_dims_front(input, shape):
    while input.ndim < len(shape):
        input = input[None, ...]
    return input


def add_dims_before_last(input, shape):
    while input.ndim < len(shape):
        input = input[..., None, :]
    return input


def extend_as(input, shape):
    input = add_dims_before_last(input, shape)
    return expand_but_last(input, shape)


# @torch.compile
def forget_mult(f, x, h=None, time_dim=-2):
    result = []
    forgets = f.split(1, dim=time_dim)
    prev_h = h
    for i, h in enumerate((f * x).split(1, dim=time_dim)):
        if prev_h is not None:
            h = h + (1 - forgets[i]) * prev_h
        result.append(h.squeeze())
        prev_h = h
    return torch.stack(result, dim=-2)


if torch.__version__.startswith("2."):
    forget_mult = torch.compile(forget_mult)


if __name__ == "__main__":
    import time

    B, T, D = 32, 64, 512
    device = "cuda"

    x = torch.randn(B, T, D).to(device)
    # model = QRNN(512, 512, num_layers=3).to(device)
    model = nn.GRU(512, 512, num_layers=3, batch_first=True).to(device)

    # model = torch.compile(model)

    x = model(x)[0]
    x = model(x)[0]

    t = time.time()
    for _ in range(1000):
        x = model(x)[0].detach()

    torch.cuda.synchronize()
    print(time.time() - t)
