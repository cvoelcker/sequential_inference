import torch


def exponential_moving_matrix(n: int, discount_factor: float):
    x = torch.arange(n).unsqueeze(0)
    x = x.repeat(n, 1)
    return torch.tril(discount_factor**x)
