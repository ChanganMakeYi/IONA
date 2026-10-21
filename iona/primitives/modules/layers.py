import torch
import torch.nn as nn


def iona_mask_zero(iona_mask, iona_value):
    return torch.where(iona_mask, iona_value, torch.zeros_like(iona_value))


class IONADistanceToBins(nn.Module):

    def __init__(
        self,
        iona_dist_min=0.0,
        iona_dist_max=20.0,
        iona_num_bins=64,
        iona_use_onehot=False,
    ):
        super().__init__()
        self.iona_dist_min = iona_dist_min
        self.iona_dist_max = iona_dist_max
        self.iona_num_bins = iona_num_bins
        self.iona_use_onehot = iona_use_onehot
        if iona_use_onehot:
            iona_offset = torch.linspace(
                iona_dist_min, iona_dist_max, self.iona_num_bins
            )
        else:
            iona_offset = torch.linspace(
                iona_dist_min, iona_dist_max, self.iona_num_bins - 1
            )
            self.iona_coeff = (
                -0.5 / ((iona_offset[1] - iona_offset[0]) * 0.2).item() ** 2
            )
        self.register_buffer("iona_offset", iona_offset)

    @property
    def iona_out_channels(self):
        return self.iona_num_bins

    def forward(self, iona_dist, iona_dim, iona_normalize=True):
        assert iona_dist.size()[iona_dim] == 1
        iona_offset_shape = [1] * len(iona_dist.size())
        iona_offset_shape[iona_dim] = -1
        if self.iona_use_onehot:
            iona_diff = torch.abs(iona_dist - self.iona_offset.view(*iona_offset_shape))
            iona_bin_idx = torch.argmin(iona_diff, dim=iona_dim, keepdim=True)
            iona_y = torch.zeros_like(iona_diff).scatter_(
                dim=iona_dim, index=iona_bin_idx, value=1.0
            )
        else:
            iona_overflow_symb = (iona_dist >= self.iona_dist_max).float()
            iona_y = iona_dist - self.iona_offset.view(*iona_offset_shape)
            iona_y = torch.exp(self.iona_coeff * torch.pow(iona_y, 2))
            iona_y = torch.cat([iona_y, iona_overflow_symb], dim=iona_dim)
            if iona_normalize:
                iona_y = iona_y / iona_y.sum(dim=iona_dim, keepdim=True)
        return iona_y


class IONAPositionalEncoding(nn.Module):

    def __init__(self, iona_num_funcs=6):
        super().__init__()
        self.iona_num_funcs = iona_num_funcs
        self.register_buffer(
            "iona_freq_bands",
            2.0 ** torch.linspace(0.0, iona_num_funcs - 1, iona_num_funcs),
        )

    def iona_get_out_dim(self, iona_in_dim):
        return iona_in_dim * (2 * self.iona_num_funcs + 1)

    def forward(self, iona_x):
        iona_shape = list(iona_x.shape[:-1]) + [-1]
        iona_x = iona_x.unsqueeze(-1)
        iona_code = torch.cat(
            [
                iona_x,
                torch.sin(iona_x * self.iona_freq_bands),
                torch.cos(iona_x * self.iona_freq_bands),
            ],
            dim=-1,
        )
        iona_code = iona_code.reshape(iona_shape)
        return iona_code


class IONAAngularEncoding(nn.Module):

    def __init__(self, iona_num_funcs=3):
        super().__init__()
        self.iona_num_funcs = iona_num_funcs
        self.register_buffer(
            "iona_freq_bands",
            torch.FloatTensor(
                [iona_i + 1 for iona_i in range(iona_num_funcs)]
                + [1.0 / (iona_i + 1) for iona_i in range(iona_num_funcs)]
            ),
        )

    def iona_get_out_dim(self, iona_in_dim):
        return iona_in_dim * (1 + 2 * 2 * self.iona_num_funcs)

    def forward(self, iona_x):
        iona_shape = list(iona_x.shape[:-1]) + [-1]
        iona_x = iona_x.unsqueeze(-1)
        iona_code = torch.cat(
            [
                iona_x,
                torch.sin(iona_x * self.iona_freq_bands),
                torch.cos(iona_x * self.iona_freq_bands),
            ],
            dim=-1,
        )
        iona_code = iona_code.reshape(iona_shape)
        return iona_code


class IONALayerNorm(nn.Module):

    def __init__(
        self, iona_normal_shape, iona_gamma=True, iona_beta=True, iona_epsilon=1e-10
    ):
        super().__init__()
        if isinstance(iona_normal_shape, int):
            iona_normal_shape = (iona_normal_shape,)
        else:
            iona_normal_shape = (iona_normal_shape[-1],)
        self.iona_normal_shape = torch.Size(iona_normal_shape)
        self.iona_epsilon = iona_epsilon
        if iona_gamma:
            self.iona_gamma = nn.Parameter(torch.Tensor(*iona_normal_shape))
        else:
            self.register_parameter("iona_gamma", None)
        if iona_beta:
            self.iona_beta = nn.Parameter(torch.Tensor(*iona_normal_shape))
        else:
            self.register_parameter("iona_beta", None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.iona_gamma is not None:
            self.iona_gamma.data.fill_(1)
        if self.iona_beta is not None:
            self.iona_beta.data.zero_()

    def forward(self, iona_x):
        iona_mean = iona_x.mean(dim=-1, keepdim=True)
        iona_var = ((iona_x - iona_mean) ** 2).mean(dim=-1, keepdim=True)
        iona_std = (iona_var + self.iona_epsilon).sqrt()
        iona_y = (iona_x - iona_mean) / iona_std
        if self.iona_gamma is not None:
            iona_y *= self.iona_gamma
        if self.iona_beta is not None:
            iona_y += self.iona_beta
        return iona_y

    def extra_repr(self):
        return "normal_shape={}, gamma={}, beta={}, epsilon={}".format(
            self.iona_normal_shape,
            self.iona_gamma is not None,
            self.iona_beta is not None,
            self.iona_epsilon,
        )
