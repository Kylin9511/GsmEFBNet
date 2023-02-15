import torch.nn as nn
from collections import OrderedDict

from utils import logger
from .component import SubArrayGSMPilotNet, SubArrayGSMBeamformingNet, SigmoidT

__all__ = ["GsmEFBNet"]


class ConvBN(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBN, self).__init__(
            OrderedDict(
                [
                    (
                        "conv",
                        nn.Conv1d(
                            in_planes, out_planes, kernel_size, stride, padding=padding, groups=groups, bias=False
                        ),
                    ),
                    ("bn", nn.BatchNorm1d(out_planes)),
                ]
            )
        )


class EncoderNet(nn.Module):
    def __init__(self, Nt, Nr, Nrf, Nk, L, B, P, expand=24):
        super(EncoderNet, self).__init__()
        self.Nr = Nr
        self.L = L
        self.pilot = SubArrayGSMPilotNet(Nt, Nr, Nrf, Nk, L, P)
        self.bn1d = nn.BatchNorm1d(2 * Nr * L)

        half_dim = Nr * L // 2
        self.conv_expand = ConvBN(2, expand, kernel_size=(half_dim + 1) if half_dim % 2 == 0 else half_dim)
        self.conv_branch1 = ConvBN(expand, expand, kernel_size=7)
        self.conv_branch2 = ConvBN(expand, expand, kernel_size=11)
        self.relu = nn.ReLU()

        self.fc_compress = nn.Linear(expand * Nr * L, B)

    def forward(self, H_real, H_imag, anneal_factor, noise_std):
        y = self.pilot(H_real, H_imag, noise_std)
        y = self.relu(self.bn1d(y))

        batch_size = H_real.size(0)
        y = y.reshape(batch_size, 2, self.Nr * self.L)
        y = self.relu(self.conv_expand(y))
        y_branch1 = self.conv_branch1(y)
        y_branch2 = self.conv_branch2(y)
        y = self.relu(y + y_branch1 + y_branch2).reshape(batch_size, -1)

        y = self.fc_compress(y)
        q = SigmoidT().apply(y, anneal_factor)
        return q


class DecoderNet(nn.Module):
    def __init__(self, Nt, Nrf, Nk, B):
        super(DecoderNet, self).__init__()
        module_list = [
            ("fc1", nn.Linear(B, 2048)),
            ("bn1", nn.BatchNorm1d(2048)),
            ("relu1", nn.ReLU()),
            ("fc2", nn.Linear(2048, 1024)),
            ("bn2", nn.BatchNorm1d(1024)),
            ("relu2", nn.ReLU()),
            ("fc3", nn.Linear(1024, 512)),
            ("bn3", nn.BatchNorm1d(512)),
            ("relu3", nn.ReLU()),
        ]
        self.decoder = nn.Sequential(OrderedDict(module_list))
        self.beamformer = SubArrayGSMBeamformingNet(Nt, Nrf, Nk, feature_dim=512)

    def forward(self, q):
        feature = self.decoder(q)
        return self.beamformer(feature)


class GsmEFBNet(nn.Module):
    def __init__(
        self, Nt: int, Nr: int, Nk: int, Nrf: int, L: int, B: int, P: int, anneal_init: float, anneal_rate: float
    ):
        r"""
        Args:
            Nt: number of antennas at the base station
            Nr: number of antennas at the user equipment
            Nk: number of antennas in each antenna group, must be divisible by Nt
            Nrf: number of RF-chains at the base station
            L: number of channel paths
            B: number of feedback bits
            P: the power constraint of the base station
            anneal_init: initial anneal factor of sigmoidT function
            anneal_rate: anneal factor update rate of sigmoidT function
        """

        super(GsmEFBNet, self).__init__()
        self.user_equipment = EncoderNet(Nt, Nr, Nrf, Nk, L, B, P)
        self.base_station = DecoderNet(Nt, Nrf, Nk, B)
        self.anneal_factor = anneal_init
        self.anneal_rate = anneal_rate
        assert isinstance(self.anneal_rate, float) and self.anneal_rate > 1, self.anneal_rate
        logger.info(f"=> Model: Using GsmEFBNet with Nt={Nt}, Nr={Nr}, Nk={Nk}, Nrf={Nrf}, B={B}, P={P}")
        logger.info(f"   The SigmoidT is activated with anneal_init={anneal_init} and anneal_rate={anneal_rate}")

    def forward(self, H_real, H_imag, noise_std):
        feedback_bits = self.user_equipment(H_real, H_imag, self.anneal_factor, noise_std)
        self.anneal_factor = min(self.anneal_factor * self.anneal_rate, 10)
        A_real, A_imag, D_real, D_imag, CSet = self.base_station(feedback_bits)
        return A_real, A_imag, D_real, D_imag, CSet
