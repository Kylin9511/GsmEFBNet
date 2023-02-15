import torch
from torch import nn

from .component import gsm_merge_beamformer

__all__ = ["BeamformingSumRateLoss"]


class BeamformingSumRateLoss(nn.Module):
    def __init__(self, Nr, Nrf, P, snr, device):
        super(BeamformingSumRateLoss, self).__init__()
        self.Nr = Nr
        self.Nrf = Nrf
        self.P = P
        self.noise_power = torch.tensor(P / 10 ** (snr / 10), dtype=torch.float32).to(device)
        self.device = device

    def forward(self, H_real, H_imag, A_real, A_imag, D_real, D_imag, CSet):
        H = torch.view_as_complex(torch.stack([H_real, H_imag], dim=-1))  # N*Nr*Nt
        HSet = H.unsqueeze(dim=1)  # N*1*Nr*Nt

        V_real, V_imag = gsm_merge_beamformer(A_real, A_imag, D_real, D_imag, CSet)
        VSet = torch.view_as_complex(torch.stack([V_real, V_imag], dim=-1))  # N*CSetSize*Nt*Nrf

        I_Nr = torch.eye(self.Nr, dtype=torch.cfloat, device=self.device)
        HSetHermitian = torch.conj(torch.transpose(HSet, -1, -2))
        VSetHermitian = torch.conj(torch.transpose(VSet, -1, -2))

        sigmaSet = (
            self.noise_power * I_Nr + 1 / self.Nrf * HSet @ (VSet @ VSetHermitian) @ HSetHermitian
        )  # N*CSetSize*Nr*Nr
        try:  # For unknown reason, the torch.det() under CUDA env (torch1.10.0) could raise kernel error.
            sigmaDetSet = torch.det(sigmaSet).real  # N*CSetSize, note that hermite matrix has a real determinant
        except RuntimeError:  # It is weird that a simple rerun would solve the RuntimeError above.
            sigmaDetSet = torch.det(sigmaSet).real
        sumRateBeamforming = torch.mean(
            torch.log2(sigmaDetSet / torch.pow(self.noise_power, self.Nr))
        )  # N*CSetSize -> 1,

        return -sumRateBeamforming
