import torch
import torch.nn as nn
import scipy.linalg as sci
import numpy as np
from scipy.special import comb
from itertools import combinations

__all__ = [
    "SigmoidT",
    "gsm_merge_beamformer",
    "generate_antenna_selection_set",
    "SubArrayGSMPilotNet",
    "SubArrayGSMBeamformingNet",
]


class SigmoidT(torch.autograd.Function):
    @staticmethod
    def forward(ctx, y, anneal_factor):
        y_tanh = torch.sigmoid(anneal_factor * y) * 2 - 1
        ctx.save_for_backward(y_tanh, torch.tensor(anneal_factor))
        y_sign = torch.sign(y)
        return y_sign

    @staticmethod
    def backward(ctx, grad_output):
        y_tanh, anneal_factor = ctx.saved_tensors
        grad = anneal_factor.item() * (y_tanh + 1) * (1 - y_tanh) / 2  # sigmoid gradient
        grad_input = grad_output * grad
        return grad_input, None


def gsm_merge_beamformer(A_real, A_imag, D_real, D_imag, CSet):
    r"""Merge structured GSM beamformer into a overall beamformer matrix V"""

    C = CSet.unsqueeze(dim=0)  # 1*CSetSize*Nt*Nrf
    ArCDr = torch.matmul(torch.matmul(A_real, C), D_real)
    ArCDi = torch.matmul(torch.matmul(A_real, C), D_imag)
    AiCDr = torch.matmul(torch.matmul(A_imag, C), D_real)
    AiCDi = torch.matmul(torch.matmul(A_imag, C), D_imag)
    V_real = ArCDr - AiCDi  # N*CSetSize*Nt*Nrf
    V_imag = ArCDi + AiCDr  # N*CSetSize*Nt*Nrf
    return V_real, V_imag


def generate_antenna_selection_set(Nt, Nrf, Nk, num_target):
    r"""Generate a set of antenna selection matrix based on the maximum hamming distance principle.

    Args:
        Nt: the number of transmit antennas
        Nrf: the number of RF-chains
        Nk: the number of antennas in each antenna group
        num_target: the number of selected antenna activation combination
    Return:
        selectSet: A selected antenna selection matrix of dimension num_target*Nt*Nrf
    """

    assert Nt % Nk == 0, f"Illegal Nt={Nt}, Nk={Nk}"
    Ng = Nt // Nk
    maxSize = int(comb(Ng, Nrf))
    globalSet = torch.zeros((maxSize, Nt, Nrf))
    for idSelectionMask, active_antennas in enumerate(combinations(range(Ng), Nrf)):
        for idAntenna, active_antenna in enumerate(active_antennas):
            globalSet[idSelectionMask, active_antenna * Nk : active_antenna * Nk + Nk, idAntenna] = 1

    availableList = []
    selectSet = torch.zeros((num_target, Nt, Nrf))
    for id in range(num_target):
        if len(availableList) == 0:
            availableList = list(range(maxSize))
        hammingDistance = torch.zeros(len(availableList))
        for idAvailable, candidateId in enumerate(availableList):
            if id == 0:
                break
            currentSelectSet = selectSet[:id, ...]
            candidateExpand = globalSet[candidateId, ...].unsqueeze(dim=0).repeat(id, 1, 1)
            assert currentSelectSet.size() == candidateExpand.size()
            # merge the selected antenna group together for the hamming distance counting
            hammingDistance[idAvailable] = torch.count_nonzero(
                currentSelectSet.sum(dim=-1) != candidateExpand.sum(dim=-1)
            )
        selectId = availableList[torch.argmax(hammingDistance)]
        selectSet[id, ...] = globalSet[selectId, ...]
        availableList.remove(selectId)
    return selectSet


class SubArrayGSMPilotNet(nn.Module):
    def __init__(self, Nt, Nr, Nrf, Nk, L, P):
        super(SubArrayGSMPilotNet, self).__init__()
        self.Nt = Nt
        self.Nr = Nr
        self.Nrf = Nrf
        self.Nk = Nk
        self.L = L
        self.P = P
        self.Ng = self.Nt // self.Nk

        # take L pilots from DFT matrix (L * Nt)
        DFT_Matrix = sci.dft(self.Nt)
        X = np.angle(DFT_Matrix[:: int(np.floor(self.Nt / L)), :][:L, :])  # take angle for the hybrid beamforming arch

        # register the learnable pilot Parameter
        self.X_theta = nn.Parameter(torch.tensor(X, dtype=torch.float32))

        # generate the pilot mask to meet the architecture of GSM beamforming
        antenna_selection_matrices = generate_antenna_selection_set(Nt, Nrf, Nk, L)  # L*Nt*Nrf
        self.register_buffer("antenna_pilot_mask", antenna_selection_matrices.sum(dim=-1))  # L*Nt

    def forward(self, H_real, H_imag, noise_std):
        # Note that the power per antenna is normalized to sqrt(P/Nk) since one RF-chain only connects to Nk antennas
        X_real = np.sqrt(self.P / self.Nk) * torch.cos(self.X_theta) * self.antenna_pilot_mask
        X_imag = np.sqrt(self.P / self.Nk) * torch.sin(self.X_theta) * self.antenna_pilot_mask
        Y_real = torch.einsum("LT,NRT -> NRL", [X_real, H_real]) - torch.einsum("LT,NRT -> NRL", [X_imag, H_imag])
        Y_imag = torch.einsum("LT,NRT -> NRL", [X_real, H_imag]) + torch.einsum("LT,NRT -> NRL", [X_imag, H_real])
        Y_real = Y_real.reshape(-1, self.Nr * self.L)
        Y_imag = Y_imag.reshape(-1, self.Nr * self.L)
        Y = torch.concat([Y_real, Y_imag], dim=1)
        Y_N = Y + torch.normal(mean=torch.zeros_like(Y), std=noise_std)
        return Y_N


class SubArrayGSMBeamformingNet(nn.Module):
    def __init__(self, Nt, Nrf, Nk, feature_dim=512):
        super(SubArrayGSMBeamformingNet, self).__init__()
        assert Nt % Nk == 0, f"Nt={Nt} must be divisible by Nk={Nk}"
        self.Nt = Nt
        self.Nrf = Nrf
        self.Nk = Nk
        self.Ng = self.Nt // self.Nk

        # Set of the antenna combination matrix C, which is a fixed buffer
        maxCSetSize = int(comb(self.Ng, self.Nrf))
        self.CSetSize = int(np.power(2, np.floor(np.log2(maxCSetSize))))
        CSet = generate_antenna_selection_set(Nt, Nrf, Nk, self.CSetSize)
        self.register_buffer(f"CSet", CSet)  # CSetSize*Nt*Nrf

        # FC layers to learn the analog and digital beamformer
        self.FC_A_theta = nn.Linear(feature_dim, self.Nt)
        self.FC_D_real = nn.Linear(feature_dim, self.CSetSize * self.Nrf * self.Nrf)
        self.FC_D_imag = nn.Linear(feature_dim, self.CSetSize * self.Nrf * self.Nrf)

    def forward(self, x):
        batch_size = x.size(0)  # N

        # Produce the low resolution Analog precoding matrix with q-bit phase shifter
        A_theta = self.FC_A_theta(x)  # N*Nt
        A_real = torch.diag_embed(torch.cos(A_theta)).unsqueeze(dim=1)  # N*Nt -> N*1*Nt*Nt
        A_imag = torch.diag_embed(torch.sin(A_theta)).unsqueeze(dim=1)  # N*Nt -> N*1*Nt*Nt

        # Produce the Digital precoding matrix
        D_real = self.FC_D_real(x).view(batch_size, self.CSetSize, self.Nrf, self.Nrf)
        D_imag = self.FC_D_imag(x).view(batch_size, self.CSetSize, self.Nrf, self.Nrf)

        # Normalization of digital precoding matrix with broadcasting
        V_real, V_imag = gsm_merge_beamformer(A_real, A_imag, D_real, D_imag, self.CSet)
        V_F_norm = torch.sqrt(torch.sum(V_real**2 + V_imag**2, dim=(2, 3)))
        V_F_norm = V_F_norm.view(batch_size, self.CSetSize, 1, 1)  # N*CSetSize*1*1
        D_real = np.sqrt(self.Nrf) * torch.div(D_real, V_F_norm)
        D_imag = np.sqrt(self.Nrf) * torch.div(D_imag, V_F_norm)

        return A_real, A_imag, D_real, D_imag, self.CSet
