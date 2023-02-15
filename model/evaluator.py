import torch
from scipy.linalg import sqrtm
import numpy as np

from .component import gsm_merge_beamformer

__all__ = ["GsmSumRate"]


class GsmSumRate:
    def __init__(self, Nr, Nrf, P, snr, device, num_samples=100):
        self.Nr = Nr
        self.Nrf = Nrf
        self.P = P
        self.noise_power = torch.tensor(P / 10 ** (snr / 10), dtype=torch.float32).cpu()
        self.device = device
        self.num_samples = num_samples

        # Generate the complex Gaussian (CN(0,1)) identical random received symbol
        sSetNumpy = np.random.normal(loc=0, scale=1.0 / np.sqrt(2), size=(self.Nr, num_samples, 2)).view(np.cdouble)
        sSet = torch.from_numpy(np.squeeze(sSetNumpy)).cpu()  # Nr*num_samples
        self.sSet = sSet.unsqueeze(dim=0).unsqueeze(dim=0)  # 1*1*Nr*num_samples

    def _to_cpu(self, H_real, H_imag, A_real, A_imag, D_real, D_imag, CSet):
        # transfer all the data to the CPU
        return H_real.cpu(), H_imag.cpu(), A_real.cpu(), A_imag.cpu(), D_real.cpu(), D_imag.cpu(), CSet.cpu()

    def __call__(self, H_real, H_imag, A_real, A_imag, D_real, D_imag, CSet):
        r"""It seems that the original GSM sum rate is too complicated for a direct end-to-end training.
        Maybe the direct unsupervised training is possible after some further deduction of simplification.
        """
        # make evaluation on the CPU
        H_real, H_imag, A_real, A_imag, D_real, D_imag, CSet = self._to_cpu(
            H_real, H_imag, A_real, A_imag, D_real, D_imag, CSet
        )

        # The beamforming sum rate
        H = torch.view_as_complex(torch.stack([H_real, H_imag], dim=-1))  # N*Nr*Nt
        HSet = H.unsqueeze(dim=1)  # N*1*Nr*Nt

        V_real, V_imag = gsm_merge_beamformer(A_real, A_imag, D_real, D_imag, CSet)
        VSet = torch.view_as_complex(torch.stack([V_real, V_imag], dim=-1))  # N*CSetSize*Nt*Nrf

        I_Nr = torch.eye(self.Nr, dtype=torch.cfloat).cpu()
        HSetHermitian = torch.conj(torch.transpose(HSet, -1, -2))
        VSetHermitian = torch.conj(torch.transpose(VSet, -1, -2))

        sigmaSet = (
            self.noise_power * I_Nr + self.P / self.Nrf * HSet @ (VSet @ VSetHermitian) @ HSetHermitian
        )  # N*CSetSize*Nr*Nr

        sigmaDetSet = torch.det(sigmaSet).real

        sumRateBeamforming = torch.mean(torch.log2(sigmaDetSet / torch.pow(self.noise_power, self.Nr))).to(
            device=self.device
        )  # N*CSetSize -> 1,

        # The additional GSM sum rate
        # TODO: considering using torch.solve(A, B) as as better torch.inverse(A) @ B
        # - https://pytorch.org/docs/stable/generated/torch.linalg.inv.html#torch.linalg.inv
        SigmaInvSet = torch.inverse(sigmaSet)  # N*CSetSize*Nr*Nr

        # TODO: current version of sqrtm is slow and does not support gradient back-propagation.
        # Note that if pytorch support sqrtm operator, the whole process can be done on GPU with huge acceleration.
        # Several open-source resource provide sqrtm operator based on pytorch.
        # Currently they are not pplied for easier coding, but could be referred to in the future if necessary.
        # - A neat implementation: https://github.com/steveli/pytorch-sqrtm
        # - A good discussion: https://github.com/pytorch/pytorch/issues/25481
        #   - A comprehensive implementation: https://github.com/msubhransu/matrix-sqrt
        batch_size, CSetSize, _, _ = sigmaSet.size()
        sigmaSetNumpy = sigmaSet.resolve_conj().resolve_neg().numpy()  # .numpy(force=True) for torch>=1.13
        sigmaSqrtSetNumpy = np.zeros((batch_size, CSetSize, self.Nr, self.Nr), dtype=np.cdouble)
        for idBatch in range(batch_size):
            for idC in range(CSetSize):
                sigmaSqrtSetNumpy[idBatch, idC, ...] = sqrtm(sigmaSetNumpy[idBatch, idC, ...])
        sigmaSqrtSet = torch.from_numpy(sigmaSqrtSetNumpy)  # N*CSetSize*Nr*Nr

        # ySet = torch.matmul(sigmaSqrtSet, self.sSet)  # N*CSetSize*Nr*num_samples, For numpy tester
        ySet = torch.matmul(sigmaSqrtSet, self.sSet).to(torch.cfloat)  # N*CSetSize*Nr*num_samples
        ySet = torch.transpose(ySet, -1, -2)  # N*CSetSize*num_samples*Nr
        ySetExpand = ySet.unsqueeze(dim=-1)  # N*CSetSize*num_samples*Nr*1
        ySetExpandHermitian = torch.conj(ySet.unsqueeze(dim=-2))  # N*CSetSize*num_samples*1*Nr
        SigmaInvSetExpand = torch.repeat_interleave(
            SigmaInvSet.unsqueeze(dim=2), repeats=self.num_samples, dim=2
        )  # N*CSetSize*num_samples*Nr*Nr

        yDensityCoefficient = 1 / (np.power(np.pi, self.Nr) * sigmaDetSet).unsqueeze(dim=-1)  # N*CSetSize*1
        yDensityExp = torch.exp(-(ySetExpandHermitian @ SigmaInvSetExpand @ ySetExpand).real)  # N*CSetSize*num_samples
        yDensitySet = yDensityCoefficient * yDensityExp.squeeze()  # N*CSetSize*num_samples

        # This is hard to organize. The loop is necessary since the CMeanSet is derived
        # from a fixed y_{idCSet} traverses through the overall CSet on the sigmaInvSet.
        yDensityCMeanSet = torch.zeros_like(yDensitySet)  # N*CSetSize*num_samples
        for idCSet in range(CSetSize):
            ySetTmp = ySet[:, idCSet, ...].unsqueeze(dim=1)  # N*1*num_samples*Nr
            ySetTmpExpand = ySetTmp.unsqueeze(dim=-1)  # N*1*num_samples*Nr*1
            ySetTmpExpandHermitian = torch.conj(ySetTmp.unsqueeze(dim=-2))  # N*1*num_samples*1*Nr
            yDensityExpTmp = torch.exp(
                -(ySetTmpExpandHermitian @ SigmaInvSetExpand @ ySetTmpExpand).real
            )  # N*CSetSize*num_samples
            yDensitySetTmp = yDensityCoefficient * yDensityExpTmp.squeeze()  # N*CSetSize*num_samples
            yDensityCMeanSet[:, idCSet, ...] = torch.mean(yDensitySetTmp, dim=1)

        sumRateSpatialModulationSet = torch.log2((yDensitySet / yDensityCMeanSet))  # N*CSetSize*num_samples
        sumRateSpatialModulation = torch.mean(sumRateSpatialModulationSet).to(
            device=self.device
        )  # N*CSetSize*num_samples -> 1

        return sumRateBeamforming + sumRateSpatialModulation
