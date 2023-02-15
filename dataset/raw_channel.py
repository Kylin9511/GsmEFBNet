import torch
import scipy.io as sio
from typing import Union

__all__ = ["MatRawChannelDataLoader"]


class MatRawChannelDataLoader(object):
    def __init__(
        self,
        data_dir: str,
        device: torch.device,
        bs: Union[int, None],
    ):
        self.data_dir = data_dir
        data = sio.loadmat(data_dir)
        self.H_real = torch.tensor(data["H_real"], dtype=torch.float32, device=device)
        self.H_imag = torch.tensor(data["H_imag"], dtype=torch.float32, device=device)

        self.bs = len(self.H_real) if bs is None else bs
        assert len(self.H_real) == len(self.H_imag)
        assert (
            len(self.H_real) % self.bs == 0
        ), f"Dataset length {self.H_real} must be dividable by batch size {self.bs}."

        self.batch_idx = 0
        self.batch_per_epoch = self.H_real.size(0) // self.bs

    def __len__(self):
        return self.batch_per_epoch

    def __iter__(self):
        self.batch_idx = 0
        return self

    def __next__(self):
        if self.batch_idx == self.batch_per_epoch:
            raise StopIteration
        batch_H_real = self.H_real[self.batch_idx * self.bs : (self.batch_idx + 1) * self.bs, ...]
        batch_H_imag = self.H_imag[self.batch_idx * self.bs : (self.batch_idx + 1) * self.bs, ...]
        self.batch_idx += 1
        return batch_H_real, batch_H_imag
