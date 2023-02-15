import torch
import numpy as np

from utils.configs import args
from utils import logger, Tester
from utils import init_device, init_model
from dataset import MatRawChannelDataLoader
from model import GsmSumRate, BeamformingSumRateLoss


def main():
    logger.info("=> PyTorch Version: {}".format(torch.__version__))

    # Environment initialization
    device = init_device(args.seed, args.cpu_affinity)

    test_loader = MatRawChannelDataLoader(data_dir=args.test_data_dir, device=device, bs=None)

    # Define model
    model = init_model(args)
    model.to(device)

    # Define loss function
    criterion = BeamformingSumRateLoss(
        Nr=args.antennas_ue,
        Nrf=args.rf_chains,
        P=args.transmit_power,
        snr=args.snr,
        device=device,
    ).to(device)

    # Define the sum rate evaluator
    evaluator = GsmSumRate(
        Nr=args.antennas_ue,
        Nrf=args.rf_chains,
        P=args.transmit_power,
        snr=args.snr,
        device=device,
        num_samples=100,
    )

    # Inference mode
    if args.evaluate:
        sum_rate = Tester(model, args.transmit_power, args.snr, device, criterion, evaluator)(test_loader)
        print(f"\n=! Final test sum_rate: {sum_rate:.3e}\n")
        return


if __name__ == "__main__":
    main()
