import argparse
import logging
import time

import torch
from torch.utils.data import DataLoader
from torcheval.metrics.functional import peak_signal_noise_ratio

from dataset import OpenVideoDataset
from models import AutoEncoderModelType, get_trained_model

DTYPE = torch.bfloat16
SEED = 42


def main(args):
    dataset = OpenVideoDataset(
        args.dataset_dir,
        max_frames=args.num_frames,
        resolution=(args.width, args.height),
        dtype=DTYPE,
    )
    generator = torch.Generator()
    generator.manual_seed(SEED)
    train_set, test_set = torch.utils.data.random_split(dataset, [0.8, 0.2], generator)

    test_dataloader = DataLoader(test_set, shuffle=True, batch_size=args.batch_size)
    model = get_trained_model(args.model_path, args.model_type, dtype=DTYPE).to(
        args.device
    )
    start_time = time.time()
    total_batches = len(test_dataloader)
    batch_psnrs = []
    with torch.no_grad():
        for i, batch in enumerate(test_dataloader):
            if args.device == "cuda":
                torch.cuda.empty_cache()

            batch = batch.to(args.device)
            latent_dist = model.encode(batch).latent_dist
            z = latent_dist.sample()
            output = model.decode(z).sample
            psnr_value = [
                peak_signal_noise_ratio(output[i], batch[i], 1.0)
                for i in range(batch.shape[0])
            ]
            batch_psnrs.append(torch.mean(torch.stack(psnr_value, 0)))
            logging.info(f"BATCH: {i + 1}/{total_batches}, PSNR: {batch_psnrs[-1]}")

    end_time = time.time()

    logging.info("Total time: %d sec" % (end_time - start_time))
    logging.info("Average PSNR %f " % (torch.mean(torch.stack(batch_psnrs)).item()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--device", default="cuda", help="Device to run on.", choices=["cpu", "cuda"]
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="dataset/",
        help="Dataset directory",
    )
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument(
        "--model-type",
        type=AutoEncoderModelType,
        default=AutoEncoderModelType.DEFAULT,
        choices=list(AutoEncoderModelType),
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default="logs/default.log",
        help="Log file name",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=32,
        help="Number of frames to extract from each video",
    )
    parser.add_argument("--height", type=int, default=256, help="Height of video")
    parser.add_argument("--width", type=int, default=256, help="Width of video")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch Size")
    args = parser.parse_args()

    logging.basicConfig(
        filename=args.log_file,
        filemode="w+",
        format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d-%H-%M",
        level=logging.INFO,
    )
    for arg in vars(args):
        logging.info(f"{arg}: {getattr(args, arg)}")

    main(args)
