import argparse
import logging
import time

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from dataset import OpenVideoDataset
from loss import AutoEncoderLoss
from models import AutoEncoderModelType, get_uninitialized_model

DTYPE = torch.bfloat16
SEED = 42


def main(args):
    dataset = OpenVideoDataset(
        args.dataset_dir,
        max_frames=args.num_frames,
        resolution=(args.width, args.height),
    )
    generator = torch.Generator()
    generator.manual_seed(SEED)
    train_set, test_set = torch.utils.data.random_split(dataset, [0.8, 0.2], generator)

    train_dataloader = DataLoader(train_set, shuffle=True, batch_size=args.batch_size)
    model = get_uninitialized_model(AutoEncoderModelType.DEFAULT).to(args.device)
    optimizer = AdamW(
        model.parameters(), weight_decay=1e-4, eps=1e-8, betas=(0.9, 0.95)
    )

    loss_fn = AutoEncoderLoss()
    start_time = time.time()

    for epoch in range(args.num_epochs):
        for i, batch in enumerate(train_dataloader):
            if args.device == "cuda":
                torch.cuda.empty_cache()

            optimizer.zero_grad()
            batch = batch.to(args.device)
            latent_dist = model.encode(batch).latent_dist
            z = latent_dist.sample()
            output = model.decode(z).sample
            loss = loss_fn(output, batch, latent_dist)
            logging.info(f"EPOCH: {epoch}, BATCH: {i}, LOSS: {loss[0]}")
            loss.backward()
            optimizer.step()

        model.save_pretrained(f"{args.model_save_dir}/epoch_{epoch + 1}", from_pt=True)

    end_time = time.time()

    logging.info("Total time: %d sec" % (end_time - start_time))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--device", default="cuda", help="Device to run on.", choices=["cpu", "cuda"]
    )
    parser.add_argument(
        "--num-epochs", type=int, default=1, help="Number of epochs to train"
    )
    parser.add_argument(
        "--model-save-dir",
        type=str,
        default="models/",
        help="Directory to save model.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="dataset/",
        help="Dataset directory",
    )
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
