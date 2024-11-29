import os

import imageio.v3 as iio
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms as t


class OpenVideoDataset(Dataset):
    def __init__(
        self,
        data_directory: str,
        max_frames: int = 36,
        resolution: tuple[int, int] = (256, 256),
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.dtype = dtype
        self.max_frames = max_frames
        self.transform = t.Compose([t.ToTensor(), t.Resize(resolution)])
        data_directory_abs_path = os.path.abspath(data_directory)
        self.video_paths = [
            os.path.join(data_directory_abs_path, file)
            for file in os.listdir(data_directory_abs_path)
        ]

    def __len__(self) -> int:
        return len(self.video_paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        frames = []
        for ind, frame in enumerate(iio.imiter(self.video_paths[index], plugin="pyav")):
            if ind == self.max_frames:
                break

            frames.append(self.transform(frame))

        return torch.stack(frames).permute(1, 0, 2, 3).type(self.dtype)
