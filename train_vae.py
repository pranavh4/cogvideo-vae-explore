# import argparse
import torch
import imageio
from diffusers import AutoencoderKLCogVideoX
from torchvision import transforms as t
import os
import random
# import numpy as np
device = "cuda"
dtype = torch.bfloat16

model = AutoencoderKLCogVideoX.from_pretrained('THUDM/CogVideoX-2b', subfolder="vae", torch_dtype=dtype).to(device)
model.enable_slicing()
model.enable_tiling()

#print(model)

dir_list = os.listdir('./dataset')
random.shuffle(dir_list)


for dir in dir_list[:10]:
    video_reader = imageio.get_reader(f'/scratch/phegde7/cogvideo-vae-explore/dataset/{dir}', "ffmpeg")
    transforms = [t.ToTensor(), t.Resize((256,256))]
    frame_transform = t.Compose(transforms)
    frames = [frame_transform(frame) for frame in video_reader][:36]
    print(frames[0].shape)
    video_reader.close()
    frames_tensor = torch.stack(frames).to(device).permute(1, 0, 2, 3).unsqueeze(0).to(dtype)

# with torch.no_grad():
#encoder_output = model.encode(frames_tensor)


#with torch.no_grad():
#    posterior = encoder_output.latent_dist
#    z = posterior.mode()
#    model.decode(z)
    model.forward(frames_tensor)
# i = 0
# for block in model.encoder.down_blocks:
#   print("Block", i)
#   j = 0
#   for downsampler in block.downsamplers:
#     print("Downsampler", j)
#     print(downsampler.compress_time)
#     j += 1
#   i += 1
