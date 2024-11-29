# import lpips
import torch
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution


class AutoEncoderLoss:
    def __init__(self, l1_weight=1, lpips_weight=1, kl_weight=1) -> None:
        self.l1_weight = l1_weight
        self.lpips_weight = lpips_weight
        self.kl_weight = kl_weight
        # self.loss_fn_lpips = lpips.LPIPS(net="alex")

    def __call__(
        self,
        output: torch.Tensor,
        target: torch.Tensor,
        latent_dist: DiagonalGaussianDistribution,
    ):
        print("TORCH L1", torch.nn.L1Loss()(output, target))
        l1_loss = torch.mean(((output - target).abs()).reshape(target.shape[0], -1), 1)
        print("L1 shape", l1_loss.shape)
        # lpips_loss = self.loss_fn_lpips(output, target)
        kl_loss = latent_dist.kl()
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]

        return (
            self.l1_weight * l1_loss
            # + self.lpips_weight * lpips_loss
            + self.kl_weight * kl_loss
        )
