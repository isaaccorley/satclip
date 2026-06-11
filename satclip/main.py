import os
from datetime import datetime
from pathlib import Path

import lightning.pytorch
import torch
from datamodules.s2geo_dataset import S2GeoDataModule
from lightning.pytorch.cli import LightningCLI
from loss import SatCLIPLoss, SoftSatCLIPLoss
from model import SatCLIP

torch.set_float32_matmul_precision('high')

class SatCLIPLightningModule(lightning.pytorch.LightningModule):
    def __init__(
        self,
        loss_type="satclip_loss", # "satclip_loss" or "soft_loss"
        soft_loss_penalty="linear", # distance->weight curve: "linear" | "exponential" | "sigmoid"
        soft_loss_rho_km=1.0, # length-scale (km) for the soft_loss penalty curve
        soft_loss_tau_km=100.0, # sigmoid width (km), only used when penalty="sigmoid"
        embed_dim=512,
        image_resolution=256,
        vision_layers=12,
        vision_width=768,
        vision_patch_size=32,
        in_channels=4,
        le_type="grid",
        pe_type="siren",
        frequency_num=16,
        max_radius=260,
        min_radius=1,
        legendre_polys=16,
        harmonics_calculation="analytic",
        sh_embedding_dims=32,
        learning_rate=1e-4,
        weight_decay=0.01,
        num_hidden_layers=2,
        capacity=256,
    ) -> None:
        super().__init__()

        self.model = SatCLIP(
            loss_type=loss_type,
            soft_loss_penalty=soft_loss_penalty,
            soft_loss_rho_km=soft_loss_rho_km,
            soft_loss_tau_km=soft_loss_tau_km,
            embed_dim=embed_dim,
            image_resolution=image_resolution,
            vision_layers=vision_layers,
            vision_width=vision_width,
            vision_patch_size=vision_patch_size,
            in_channels=in_channels,
            le_type=le_type,
            pe_type=pe_type,
            frequency_num=frequency_num,
            max_radius=max_radius,
            min_radius=min_radius,
            legendre_polys=legendre_polys,
            harmonics_calculation=harmonics_calculation,
            sh_embedding_dims=sh_embedding_dims,
            num_hidden_layers=num_hidden_layers,
            capacity=capacity,
        )

        if loss_type == "soft_loss":
            self.loss_type = "soft_loss"
            self.loss_fun = SoftSatCLIPLoss()
        elif loss_type == "satclip_loss":
            self.loss_type = "satclip_loss"
            self.loss_fun = SatCLIPLoss()
        print(f"using loss function: {loss_type}")

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.save_hyperparameters()

    def on_after_batch_transfer(self, batch, dataloader_idx):
        # GPU-side image transform for the `transform: gpu` dataloader path: workers
        # hand off raw uint16 (12-band) to minimise PCIe traffic, and we do the float
        # cast, /10000 scaling, B10 zero-band insertion and augmentation here on-device.
        # Math is identical to get_pretrained_s2_train_transform (verified); a float
        # batch (other transforms) passes through untouched.
        img = batch.get("image") if isinstance(batch, dict) else None
        if img is not None and img.dtype == torch.uint16:
            x = img.float().div_(10000.0)
            # insert an all-zero B10 band at index 10 (12 -> 13 bands)
            z = torch.zeros((x.shape[0], 1, *x.shape[2:]), device=x.device, dtype=x.dtype)
            x = torch.cat([x[:, :10], z, x[:, 10:]], dim=1)
            x = self._gpu_augment(x)
            batch["image"] = x
        return batch

    def _gpu_augment(self, x):
        # RandomCrop(256) is a no-op at the native 256 resolution; then per-sample
        # H/V flips (p=0.5 each) and GaussianBlur(3, sigma~U(0.1,2.0)) matching
        # torchvision T.GaussianBlur(3).
        B, dev = x.shape[0], x.device
        hflip = torch.rand(B, device=dev) < 0.5
        if hflip.any():
            x[hflip] = x[hflip].flip(-1)
        vflip = torch.rand(B, device=dev) < 0.5
        if vflip.any():
            x[vflip] = x[vflip].flip(-2)
        sigma = torch.empty(B, device=dev).uniform_(0.1, 2.0)
        return self._gaussian_blur(x, kernel_size=3, sigma=sigma)

    @staticmethod
    def _gaussian_blur(img, kernel_size, sigma):
        # Separable depthwise blur with a per-sample kernel (matches torchvision's
        # per-sample GaussianBlur to ~1e-7; reflect padding as torchvision uses).
        B, C, H, W = img.shape
        half = (kernel_size - 1) / 2
        coords = torch.arange(kernel_size, device=sigma.device, dtype=torch.float32) - half
        k1 = torch.exp(-0.5 * (coords[None, :] / sigma[:, None]) ** 2)
        k1 = k1 / k1.sum(dim=1, keepdim=True)  # [B, k]
        pad = kernel_size // 2
        kx = k1.view(B, 1, 1, kernel_size).expand(B, C, 1, kernel_size).reshape(B * C, 1, 1, kernel_size)
        ky = k1.view(B, 1, kernel_size, 1).expand(B, C, kernel_size, 1).reshape(B * C, 1, kernel_size, 1)
        xr = img.reshape(1, B * C, H, W)
        xr = torch.nn.functional.conv2d(torch.nn.functional.pad(xr, (pad, pad, 0, 0), mode="reflect"), kx, groups=B * C)
        xr = torch.nn.functional.conv2d(torch.nn.functional.pad(xr, (0, 0, pad, pad), mode="reflect"), ky, groups=B * C)
        return xr.reshape(B, C, H, W)

    def common_step(self, batch, batch_idx):
        images = batch["image"]
        t_points = batch["point"].float()

        if self.loss_type == "soft_loss":
            logits_per_image, logits_per_coord, weights = self.model(images, t_points)
            loss = self.loss_fun(logits_per_image, logits_per_coord, weights)
        else:
            logits_per_image, logits_per_coord = self.model(images, t_points)
            loss = self.loss_fun(logits_per_image, logits_per_coord)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        exclude = (
            lambda n, p: p.ndim < 2
            or "bn" in n
            or "ln" in n
            or "bias" in n
            or "logit_scale" in n
        )
        include = lambda n, p: not exclude(n, p)

        named_parameters = list(self.model.named_parameters())
        gain_or_bias_params = [
            p for n, p in named_parameters if exclude(n, p) and p.requires_grad
        ]
        rest_params = [
            p for n, p in named_parameters if include(n, p) and p.requires_grad
        ]

        optimizer = torch.optim.AdamW(
            [
                {"params": gain_or_bias_params, "weight_decay": 0.0},
                {
                    "params": rest_params,
                    "weight_decay": self.weight_decay,
                },  # specify in configs/default.yaml
            ],
            lr=self.learning_rate,  # specify in configs/default.yaml
        )

        return optimizer


class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_argument("--watchmodel", action="store_true")


def cli_main(default_config_filename="./configs/default.yaml"):
    save_config_fn = default_config_filename.replace(".yaml", "-latest.yaml")
    # modify configs/default.yaml for learning rate etc.
    cli = MyLightningCLI(
        model_class=SatCLIPLightningModule,
        datamodule_class=S2GeoDataModule,
        save_config_kwargs=dict(
            config_filename=save_config_fn,
            overwrite=True,
        ),
        trainer_defaults={
            "accumulate_grad_batches": 16,
            "log_every_n_steps": 10,
        },
        parser_kwargs={"default_config_files": [default_config_filename]},
        seed_everything_default=0,
        run=False,
    )

    ts = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    run_name = f"SatCLIP_{cli.model.loss_type}_{ts}"
    if cli.trainer.logger is not None:
        cli.trainer.logger.experiment.name = run_name
        # this seems to be necessary to force logging of datamodule hyperparams
        cli.trainer.logger.log_hyperparams(cli.datamodule.hparams)

    # Create folder to log configs
    # NOTE: Lightning does not handle config paths with subfolders
    dirname_cfg = Path(default_config_filename).parent
    dir_log_cfg = Path(cli.trainer.log_dir) / dirname_cfg
    dir_log_cfg.mkdir(parents=True, exist_ok=True)

    cli.trainer.fit(
        model=cli.model,
        datamodule=cli.datamodule,
    )


if __name__ == "__main__":
    # Select config via SATCLIP_CONFIG env var (set by the slurm script) so we
    # can swap baseline.yaml / softloss.yaml without touching argv parsing.
    config_fn = os.environ.get("SATCLIP_CONFIG", "./configs/default.yaml")

    #A100 go vroom vroom 🚗💨
    # if torch.cuda.get_device_name(device=0)=='NVIDIA A100 80GB PCIe':
    #     torch.backends.cuda.matmul.allow_tf32 = True
    #     print('Superfastmode! 🚀')
    # else:
    #     torch.backends.cuda.matmul.allow_tf32 = False
    cli_main(config_fn)