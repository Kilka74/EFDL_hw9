import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from model import Resnet101

# import wandb
import hydra
from train import train
from torch import nn as nn


@hydra.main(config_path="conf", config_name="config")
def main(cfg):
    model = Resnet101(num_classes=10, pretrained=True).to(cfg.device)
    # wandb.login(key="", relogin=True)
    # wandb.init(
    #     entity="kilka74",
    #     project="EFDL_hw9",
    #     name=cfg.name,
    #     config={
    #         "lr": cfg.optimizer.lr,
    #         "n_epochs": cfg.n_epochs,
    #     },
    # )
    # artifact = wandb.Artifact(
    #     name="run_config",
    #     type="config",
    # )
    # artifact.add_file(f"{hydra.core.hydra_config.HydraConfig.get().runtime.output_dir}/.hydra/config.yaml")
    # wandb.log_artifact(
    #     artifact
    # )
    train_transforms = transforms.Compose([transforms.ToTensor()])

    train_dataset = CIFAR10(
        "cifar10",
        train=True,
        download=True,
        transform=train_transforms,
    )

    test_dataset = CIFAR10(
        "cifar10",
        train=False,
        download=True,
        transform=train_transforms,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        shuffle=True,
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        shuffle=False,
    )

    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    loss_fn = nn.CrossEntropyLoss()
    train(
        model,
        optimizer,
        train_dataloader,
        test_dataloader,
        loss_fn=loss_fn,
        n_epochs=cfg.n_epochs,
        device=cfg.device,
        debug=True,
    )
    # wandb.finish()


if __name__ == "__main__":
    main()
