from torchvision import transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader


def download_data(batch_size, num_workers):
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
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
    )
    return train_dataloader, test_dataloader
