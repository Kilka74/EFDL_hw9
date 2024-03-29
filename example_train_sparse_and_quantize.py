from pathlib import Path
from tqdm.auto import tqdm

import torch
import torch.nn as nn

from sparseml.pytorch.optim import ScheduledModifierManager
from sparseml.pytorch.utils import export_onnx
from task1.model import Resnet101
from task1.dataset import download_data


def save_onnx(model, export_path, convert_qat):
    # It is important to call torch_model.eval() or torch_model.train(False) before exporting the model, to turn the model to inference mode.
    # This is required since operators like dropout or batchnorm behave differently in inference and training mode.
    model.eval()
    sample_batch = torch.randn((1, 3, 224, 224))
    export_onnx(model, sample_batch, export_path, convert_qat=convert_qat)


def main():
    # TODO: add argparse/hydra/... to manage hyperparameters like batch_size, path to pretrained model, etc

    # Sparsification recipe -- yaml file with instructions on how to sparsify the model
    recipe_path = "recipe.yaml"
    assert Path(recipe_path).exists(), "Didn't find sparsification recipe!"

    checkpoints_path = Path("checkpoints")
    checkpoints_path.mkdir(exist_ok=True)

    # Model creation
    # TODO: change to your best model from subtasks 1.1 - 1.3
    NUM_CLASSES = 10  # number of Imagenette classes
    model = Resnet101(num_classes=NUM_CLASSES, pretrained=False)
    model.load_state_dict(torch.load("distilled_with_mse.pth"), strict=False)

    save_onnx(model, checkpoints_path / "baseline_resnet.onnx", convert_qat=False)

    # Dataset creation
    # TODO: change to CIFAR10, add test dataset
    batch_size = 64
    train_loader, test_loader = download_data(batch_size=batch_size, num_workers=4)

    # Device setup
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Loss setup
    criterion = nn.CrossEntropyLoss()
    # Note that learning rate is being modified in `recipe.yaml`
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    # SparseML Integration
    manager = ScheduledModifierManager.from_yaml(recipe_path)
    optimizer = manager.modify(model, optimizer, steps_per_epoch=len(train_loader))

    # TODO: implement `train_one_epoch` function to structure the code better
    pbar = tqdm(range(manager.max_epochs), desc="epoch")
    for epoch in pbar:
        running_loss = 0.0
        running_corrects = 0.0
        model.train()
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                loss.backward()
                optimizer.step()

            running_loss += loss * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss.item() / len(train_loader.dataset)
        epoch_acc = running_corrects.double().item() / len(train_loader.dataset)
        pbar.set_description(f"Training loss: {epoch_loss:.3f}  Accuracy: {epoch_acc:.3f}")
        model.eval()
        with torch.no_grad():
            running_corrects = 0
            for inputs, labels in test_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                preds = torch.argmax(outputs, dim=-1)
                running_corrects += torch.sum(preds == labels.data)
            print(f"Test Accuracy: {running_corrects / len(test_loader.dataset)}")

    # TODO: implement `evaluate` function to measure accuracy on the test set

    manager.finalize(model)

    # Saving model
    save_onnx(model, checkpoints_path / "pruned_quantized_resnet.onnx", convert_qat=True)


if __name__ == "__main__":
    main()
