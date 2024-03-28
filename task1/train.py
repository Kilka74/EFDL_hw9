import torch
import wandb
from tqdm import tqdm
from torch import nn
from model import Resnet101
from dataset import download_data
from torch.nn.functional import softmax, log_softmax


def wandb_init(params):
    wandb.login(key=params["wandb_key"], relogin=True)
    wandb.init(
        entity="kilka74",
        project="EFDL_hw9",
        name=params["name"],
        config={
            "lr": params["lr"],
            "n_epochs": params["n_epochs"],
            "batch_size": params["batch_size"],
            "weight_decay": params["weight_decay"],
        },
    )


def train_step(model, inputs, labels, optimizer, loss_fn, device):
    optimizer.zero_grad()
    inputs = inputs.to(device)
    labels = labels.to(device)
    preds = model(inputs)
    loss = loss_fn(preds, labels)
    loss.backward()
    optimizer.step()
    return loss.item()


def train_step_distilled(
    teacher, student, inputs, labels, optimizer, loss_fn, device, add_mse
):
    optimizer.zero_grad()
    inputs = inputs.to(device)
    labels = labels.to(device)
    student_preds = student(inputs, return_temporary=add_mse)
    with torch.no_grad():
        teacher_preds = teacher(inputs, return_temporary=add_mse)
    if not add_mse:
        student_loss = loss_fn(student_preds, labels)
        distil_loss = loss_fn(log_softmax(student_preds, dim=-1), softmax(teacher_preds, dim=-1))
        loss = 0.75 * student_loss + 0.25 * distil_loss
    else:
        student_loss = loss_fn(student_preds[0], labels)
        distil_loss = loss_fn(log_softmax(student_preds[0], dim=-1), softmax(teacher_preds[0], dim=-1))
        mse = nn.MSELoss()
        loss = (
            0.5 * student_loss
            + 0.25 * distil_loss
            + 0.25 * 0.33 * (mse(student_preds[1], teacher_preds[1])
            + mse(student_preds[2], teacher_preds[2])
            + mse(student_preds[3], teacher_preds[3]))
        )

    loss.backward()
    optimizer.step()
    return loss.item()


def train_epoch(model, optimizer, dataloader, loss_fn, device, wandb_log):
    model.train()
    pbar = tqdm(dataloader)
    for inputs, labels in pbar:
        train_loss = train_step(
            model=model,
            inputs=inputs,
            labels=labels,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
        )
        if wandb_log:
            wandb.log({"loss": train_loss})
        pbar.set_description(f"Loss: {train_loss:.4f}")


def train_epoch_distilled(
    teacher, student, optimizer, dataloader, loss_fn, device, wandb_log, add_mse=False
):
    student.train()
    pbar = tqdm(dataloader)
    for inputs, labels in pbar:
        train_loss = train_step_distilled(
            teacher=teacher,
            student=student,
            inputs=inputs,
            labels=labels,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            add_mse=add_mse,
        )
        if wandb_log:
            wandb.log({"loss": train_loss})
        pbar.set_description(f"Loss: {train_loss:.4f}")


@torch.no_grad()
def eval_epoch(model, dataloader, device, wandb_log):
    model.eval()
    pbar = tqdm(dataloader)
    n_correct = 0
    n_all = 0
    for inputs, labels in pbar:
        inputs = inputs.to(device)
        labels = labels.to(device)
        preds = model(inputs)
        pred_labels = torch.argmax(preds, dim=-1)
        n_correct += (pred_labels.squeeze() == labels).sum()
        n_all += inputs.shape[0]
        pbar.set_description(f"Accuracy: {n_correct / n_all}")
    if wandb_log:
        wandb.log({"accuracy": n_correct / n_all})
    return n_correct / n_all


def train(
    device,
    params,
    path="default",
):
    model = Resnet101(num_classes=10, pretrained=False).to(device)

    train_dataloader, test_dataloader = download_data(params["batch_size"], params["num_workers"])

    optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
    loss_fn = nn.CrossEntropyLoss()

    if params["wandb_log"]:
        wandb_init(params)

    val_accs = [0, 0]
    for _ in range(params["n_epochs"]):
        train_epoch(
            model=model,
            optimizer=optimizer,
            dataloader=train_dataloader,
            loss_fn=loss_fn,
            device=device,
            wandb_log=params["wandb_log"],
        )
        acc = eval_epoch(
            model=model, dataloader=test_dataloader, device=device, wandb_log=params["wandb_log"]
        )
        if (abs(acc - val_accs[-1]) < 0.01) and (abs(val_accs[-1] - val_accs[-2]) < 0.01):
            print("stop")
            val_accs.append(acc)
            break
        val_accs.append(acc)
    if params["save"]:
        save_checkpoint(model=model, path=path)
    print("finish")
    wandb.finish()


def train_distilled(
    device,
    params,
    teacher_path,
    path="distilled",
    add_mse=False
):
    teacher_model = Resnet101(num_classes=10, pretrained=True).to(device)
    teacher_model.load_state_dict(torch.load(teacher_path), strict=False)
    teacher_model.eval()
    student_model = Resnet101(num_classes=10, pretrained=False).to(device)

    train_dataloader, test_dataloader = download_data(params["batch_size"], params["num_workers"])

    optimizer = torch.optim.Adam(
        student_model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"]
    )
    loss = nn.CrossEntropyLoss()
    wandb_init(params)

    val_accs = [0, 0]
    for _ in range(params["n_epochs"]):
        train_epoch_distilled(
            teacher=teacher_model,
            student=student_model,
            optimizer=optimizer,
            dataloader=train_dataloader,
            loss_fn=loss,
            device=device,
            wandb_log=params["wandb_log"],
            add_mse=add_mse
        )
        acc = eval_epoch(
            model=student_model,
            dataloader=test_dataloader,
            device=device,
            wandb_log=params["wandb_log"],
        )
        if abs(acc - val_accs[-1]) < 0.01 and abs(val_accs[-1] - val_accs[-2]) < 0.01:
            val_accs.append(acc)
            break
        val_accs.append(acc)
    if params["save"]:
        save_checkpoint(model=student_model, path=path)
    wandb.finish()


def save_checkpoint(model, path: str):
    state = {
        "resnet": model.model.state_dict(),
    }
    filename = str("{}.pth".format(path))
    torch.save(state, filename)
