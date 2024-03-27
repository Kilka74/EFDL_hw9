import torch
import wandb
from tqdm import tqdm


def train_step(model, inputs, labels, optimizer, loss_fn, device):
    optimizer.zero_grad()
    inputs = inputs.to(device)
    labels = labels.to(device)
    preds = model(inputs)
    loss = loss_fn(preds, labels)
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
    model,
    optimizer,
    train_dataloader,
    test_dataloader,
    loss_fn,
    n_epochs,
    path="default",
    device="cuda",
    save=False,
    wandb_log=False,
    debug=False
):
    val_accs = [0, 0]
    print("start of train")
    for _ in range(n_epochs):
        train_epoch(
            model=model,
            optimizer=optimizer,
            dataloader=train_dataloader,
            loss_fn=loss_fn,
            device=device,
            wandb_log=wandb_log,
        )
        print("passed train epoch")
        acc = eval_epoch(
            model=model, dataloader=test_dataloader, device=device, wandb_log=wandb_log
        )
        print("passed eval epoch")
        if abs(acc - val_accs[-1]) < 0.01 and abs(val_accs[-1] - val_accs[-2]) < 0.01:
            val_accs.append(acc)
            break
        val_accs.append(acc)
    if save:
        save_checkpoint(model=model, path=path)


def save_checkpoint(model, path: str):
    state = {
        "unet": model.model.state_dict(),
    }
    filename = str("{}.pth".format(path))
    torch.save(state, filename)
