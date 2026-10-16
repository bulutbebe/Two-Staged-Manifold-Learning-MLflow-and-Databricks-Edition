"""Training loops separated from data loading and evaluation."""
import torch
from torch import nn


def train_autoencoder(model, dataloader, epochs: int, learning_rate: float,
                      device: torch.device, label: str = "model") -> list[float]:
    """Train an autoencoder and return mean training loss for each epoch."""
    if epochs < 1:
        raise ValueError("epochs must be at least 1")
    model.to(device)
    optimizer = torch.optim.Adam(
        (p for p in model.parameters() if p.requires_grad), lr=learning_rate
    )
    loss_fn = nn.MSELoss()
    history = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        n_samples = 0
        for batch in dataloader:
            # DataLoader may yield a tensor or a tuple/list.
            if isinstance(batch, (tuple, list)):
                batch = batch[0]
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            output = model(batch)
            reconstruction = output[0]
            loss = loss_fn(reconstruction, batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.shape[0]
            n_samples += batch.shape[0]

        epoch_loss = total_loss / max(n_samples, 1)
        history.append(epoch_loss)
        print(f"[{label}] epoch {epoch + 1}/{epochs}: loss={epoch_loss:.6f}")
    return history
