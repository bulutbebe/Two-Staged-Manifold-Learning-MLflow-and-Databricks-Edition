"""Neural-network definitions extracted from the research script."""
import torch
from torch import nn


class AEHankelRound1(nn.Module):
    """First autoencoder, trained on nominal samples."""
    def __init__(self, input_dim: int, latent_dim: int = 50, hidden_dim: int = 1024):
        super().__init__()
        if not 0 < latent_dim < input_dim:
            raise ValueError("latent_dim must be positive and smaller than input_dim")
        self.enc1 = nn.Linear(input_dim, hidden_dim)
        self.enc2 = nn.Linear(hidden_dim, latent_dim)
        self.dec1 = nn.Linear(latent_dim, hidden_dim)
        self.dec2 = nn.Linear(hidden_dim, input_dim)
        self.act = nn.Tanh()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.enc2(self.act(self.enc1(x)))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.dec2(self.act(self.dec1(z)))

    def forward(self, x: torch.Tensor):
        z = self.encode(x)
        return self.decode(z), z


class AEHankelRound2(nn.Module):
    """Second-stage model with a frozen first-stage encoder and decoder."""
    def __init__(self, ae1: AEHankelRound1, latent_dim: int, input_dim: int, hidden_dim: int):
        super().__init__()
        self.act = nn.Tanh()
        self.enc1 = ae1.enc1
        self.enc2 = ae1.enc2
        for parameter in self.enc1.parameters():
            parameter.requires_grad = False
        for parameter in self.enc2.parameters():
            parameter.requires_grad = False

        self.comp_head = nn.Linear(hidden_dim, input_dim - latent_dim)
        self.dec1 = nn.Linear(input_dim, hidden_dim)
        self.dec2 = ae1.dec2

        with torch.no_grad():
            self.dec1.weight[:, :latent_dim].copy_(ae1.dec1.weight)
            self.dec1.bias.copy_(ae1.dec1.bias)

        self.dec1.bias.requires_grad = False
        for parameter in self.dec2.parameters():
            parameter.requires_grad = False

        # Prevent updates to the copied first latent_dim columns.
        self.dec1.weight.register_hook(self._freeze_first_columns(latent_dim))

    @staticmethod
    def _freeze_first_columns(latent_dim: int):
        def hook(grad: torch.Tensor) -> torch.Tensor:
            grad = grad.clone()
            grad[:, :latent_dim] = 0
            return grad
        return hook

    def forward(self, x: torch.Tensor):
        h = self.act(self.enc1(x))
        z_characteristic = self.enc2(h)
        z_complementary = self.comp_head(h)
        z_full = torch.cat([z_characteristic, z_complementary], dim=1)
        reconstruction = self.dec2(self.act(self.dec1(z_full)))
        return reconstruction, z_characteristic, z_complementary
