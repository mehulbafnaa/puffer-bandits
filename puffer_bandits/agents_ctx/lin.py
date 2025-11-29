
import torch
from ..core.linear import sherman_morrison_update
from ..utils.constants import TINY
from ..utils.linalg import safe_cholesky
from .base import CtxAgent, CtxAgentCfg


class LinUCB(CtxAgent):
    def __init__(self, cfg: CtxAgentCfg, alpha: float = 1.0, lam: float = 1.0):
        super().__init__(cfg)
        if lam <= 0:
            raise ValueError("lam must be > 0")
        self.alpha = float(alpha)
        self.lam = float(lam)
        n, k, d = self.num_envs, self.k, self.d
        eye = torch.eye(d, device=self.device, dtype=torch.float32)
        self.A_inv = torch.stack([eye for _ in range(n * k)], dim=0).view(n, k, d, d) / self.lam
        self.b = torch.zeros((n, k, d), device=self.device, dtype=torch.float32)

    def reset(self) -> None:
        d = self.d
        eye = torch.eye(d, device=self.device, dtype=torch.float32)
        self.A_inv[...] = eye / self.lam
        self.b.zero_()

    @torch.no_grad()
    def select_actions(self, t: int, obs: torch.Tensor) -> torch.LongTensor:
        X = obs.to(self.device, dtype=torch.float32)
        theta = torch.matmul(self.A_inv, self.b.unsqueeze(-1)).squeeze(-1)  # (n,k,d)
        mean = (X * theta).sum(dim=-1)
        Ainv_x = torch.matmul(self.A_inv, X.unsqueeze(-1)).squeeze(-1)
        conf2 = (X * Ainv_x).sum(dim=-1).clamp_min(1e-12)
        score = mean + self.alpha * torch.sqrt(conf2)
        score = score + TINY * torch.rand(score.shape, device=self.device, generator=self.rng)
        return score.argmax(dim=1).long()

    @torch.no_grad()
    def update(self, actions: torch.LongTensor, rewards: torch.Tensor, obs: torch.Tensor) -> None:
        idx = actions.long()
        r = rewards.to(self.device, dtype=torch.float32).view(-1, 1)
        X = obs.to(self.device, dtype=torch.float32)
        batch = torch.arange(self.num_envs, device=self.device)
        x_sel = X[batch, idx, :]
        Ainv_sel = self.A_inv[batch, idx, :, :]
        Ainv_new = sherman_morrison_update(Ainv_sel, x_sel)
        self.A_inv[batch, idx, :, :] = Ainv_new
        self.b[batch, idx, :] = self.b[batch, idx, :] + (r * x_sel)


class LinTS(CtxAgent):
    def __init__(self, cfg: CtxAgentCfg, v: float = 0.1, lam: float = 1.0):
        super().__init__(cfg)
        if lam <= 0:
            raise ValueError("lam must be > 0")
        self.v = float(v)
        self.lam = float(lam)
        n, k, d = self.num_envs, self.k, self.d
        eye = torch.eye(d, device=self.device, dtype=torch.float32)
        self.A_inv = torch.stack([eye for _ in range(n * k)], dim=0).view(n, k, d, d) / self.lam
        self.b = torch.zeros((n, k, d), device=self.device, dtype=torch.float32)

    def reset(self) -> None:
        d = self.d
        eye = torch.eye(d, device=self.device, dtype=torch.float32)
        self.A_inv[...] = eye / self.lam
        self.b.zero_()

    @torch.no_grad()
    def select_actions(self, t: int, obs: torch.Tensor) -> torch.LongTensor:
        X = obs.to(self.device, dtype=torch.float32)
        mu = torch.matmul(self.A_inv, self.b.unsqueeze(-1)).squeeze(-1)
        L = safe_cholesky(self.A_inv)
        z = torch.randn((self.num_envs, self.k, self.d), device=self.device, generator=self.rng)
        theta = mu + self.v * torch.matmul(L, z.unsqueeze(-1)).squeeze(-1)
        score = (X * theta).sum(dim=-1)
        score = score + TINY * torch.rand(score.shape, device=self.device, generator=self.rng)
        return score.argmax(dim=1).long()

    @torch.no_grad()
    def update(self, actions: torch.LongTensor, rewards: torch.Tensor, obs: torch.Tensor) -> None:
        idx = actions.long()
        r = rewards.to(self.device, dtype=torch.float32).view(-1, 1)
        X = obs.to(self.device, dtype=torch.float32)
        batch = torch.arange(self.num_envs, device=self.device)
        x_sel = X[batch, idx, :]
        Ainv_sel = self.A_inv[batch, idx, :, :]
        Ainv_new = sherman_morrison_update(Ainv_sel, x_sel)
        self.A_inv[batch, idx, :, :] = Ainv_new
        self.b[batch, idx, :] = self.b[batch, idx, :] + (r * x_sel)
