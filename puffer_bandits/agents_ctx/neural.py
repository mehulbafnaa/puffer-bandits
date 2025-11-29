
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core.linear import sherman_morrison_update
from ..utils.linalg import safe_cholesky
from .base import CtxAgent, CtxAgentCfg
from ..utils.constants import TINY


class NeuralLinearTS(CtxAgent):
    def __init__(
        self,
        cfg: CtxAgentCfg,
        m: int = 64,
        hidden: int = 128,
        depth: int = 2,
        dropout: float = 0.1,
        lam: float = 1.0,
        v: float = 0.1,
        lr: float = 1e-3,
        amp: bool = True,
        use_compile: bool = True,
    ):
        super().__init__(cfg)
        if lam <= 0:
            raise ValueError("lam must be > 0")
        self.m = int(m)
        self.lam = float(lam)
        self.v = float(v)
        self.encoder = self._make_encoder(self.d, hidden, depth, dropout, out_dim=self.m).to(self.device)
        self.opt = torch.optim.Adam(self.encoder.parameters(), lr=lr)
        self.use_amp = bool(amp)
        # Optional torch.compile for the encoder
        self.use_compile = bool(use_compile)
        if self.use_compile and hasattr(torch, "compile") and self.device.type == "cuda":
            try:
                self.encoder = torch.compile(self.encoder, mode="reduce-overhead")  # type: ignore[attr-defined]
            except Exception:
                pass
        n, k, m = self.num_envs, self.k, self.m
        eye = torch.eye(m, device=self.device, dtype=torch.float32)
        self.A_inv = torch.stack([eye for _ in range(n * k)], dim=0).view(n, k, m, m) / self.lam
        self.b = torch.zeros((n, k, m), device=self.device, dtype=torch.float32)

    def _make_encoder(self, d: int, hidden: int, depth: int, dropout: float, out_dim: int) -> nn.Module:
        layers = []
        in_dim = d
        for _ in range(depth):
            layers += [nn.Linear(in_dim, hidden), nn.ReLU()]
            if dropout > 0:
                layers += [nn.Dropout(dropout)]
            in_dim = hidden
        layers += [nn.Linear(in_dim, out_dim)]
        return nn.Sequential(*layers)

    def reset(self) -> None:
        for p in self.encoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.zeros_(p)
        m = self.m
        eye = torch.eye(m, device=self.device, dtype=torch.float32)
        self.A_inv[...] = eye / self.lam
        self.b.zero_()

    @torch.no_grad()
    def select_actions(self, t: int, obs: torch.Tensor) -> torch.LongTensor:
        n, k, d = obs.shape
        obs_flat = obs.reshape(-1, d)
        if self.use_amp and self.device.type in ("cuda", "mps"):
            dtype = torch.float16 if self.device.type == "mps" else torch.float16
            with torch.autocast(self.device.type, dtype=dtype):
                z_flat = self.encoder(obs_flat)
        else:
            z_flat = self.encoder(obs_flat)
        z = z_flat.reshape(n, k, self.m)
        mu = torch.matmul(self.A_inv, self.b.unsqueeze(-1)).squeeze(-1)
        L = safe_cholesky(self.A_inv)
        zeta = torch.randn((n, k, self.m), device=self.device, generator=self.rng)
        theta = mu + self.v * torch.matmul(L, zeta.unsqueeze(-1)).squeeze(-1)
        scores = (z * theta).sum(dim=-1)
        scores = scores + TINY * torch.rand(scores.shape, device=self.device, generator=self.rng)
        return scores.argmax(dim=1).long()

    def update(self, actions: torch.LongTensor, rewards: torch.Tensor, obs: torch.Tensor) -> None:
        a = actions.long().view(-1)
        r = rewards.to(self.device, dtype=torch.float32).view(-1)
        n, k, d = obs.shape
        batch = torch.arange(self.num_envs, device=self.device)
        obs_flat = obs.reshape(-1, d)
        if self.use_amp and self.device.type in ("cuda", "mps"):
            dtype = torch.float16 if self.device.type == "mps" else torch.float16
            with torch.autocast(self.device.type, dtype=dtype):
                z_flat = self.encoder(obs_flat)
        else:
            z_flat = self.encoder(obs_flat)
        z = z_flat.reshape(n, k, self.m)
        z_sel = z[batch, a, :]
        Ainv_sel = self.A_inv[batch, a, :, :]
        Ainv_new = sherman_morrison_update(Ainv_sel, z_sel)
        self.A_inv[batch, a, :, :] = Ainv_new
        self.b[batch, a, :] = self.b[batch, a, :] + (r.view(-1, 1) * z_sel)
        with torch.no_grad():
            mu = torch.matmul(Ainv_new, self.b[batch, a, :].unsqueeze(-1)).squeeze(-1)
        logits = (mu * z_sel).sum(dim=-1)
        loss = F.binary_cross_entropy_with_logits(logits, r)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()


class NeuralTS(CtxAgent):
    def __init__(
        self,
        cfg: CtxAgentCfg,
        hidden: int = 128,
        depth: int = 2,
        ensembles: int = 5,
        dropout: float = 0.1,
        lr: float = 1e-3,
        amp: bool = True,
        use_compile: bool = True,
    ):
        super().__init__(cfg)
        assert ensembles >= 1
        self.B = int(ensembles)
        self.dropout_p = float(dropout)
        self.models = torch.nn.ModuleList([
            self._make_mlp(self.d, hidden, depth, self.dropout_p).to(self.device) for _ in range(self.B)
        ])
        self.optims = [torch.optim.Adam(m.parameters(), lr=lr) for m in self.models]
        self._last_heads: torch.Tensor | None = None
        self.use_amp = bool(amp)
        self.use_compile = bool(use_compile)
        if self.use_compile and hasattr(torch, "compile") and self.device.type == "cuda":
            try:
                self.models = torch.nn.ModuleList([
                    torch.compile(m, mode="reduce-overhead")  # type: ignore[attr-defined]
                    for m in self.models
                ])
            except Exception:
                pass

    def _make_mlp(self, d: int, hidden: int, depth: int, dropout: float) -> torch.nn.Module:
        layers = []
        in_dim = d
        for _ in range(depth):
            layers += [torch.nn.Linear(in_dim, hidden), torch.nn.ReLU()]
            if dropout > 0:
                layers += [torch.nn.Dropout(dropout)]
            in_dim = hidden
        layers += [torch.nn.Linear(in_dim, 1)]
        return torch.nn.Sequential(*layers)

    def reset(self) -> None:
        for m in self.models:
            for p in m.parameters():
                if p.dim() > 1:
                    torch.nn.init.xavier_uniform_(p)
                else:
                    torch.nn.init.zeros_(p)
        self._last_heads = None

    @torch.no_grad()
    def select_actions(self, t: int, obs: torch.Tensor) -> torch.LongTensor:
        n, k, d = obs.shape
        heads = torch.randint(self.B, size=(n,), device=self.device, generator=self.rng)
        self._last_heads = heads
        # Batched across heads: compute logits for all heads over all rows once
        P = []  # (B,n,k)
        obs_flat_all = obs.reshape(-1, d)
        if self.use_amp and self.device.type in ("cuda", "mps"):
            dtype = torch.float16 if self.device.type == "mps" else torch.float16
            with torch.autocast(self.device.type, dtype=dtype):
                for b in range(self.B):
                    logits = self.models[b](obs_flat_all).reshape(n, k)
                    P.append(logits)
        else:
            for b in range(self.B):
                logits = self.models[b](obs_flat_all).reshape(n, k)
                P.append(logits)
        P = torch.stack(P, dim=0)  # (B,n,k)
        # Select per-row scores from the assigned head
        scores = P.permute(1, 0, 2)  # (n,B,k)
        idx = heads.view(n, 1, 1).expand(n, 1, k)
        scores = scores.gather(1, idx).squeeze(1)  # (n,k)
        scores = scores + TINY * torch.rand(scores.shape, device=self.device, generator=self.rng)
        return scores.argmax(dim=1).long()

    def update(self, actions: torch.LongTensor, rewards: torch.Tensor, obs: torch.Tensor) -> None:
        assert self._last_heads is not None, "select_actions must be called before update"
        a = actions.long().view(-1)
        r = rewards.to(self.device, dtype=torch.float32).view(-1)
        heads = self._last_heads
        n, k, d = obs.shape
        batch_idx = torch.arange(n, device=self.device)
        x_sel = obs[batch_idx, a, :]
        for b in range(self.B):
            idx = (heads == b)
            if not torch.any(idx):
                continue
            xb = x_sel[idx]
            rb = r[idx]
            model = self.models[b]
            optim = self.optims[b]
            model.train()
            optim.zero_grad(set_to_none=True)
            logits = model(xb).squeeze(-1)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, rb)
            loss.backward()
            optim.step()
