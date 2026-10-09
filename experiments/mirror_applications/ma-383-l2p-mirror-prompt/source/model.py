"""Prompt-value pool variants and task-identity-free nearest-key retrieval."""
from __future__ import annotations

import torch

INPUT_DIM = 8
PROMPT_DIM = 64
KEY_DIM = 8
PROMPTS = 8
RANK = 2


class PromptBank(torch.nn.Module):
    METHODS = ("independent", "tied", "scalar", "coeff", "mirror")

    def __init__(self, method: str, seed: int):
        super().__init__()
        if method not in self.METHODS:
            raise ValueError(method)
        self.method = method
        torch.manual_seed(seed)
        if method == "independent":
            self.prompt = torch.nn.Parameter(0.1 * torch.randn(PROMPTS, PROMPT_DIM))
        elif method in ("tied", "scalar"):
            self.prompt = torch.nn.Parameter(0.1 * torch.randn(PROMPT_DIM))
            if method == "scalar":
                self.code = torch.nn.Parameter(torch.ones(PROMPTS))
        else:
            self.basis = torch.nn.Parameter(0.1 * torch.randn(PROMPT_DIM, RANK))
            if method == "coeff":
                self.code = torch.nn.Parameter(torch.randn(PROMPTS, RANK))
            else:
                self.angle = torch.nn.Parameter(torch.zeros(PROMPTS, 1))

    def prompt_values(self) -> torch.Tensor:
        if self.method == "independent":
            return self.prompt
        if self.method == "tied":
            return self.prompt[None, :].expand(PROMPTS, -1)
        if self.method == "scalar":
            return self.prompt[None, :] * self.code[:, None]
        if self.method == "coeff":
            return self.code @ self.basis.T
        coords = torch.cat((torch.cos(self.angle), torch.sin(self.angle)), dim=-1)
        return coords @ self.basis.T

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply all prompt-conditioned linear residual decoders to x."""
        matrices = self.prompt_values().reshape(PROMPTS, INPUT_DIM, INPUT_DIM)
        return torch.einsum("bd,ndk->bnk", x, matrices.transpose(1, 2))

    def forward_selected(self, x: torch.Tensor, prompt_ids: torch.Tensor) -> torch.Tensor:
        """Apply only the retrieved prompt for each query example."""
        matrices = self.prompt_values()[prompt_ids].reshape(-1, INPUT_DIM, INPUT_DIM)
        return torch.einsum("bd,bdk->bk", x, matrices.transpose(1, 2))

    def payload_arrays(self, keys: torch.Tensor) -> dict[str, object]:
        arrays: dict[str, object] = {
            "method": self.method,
            "input_dim": INPUT_DIM,
            "prompt_dim": PROMPT_DIM,
            "key_dim": KEY_DIM,
            "prompt_count": PROMPTS,
            "retrieval_keys": keys.detach().cpu().numpy().astype("<f4"),
        }
        for name, value in self.state_dict().items():
            arrays[f"bank_{name}"] = value.detach().cpu().numpy().astype("<f4")
        return arrays


def bank_from_payload(payload: dict[str, object]) -> PromptBank:
    bank = PromptBank(str(payload["method"]), 0).cpu()
    state = {k.removeprefix("bank_"): torch.as_tensor(v, dtype=torch.float32)
             for k, v in payload.items() if k.startswith("bank_")}
    bank.load_state_dict(state)
    return bank.eval()


def nearest_key(query: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
    """Cosine-style retrieval; stored keys are normalized/orthonormal per world."""
    return torch.argmax(query @ keys.T, dim=-1)


def select_prompt_outputs(all_outputs: torch.Tensor, prompt_ids: torch.Tensor) -> torch.Tensor:
    """Gather the selected prompt output for each query example."""
    return all_outputs.gather(1, prompt_ids[:, None, None].expand(-1, 1, all_outputs.shape[-1])).squeeze(1)


def reconstruction_mac_proxy(method: str) -> int:
    if method in ("independent", "tied"):
        return 0
    if method == "scalar":
        return PROMPTS * PROMPT_DIM
    return PROMPTS * PROMPT_DIM * RANK


def retrieval_mac_proxy() -> int:
    return KEY_DIM * PROMPTS


def selected_prompt_mac_proxy() -> int:
    return INPUT_DIM * INPUT_DIM
