"""DualPrompt-style common general prompt plus task expert prompt bank."""
from __future__ import annotations

import torch

INPUT_DIM = 8
PROMPT_DIM = 64
TASKS = 8
RANK = 2


class ExpertPromptBank(torch.nn.Module):
    METHODS = ("independent", "tied", "scalar", "coeff", "mirror")

    def __init__(self, method: str, seed: int, general_prompt: torch.Tensor):
        super().__init__()
        if method not in self.METHODS:
            raise ValueError(method)
        self.method = method
        torch.manual_seed(seed)
        self.register_buffer("general", general_prompt.detach().clone().float())
        if method == "independent":
            self.expert = torch.nn.Parameter(0.1 * torch.randn(TASKS, PROMPT_DIM))
        elif method == "tied":
            self.expert = torch.nn.Parameter(0.1 * torch.randn(PROMPT_DIM))
        elif method == "scalar":
            self.expert = torch.nn.Parameter(0.1 * torch.randn(PROMPT_DIM))
            self.code = torch.nn.Parameter(torch.ones(TASKS))
        else:
            self.basis = torch.nn.Parameter(0.1 * torch.randn(PROMPT_DIM, RANK))
            if method == "coeff":
                self.code = torch.nn.Parameter(torch.zeros(TASKS, RANK))
            else:
                self.angle = torch.nn.Parameter(torch.zeros(TASKS, 1))

    def expert_values(self) -> torch.Tensor:
        if self.method == "independent":
            return self.expert
        if self.method == "tied":
            return self.expert[None, :].expand(TASKS, -1)
        if self.method == "scalar":
            return self.expert[None, :] * self.code[:, None]
        if self.method == "coeff":
            return self.code @ self.basis.T
        coords = torch.cat((torch.cos(self.angle), torch.sin(self.angle)), dim=-1)
        return coords @ self.basis.T

    def forward_all(self, x: torch.Tensor) -> torch.Tensor:
        expert = self.expert_values().reshape(TASKS, INPUT_DIM, INPUT_DIM)
        general = self.general.reshape(INPUT_DIM, INPUT_DIM)
        y = torch.einsum("bd,ndk->bnk", x, expert.transpose(1, 2))
        common = x @ general.T
        return y + common[:, None, :]

    def forward_task(self, x: torch.Tensor, task_ids: torch.Tensor) -> torch.Tensor:
        experts = self.expert_values()[task_ids].reshape(-1, INPUT_DIM, INPUT_DIM)
        general = self.general.reshape(INPUT_DIM, INPUT_DIM)
        return torch.einsum("bd,bdk->bk", x, experts.transpose(1, 2)) + x @ general.T

    def payload_arrays(self) -> dict[str, object]:
        arrays: dict[str, object] = {
            "method": self.method,
            "input_dim": INPUT_DIM,
            "prompt_dim": PROMPT_DIM,
            "task_count": TASKS,
            "general_prompt": self.general.detach().cpu().numpy().astype("<f4"),
        }
        for name, value in self.state_dict().items():
            if name != "general":
                arrays[f"expert_{name}"] = value.detach().cpu().numpy().astype("<f4")
        return arrays


def bank_from_payload(payload: dict[str, object]) -> ExpertPromptBank:
    general = torch.as_tensor(payload["general_prompt"], dtype=torch.float32)
    bank = ExpertPromptBank(str(payload["method"]), 0, general)
    state = {"general": general}
    state.update({k.removeprefix("expert_"): torch.as_tensor(v, dtype=torch.float32)
                  for k, v in payload.items() if k.startswith("expert_")})
    bank.load_state_dict(state)
    return bank.eval()


def prompt_reconstruction_mac_proxy(method: str) -> int:
    if method in ("independent", "tied"):
        return 0
    if method == "scalar":
        return TASKS * PROMPT_DIM
    return TASKS * PROMPT_DIM * RANK


def selected_prompt_mac_proxy() -> int:
    return INPUT_DIM * INPUT_DIM
