from typing import Any, Sequence
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from mmap_ninja import numpy as RaggedMmap
from einops import rearrange

# Constants

NUM_STAGES = 33
NUM_CHARACTERS = 34

MAX_ACTIONSTATE = 392
MAX_SHIELD = 60.0
MAX_JUMPS = 6

ACTION_DIM = 6

LOW_DISCREPANCY_SEQ = np.array([0, 1 / 2, 1 / 4, 3 / 4, 1 / 8, 5 / 8, 3 / 8, 7 / 8])


def is_game_start(state):
    return state[..., 0] == 322.0


def is_ko_state(prev_state, state):
    kos = (prev_state > 0xA) * (state <= 0xA)
    kos = np.concatenate([kos[0:1], kos], axis=0)
    return kos


def is_damage_state(state):
    return (state >= 0x4B) * (state <= 0x5B)


def compute_reward(state, ko_mult=1.0, hitstun_mult=0.001):
    # assert state.ndim == 2 and state.shape[-1] == 2
    kos = is_ko_state(state[:-1], state[1:])
    hitstun = is_damage_state(state)
    reward = ko_mult * kos + hitstun_mult * hitstun
    reward = reward[..., 1] - reward[..., 0]
    return reward


def compute_episode_end(state):
    game_starts = is_game_start(state)
    episode_ends = np.concatenate(
        [game_starts[1:], np.ones_like(game_starts[0:1])], axis=0
    )
    return episode_ends


def compute_discounted_returns(rewards, terminal_states, discount=0.999):
    discounts = discount * (1 - terminal_states)
    discounted_returns = np.zeros_like(rewards)
    discounts = (1 - terminal_states) * discount
    returns = 0
    for i in reversed(range(len(rewards))):
        returns = rewards[i] + discounts[i] * returns
        discounted_returns[i] = returns
    return discounted_returns


def compute_returns_from_state(state):
    rewards = compute_reward(state)
    episode_ends = compute_episode_end(state)
    returns = compute_discounted_returns(rewards, episode_ends)
    return returns


class CategoricalActionConverter:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.join_buttons = True

        # bounds
        self.stick_bounds = (
            0.5 + torch.cat([torch.arange(-80, -22), torch.arange(22, 80)]).contiguous()
        )
        self.trigger_bounds = 0.5 + torch.arange(42, 140).contiguous()
        self.button_mults = 2 ** torch.arange(7).contiguous()

        # values
        self.stick_values = torch.cat(
            [torch.arange(-80, -22), torch.tensor([0]), torch.arange(23, 81)]
        ).contiguous()
        self.trigger_values = torch.cat(
            [torch.tensor([0]), torch.arange(43, 141)]
        ).contiguous()

        self.num_stick_values = 117
        self.num_trigger_values = 99
        self.num_buttons_values = 128
        self.max_num_values = max(
            self.num_stick_values, self.num_trigger_values, self.num_buttons_values
        )
        self.all_num_values = np.array(
            [
                self.num_stick_values,
                self.num_stick_values,
                self.num_stick_values,
                self.num_stick_values,
                self.num_trigger_values,
                self.num_buttons_values,
            ]
        )
        self.total_num_values = self.all_num_values.sum()

    def to_categorical(self, action):
        self.stick_bounds = self.stick_bounds.to(action.device)
        self.trigger_bounds = self.trigger_bounds.to(action.device)
        self.button_mults = self.button_mults.to(action.device)

        sticks = action[..., :4].contiguous().clamp(-80, 80)
        trigger = action[..., 4:5].to(torch.uint8).contiguous().clamp(0, 140)
        buttons = action[..., 5:].contiguous()

        sticks = torch.bucketize(sticks, self.stick_bounds, out_int32=True)
        trigger = torch.bucketize(trigger, self.trigger_bounds, out_int32=True)
        if self.join_buttons:
            buttons = (
                (buttons * self.button_mults).sum(dim=-1, keepdim=True).clamp(0, 127)
            )

        action = torch.cat([sticks, trigger, buttons], dim=-1)
        return action

    def to_int(self, action):
        self.stick_values = self.stick_values.to(action.device)
        self.trigger_values = self.trigger_values.to(action.device)
        self.button_mults = self.button_mults.to(action.device)

        sticks = action[..., :4]
        trigger = action[..., 4:5]
        buttons = action[..., 5:]

        sticks = self.stick_values.take(sticks)
        trigger = self.trigger_values.take(trigger)
        if self.join_buttons:
            buttons = buttons.bitwise_and(self.button_mults).ne(0).byte()

        action = torch.cat([sticks, trigger, buttons], dim=-1)
        return action


def mask_neutral_controller_sequence(actions, max_neutral=None):
    if max_neutral is None:
        max_neutral = actions.shape[-2] - 1
    x = (actions == 0).sum(-1) == actions.shape[-1]
    n = x.sum(-1)
    mask = n > max_neutral
    return mask


class AbstractMeleeDataset(Dataset):
    def __init__(
        self,
        path: str | Path,
        data_types: Sequence[str] = ["actions", "states", "metadata"],
    ):
        path = Path(path)
        if not (
            "actions" in data_types
            or "states" in data_types
            or "metadata" in data_types
        ):
            raise
        self.actions = None
        self.states = None
        self.metadata = None
        if "actions" in data_types:
            self.actions = RaggedMmap.open_existing(path / "mmap_actions")
            self.max_len = self.actions.shape[0]
        if "states" in data_types:
            self.states = RaggedMmap.open_existing(path / "mmap_states")
            self.max_len = self.states.shape[0]
        if "metadata" in data_types:
            self.metadata = RaggedMmap.open_existing(path / "mmap_metadata")
            self.max_len = self.metadata.shape[0]

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError


class MeleeDataloader(AbstractMeleeDataset):
    def __init__(self, seq_len: int, **kwargs):
        super().__init__(**kwargs)
        self.seq_len = seq_len
        self.split_size = min(2**19, self.max_len)

    def len(self, batch_size):
        assert self.seq_len * batch_size <= self.split_size
        num_splits = self.max_len // self.split_size

        total_num_batch = 0
        for i in range(num_splits):
            idx_start = i * self.split_size
            idx_stop = min((i + 1) * self.split_size, self.max_len)

            l = idx_stop - idx_start - self.seq_len
            n = self.seq_len * batch_size
            l = l // n * n

            num_batch = l // n
            total_num_batch += num_batch
        return total_num_batch

    def get_samples(
        self,
        batch_size: int,
        device: torch.device | str,
        return_rewards=False,
        low_discrepancy_index=0,
    ):
        assert self.seq_len * batch_size <= self.split_size
        offset = int(self.seq_len * LOW_DISCREPANCY_SEQ[low_discrepancy_index % 8]) * (
            self.seq_len - 1
        )
        # offset = (
        #     (5 * low_discrepancy_index) % (self.seq_len - 1) if self.seq_len > 1 else 0
        # )
        num_splits = self.max_len // self.split_size

        for i in range(num_splits):
            idx_start = i * self.split_size + offset
            idx_stop = min((i + 1) * self.split_size + offset, self.max_len)

            l = idx_stop - idx_start - self.seq_len
            n = self.seq_len * batch_size
            l = l // n * n
            num_batches = l // n

            if self.metadata is not None:
                metadata = self.metadata[idx_start:idx_stop]
                metadata = rearrange(
                    metadata[:l], "(B N T) D -> N B T D", B=batch_size, T=self.seq_len
                )[..., 0, :]
                metadata = torch.from_numpy(metadata.copy()).to(device)
                metadata[..., 0] = metadata[..., 0].clamp(0, NUM_STAGES - 1)
                metadata[..., 1:] = metadata[..., 1:].clamp(0, NUM_CHARACTERS - 1)
                num_batches = metadata.shape[0]

            if self.actions is not None:
                actions = self.actions[idx_start:idx_stop]
                actions = rearrange(
                    actions[:l],
                    "(B N T) P D -> N B P T D",
                    B=batch_size,
                    T=self.seq_len,
                )
                actions = torch.from_numpy(actions.copy()).int().to(device)
                # actions[...,:4] = actions[...,:4] + 127
                actions[..., 4] = actions[..., 4].to(torch.uint8)
                num_batches = actions.shape[0]

            if self.states is not None:
                states = self.states[idx_start:idx_stop]
                states = rearrange(
                    states[:l], "(B N T) P D -> N B P T D", B=batch_size, T=self.seq_len
                )
                states = torch.from_numpy(states.copy()).to(device)
                states[..., 0] = states[..., 0].clamp(0, MAX_ACTIONSTATE)
                states[..., 1] = states[..., 1].clamp(0, 119.0)
                states[..., 2] = states[..., 2].clamp(0, 300.0)
                states[..., 3] = states[..., 3].clamp(-1.0, 1.0)
                states[..., 4] = states[..., 4].clamp(-275.0, 275.0)
                states[..., 5] = states[..., 5].clamp(-170.0, 340.0)
                states[..., 6] = states[..., 6].clamp(0, MAX_SHIELD)
                states[..., 7] = states[..., 7].clamp(0, MAX_JUMPS)
                num_batches = states.shape[0]

            returns = None
            if return_rewards and self.states is not None:
                returns = compute_returns_from_state(
                    self.states[idx_start:idx_stop, ..., 0]
                )
                returns = rearrange(
                    returns[:l],
                    "(B N T) -> N B T",
                    B=batch_size,
                    T=self.seq_len,
                )
                returns = (
                    torch.from_numpy(returns.copy()).float().to(device).contiguous()
                )

            def _get_one_item(idx):
                if self.actions is not None:
                    a = actions[idx]
                    yield a
                if self.states is not None:
                    s = states[idx]
                    yield s
                if self.metadata is not None:
                    m = metadata[idx]
                    yield m
                if returns is not None:
                    r = returns[idx]
                    yield r

            perm = torch.randperm(num_batches)

            for idx in perm:
                x = list(_get_one_item(idx))
                if len(x) == 1:
                    x = x[0]
                yield x


if __name__ == "__main__":
    loader = MeleeDataloader(seq_len=256, path="/media/DATA/Melee/FloatData/")
    data = loader.get_samples(1024, return_rewards=True, device="cuda")

    m = torch.inf
    M = -torch.inf
    for _ in range(100000):
        _, s, _, r = next(data)
        m = min(r.min(), m)
        M = max(r.max(), M)
        print(m, M)
        # s = s.flatten(end_dim=-2)
        # m_ = s.min(0).values
        # M_ = s.max(0).values
        # m = torch.minimum(m, m_)
        # M = torch.maximum(M, M_)
        # print(m.numpy(), M.numpy())

        # a = a[:, 0, 0]
        # a = a[0,0,:,4].to(torch.uint8) / 140.

        # print(a[..., 5].sum() / a[..., 5].numel())
        # ac = action_to_categorical(a)
        # print(torch.unique(a[...,:4]))
        # print(torch.unique(a[..., 4].to(torch.uint8)))

    # conv = CategoricalActionConverter()
