from typing import Sequence
from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from einops import rearrange

from dataset import (
    NUM_STAGES,
    NUM_CHARACTERS,
    MAX_ACTIONSTATE,
    MAX_SHIELD,
    MAX_JUMPS,
    CategoricalActionConverter,
)
import utils


class Concat(nn.Module):
    def __init__(self, fn, dim=-1):
        super().__init__()
        self.fn = fn
        self.dim = dim

    def forward(self, *args, **kwargs):
        return self.fn(torch.cat(tuple(args), dim=self.dim), **kwargs)


class FeedForward(nn.Sequential):
    def __init__(
        self,
        input_dim,
        hidden_dims,
        output_dim,
        activation=nn.Mish,
        use_layer_norm=False,
    ):
        super().__init__()
        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims]
        self.add_module("fc_0", nn.Linear(input_dim, hidden_dims[0]))
        dims = hidden_dims + [output_dim]
        for k, (i, j) in enumerate(zip(dims[:-1], dims[1:])):
            self.add_module(f"act_{k+1}", activation(inplace=True))
            self.add_module(f"fc_{k+1}", nn.Linear(i, j))

        if use_layer_norm:
            self.add_module("ln", nn.LayerNorm(output_dim))


class QRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super().__init__()
        self.fcs = nn.ModuleList(
            [nn.Linear(input_size, 3 * hidden_size)]
            + [nn.Linear(hidden_size, 3 * hidden_size) for _ in range(num_layers - 1)]
        )

    def init_input_passthrough(self):
        l = self.fcs[0].bias.shape[0] // 3
        for i in range(len(self.fcs)):
            self.fcs[i].bias.data[l : 2 * l].fill_(6.0)

    def init_memory_passthrough(self):
        l = self.fcs[0].bias.shape[0] // 3
        for i in range(len(self.fcs)):
            self.fcs[i].bias.data[l : 2 * l].fill_(-6.0)

    def forward(self, x, h=None, keep_all_hs=False):
        if h is None:
            hs = len(self.fcs) * [None]
        else:
            assert len(self.fcs) == h.shape[-2], f"{len(self.fcs)} != {h.shape[-2]}"
            hs = list(h.chunk(len(self.fcs), dim=-2))

        for i, (fc, h) in enumerate(zip(self.fcs, hs)):
            Z, F, O = fc(x).chunk(3, dim=-1)
            C = utils.forget_mult(F.sigmoid(), Z.relu(), h)
            x = O.sigmoid() * C
            if keep_all_hs:
                hs[i] = C
            else:
                hs[i] = C[..., -1, :]

        h = torch.stack(hs, dim=-2)
        return x, h


class StateEmb(nn.Module):
    dim: int = (MAX_ACTIONSTATE + 1) + 6 + (MAX_JUMPS + 1)

    def __init__(self):
        super().__init__()
        self.as_temperature = nn.Parameter(torch.tensor(1.0, requires_grad=True))
        self.jumps_temperature = nn.Parameter(torch.tensor(1.0, requires_grad=True))

    def forward(self, x):
        as_, asc, percent, dir_, pos_x, pos_y, shield, jumps = x.unbind(-1)
        as_ = F.one_hot(as_.long(), MAX_ACTIONSTATE + 1) / F.softplus(
            self.as_temperature
        )
        asc = asc[..., None] / 100.0
        percent = percent[..., None] / 100.0
        dir_ = dir_[..., None]
        pos_x = pos_x[..., None] / 100.0
        pos_y = pos_y[..., None] / 75.0
        shield = shield[..., None] / MAX_SHIELD
        jumps = F.one_hot(jumps.long(), MAX_JUMPS + 1) / F.softplus(
            self.jumps_temperature
        )
        return torch.cat([as_, asc, percent, dir_, pos_x, pos_y, shield, jumps], dim=-1)


class ActionEmb(CategoricalActionConverter, nn.Module):
    dim: int = 4 * 117 + 99 + 128

    def forward(self, x):
        sx, sy, cx, cy, trigger, buttons = self.to_categorical(x.int()).unbind(-1)
        sx = F.one_hot(sx, self.num_stick_values)  # .log_softmax(-1)
        sy = F.one_hot(sy, self.num_stick_values)  # .log_softmax(-1)
        cx = F.one_hot(cx, self.num_stick_values)  # .log_softmax(-1)
        cy = F.one_hot(cy, self.num_stick_values)  # .log_softmax(-1)
        trigger = F.one_hot(trigger, self.num_trigger_values)  # .log_softmax(-1)
        buttons = F.one_hot(buttons, self.num_buttons_values)  # .log_softmax(-1)
        return torch.cat([sx, sy, cx, cy, trigger, buttons], dim=-1)


class MetadataEmb(nn.Module):
    dim: int = sum((NUM_STAGES, NUM_CHARACTERS, NUM_CHARACTERS))

    def forward(self, x):
        stage, char1, char2 = x.long().unbind(-1)
        stage = F.one_hot(stage, NUM_STAGES)
        char1 = F.one_hot(char1, NUM_CHARACTERS)
        char2 = F.one_hot(char2, NUM_CHARACTERS)
        return torch.cat([stage, char1, char2], dim=-1)


class StateDecoder(nn.Module):
    dim: int = (MAX_ACTIONSTATE + 1) + 6 + (MAX_JUMPS + 1)
    splits: Sequence[int] = (MAX_ACTIONSTATE + 1, 1, 1, 1, 1, 1, 1, MAX_JUMPS + 1)

    def __init__(self):
        super().__init__()
        self.register_buffer(
            "normalization",
            torch.tensor(
                [
                    1.0,
                    1 / 100.0,
                    1 / 100.0,
                    1.0,
                    1 / 100.0,
                    1 / 75.0,
                    1 / MAX_SHIELD,
                    1.0,
                ],
                requires_grad=False,
            ),
        )

    def loss(self, x, target):
        (
            as_target,
            asc_target,
            percent_target,
            dir_target,
            pos_x_target,
            pos_y_target,
            shield_target,
            jumps_target,
        ) = (self.normalization * target).unbind(-1)
        as_, asc, percent, dir_, pos_x, pos_y, shield, jumps = x.split(
            self.splits, dim=-1
        )
        asc = asc[..., 0]
        percent = percent[..., 0]
        dir_ = dir_[..., 0]
        pos_x = pos_x[..., 0]
        pos_y = pos_y[..., 0]
        shield = shield[..., 0]

        loss = 0.0
        loss += -torch.distributions.Categorical(logits=as_).log_prob(as_target.int())
        loss += F.mse_loss(asc, asc_target, reduction="none").clamp(-1.0, 1.0)
        loss += F.mse_loss(percent, percent_target, reduction="none").clamp(-1.0, 1.0)
        loss += F.mse_loss(dir_, dir_target, reduction="none").clamp(-1.0, 1.0)
        loss += F.mse_loss(pos_x, pos_x_target, reduction="none").clamp(-1.0, 1.0)
        loss += F.mse_loss(pos_y, pos_y_target, reduction="none").clamp(-1.0, 1.0)
        loss += F.mse_loss(shield, shield_target, reduction="none").clamp(-1.0, 1.0)
        loss += -torch.distributions.Categorical(logits=jumps).log_prob(
            jumps_target.int()
        )
        loss = loss.mean()
        infos = {}
        infos["accuracy_as"] = (as_.argmax(-1) == as_target.int()).float().mean()
        infos["rmse_asc"] = (asc - asc_target).square().mean().sqrt()
        infos["rmse_percent"] = (percent - percent_target).square().mean().sqrt()
        infos["rmse_dir"] = (dir_ - dir_target).square().mean().sqrt()
        infos["rmse_pos_x"] = (pos_x - pos_x_target).square().mean().sqrt()
        infos["rmse_pos_y"] = (pos_y - pos_y_target).square().mean().sqrt()
        infos["rmse_shield"] = (shield - shield_target).square().mean().sqrt()
        infos["acc_jumps"] = (jumps.argmax(-1) == jumps_target.int()).float().mean()
        infos["loss"] = loss.detach()
        return loss, infos


class ControllerDecoder(CategoricalActionConverter, nn.Module):
    def __init__(self, input_dim, hidden_dim=None, emb_dim=64, bias=True):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = 2 * input_dim
        self.register_buffer(
            "offsets",
            torch.from_numpy(
                np.cumsum([0] + 4 * [self.num_stick_values] + [self.num_trigger_values])
            ),
        )
        self.embed_token = nn.Embedding(self.total_num_values, emb_dim)
        self.to_logits = nn.ModuleList(
            [
                nn.Linear(input_dim + i * emb_dim, d, bias=bias)
                for i, d in enumerate(self.all_num_values)
            ]
        )

    def loss(self, x, targets, advantage=None, reduce=True):
        targets = self.to_categorical(targets)
        emb = self.embed_token(targets[..., 0:-1] + self.offsets[:-1])
        src = [
            torch.cat([x, *emb[..., :i, :].unbind(-2)], dim=-1)
            for i in range(len(self.all_num_values))
        ]
        logits = [f(s) for f, s in zip(self.to_logits, src)]
        logprobs = sum(
            [
                torch.distributions.Categorical(logits=l).log_prob(t)
                for l, t in zip(logits, targets.unbind(-1))
            ]
        )
        if advantage is None:
            loss: torch.Tensor = -logprobs
        else:
            loss = -(torch.exp(advantage) * logprobs).sum(-1)
            loss = loss / torch.exp(advantage).sum(-1)
        if reduce:
            loss = loss.mean()

        accuracy = (
            torch.stack(
                [l.argmax(-1) == t for l, t in zip(logits, targets.unbind(-1))], dim=-1
            )
            .float()
            .mean()
        )
        infos = {
            "accuracy": accuracy.mean().detach(),
            "loss": loss.mean().detach(),
        }
        return loss, infos

    @torch.no_grad()
    def sample(self, x, temperature=1.0, topk=0):
        actions = []
        entropy = 0
        for i, f in enumerate(self.to_logits):
            logits = f(x)  # + self.mask[i]
            entropy += torch.distributions.Categorical(logits=logits).entropy()
            if topk > 0:
                indices_to_remove = logits < torch.topk(logits, topk)[0][..., -1, None]
                logits[indices_to_remove] = -torch.inf
            if temperature > 0:
                pred = torch.distributions.Categorical(
                    logits=logits / temperature
                ).sample()
            else:
                pred = logits.argmax(-1)
            actions.append(pred)
            if i < len(self.to_logits) - 1:
                emb = self.embed_token(pred + self.offsets[i])
                x = torch.cat([x, emb], dim=-1)

        actions = torch.stack(actions, dim=-1)
        return actions, {"entropy": entropy}


class Predictor(nn.Module):
    """Prediction network from https://arxiv.org/abs/1810.07286"""

    def __init__(self, input_dim):
        super().__init__()
        self.DNF = Concat(
            FeedForward(
                ActionEmb.dim + 2 * StateEmb.dim + input_dim,
                512,
                3 * 2 * StateEmb.dim,
            )
        )
        self._init_dnf()

    def _init_dnf(self):
        l = self.DNF.fn[-1].bias.shape[0]
        # init D to ~=0
        self.DNF.fn[-1].bias.data[:l].zero_()
        self.DNF.fn[-1].weight.data[:l].normal_(0, 0.01)
        # init F to bypass N
        self.DNF.fn[-1].bias.data[2 * l :].fill_(6.0)

    def forward(self, action, state, *args):
        D, N, F = self.DNF(action, state, *args).chunk(3, dim=-1)
        F = F.sigmoid()
        return F * (state + D) + (1 - F) * N

    def unroll(self, actions, state, *args):
        for action in actions.unbind(-2):
            state = self(action, state.detach(), *args)
        return state

    def unroll_keep_all(self, actions, state, *args):
        states = []
        for action in actions.unbind(-2):
            state = self(action, state.detach(), *args)
            states.append(state)
        return torch.stack(states, dim=-2)


class Memory(nn.Module):
    def __init__(self, hidden_dim, num_layers=1):
        super().__init__()
        self.rnn = Concat(
            QRNN(
                ActionEmb.dim + 2 * StateEmb.dim + MetadataEmb.dim,
                hidden_dim,
                num_layers=num_layers,
            )
        )
        self.rnn.fn.init_input_passthrough()

    def forward(self, prev_actions, states, metadata, h=None, keep_all_hs=False):
        x, h = self.rnn(prev_actions, states, metadata, h=h, keep_all_hs=keep_all_hs)
        return x, h


@dataclass
class BCModelConfig:
    memory_dim: int
    memory_layers: int

    policy_hidden_dims: int | Sequence[int]
    policy_dim: int
    delay: int

    controller_decoder_emb_dim: int = 64

    def __post_init__(self):
        assert self.delay > 0


class BCModel(nn.Module):
    def __init__(self, config: BCModelConfig):
        super().__init__()
        self.config = config
        self.metadata_emb = MetadataEmb()
        self.action_emb = ActionEmb()
        self.state_emb = StateEmb()
        self.memory = Memory(config.memory_dim, config.memory_layers)
        self.predictor = Predictor(config.memory_dim + MetadataEmb.dim)
        self.state_decoder = StateDecoder()
        self.critic = FeedForward(
            input_dim=config.memory_dim,
            hidden_dims=config.policy_hidden_dims,
            output_dim=1,
        )
        self.policy = FeedForward(
            input_dim=config.memory_dim,
            hidden_dims=config.policy_hidden_dims,
            output_dim=config.policy_dim,
            use_layer_norm=True,
        )
        self.action_decoder = ControllerDecoder(config.policy_dim, emb_dim=config.controller_decoder_emb_dim)

    def _split_by_delay(self, x, dim=-2):
        d = self.config.delay
        x_past, x_future = x.split([x.shape[dim] - d, d], dim=dim)
        return x_past, x_future

    def _unfold_nexts(self, x, size):
        return rearrange(
            x[..., 1:, :].unfold(dimension=-2, size=size, step=1),
            "... D T -> ... T D",
        )

    def _loss_predictor(self, z, states_targets):
        z = rearrange(z, "... (P D) -> P ... D", P=2)
        return self.state_decoder.loss(z, states_targets)

    def loss(self, actions, states1, states2, metadata, rewards):
        infos = {}
        delay = self.config.delay
        actions_target = actions[..., 1 + delay :, :]

        states = torch.stack([states1, states2], dim=0)[..., :-1, :]
        states_target_for_predictor = self._unfold_nexts(states, delay)

        prev_actions = self.action_emb(actions[..., :-1, :])
        states = self.state_emb(states)
        states = rearrange(states, "P ... D -> ... (P D)")
        metadata = self.metadata_emb(metadata)

        states_past, _ = self._split_by_delay(states)

        z, h = self.memory(
            prev_actions,
            states,
            utils.extend_as(metadata, states.shape),
            keep_all_hs=True,
        )

        adv = torch.zeros_like(rewards[..., 1 + delay :])
        value = self.critic(z[..., delay:, :])[..., 0]
        adv = rewards[..., 1 + delay :] - value
        loss_critic = adv.square().mean()
        adv = adv.clamp(-1.4, 1.4).detach()

        h, _ = self._split_by_delay(h, dim=-3)
        z_past, _ = self._split_by_delay(z)

        prev_actions_unfolded = self._unfold_nexts(prev_actions, delay)
        states_pred = self.predictor.unroll_keep_all(
            prev_actions_unfolded,
            states_past,
            utils.extend_as(metadata, z_past.shape),
            z_past,
        )

        z_pred, _ = self.memory(
            prev_actions_unfolded,
            states_pred,
            utils.extend_as(metadata, states_pred.shape),
            h=h,
        )
        z_pred = z_pred[..., -1, :]

        loss_predictor, infos_predictor = self._loss_predictor(
            states_pred, states_target_for_predictor
        )

        z_policy = self.policy(z_pred)
        loss_action, infos_action = self.action_decoder.loss(
            z_policy, actions_target, advantage=adv
        )

        infos.update({"loss_critic": loss_critic.detach()})
        infos.update({k + "_predictor": v for k, v in infos_predictor.items()})
        infos.update({k + "_action": v for k, v in infos_action.items()})
        return loss_predictor + loss_action + loss_critic, infos

    @torch.no_grad()
    def inference(self, actions, state1, state2, metadata, temperature=1.0, topk=0):
        delay = self.config.delay
        actions = actions[:, -delay - 1 :, :]
        state1 = state1[:, -delay - 1 : -delay, :]
        state2 = state2[:, -delay - 1 : -delay, :]

        action_prev, actions_future = self._split_by_delay(self.action_emb(actions))
        state = torch.stack([state1, state2], dim=0)
        state = self.state_emb(state)
        state = rearrange(state, "P ... D -> ... (P D)")
        metadata = self.metadata_emb(metadata)

        try:
            memory_state = self.memory_state
        except:
            print("No memory states")
            memory_state = None

        z, self.memory_state = self.memory(
            action_prev, state, metadata[..., None, :], h=memory_state
        )
        z = z[..., -1, :]

        states_pred = self.predictor.unroll_keep_all(
            actions_future, state[..., -1, :], metadata, z
        )
        z_pred, _ = self.memory(
            actions_future,
            states_pred,
            utils.extend_as(metadata, states_pred.shape),
            h=self.memory_state,
        )
        z_policy = self.policy(z_pred[..., -1:, :])

        act, infos = self.action_decoder.sample(z_policy, temperature, topk)
        infos = {k: v.cpu().numpy() for k, v in infos.items()}
        return self.action_decoder.to_int(act).cpu().numpy(), infos

    def save_with_config(self, path):
        config = self.config
        s = self.state_dict()
        s["config"] = config
        try:
            torch.save(s, path)
        except Exception as e:
            print(e)


if __name__ == "__main__":
    from dataset import MeleeDataloader

    def prepare_data(actions, states, metadata, rewards, epoch=0):
        index_self = epoch % 2
        index_other = (epoch + 1) % 2
        states1 = states[:, index_self]
        states2 = states[:, index_other]
        actions = actions[:, index_self]
        metadata = metadata.int()
        if index_self == 1:
            metadata = metadata[:, [0, 2, 1]]
            rewards = -rewards
        return (
            actions,
            states1,
            states2,
            metadata,
            rewards,
        )

    config = BCModelConfig(
        memory_dim=512,
        memory_layers=3,
        policy_hidden_dims=1024,
        policy_dim=512,
        delay=4,
    )
    data = MeleeDataloader(
        seq_len=15 + 1,
        path="./data/mmaped/",
        data_types=["actions", "states", "metadata"],
    )

    model = BCModel(config)

    actions, states, metadata, rewards = next(
        data.get_samples(
            batch_size=2,
            device="cpu",
            return_rewards=True,
            low_discrepancy_index=0,
        )
    )
    (
        actions,
        states1,
        states2,
        metadata,
        rewards,
    ) = prepare_data(actions, states, metadata, rewards)

    states_future_ema = None  # model.embed_states(states1_future, states2_future)

    loss, infos = model.loss(actions, states1, states2, metadata, rewards)
    print(loss)
    print(infos)
