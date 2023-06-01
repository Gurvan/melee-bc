from typing import Sequence, Tuple

# from dataclasses import dataclass

import torch
from torch import native_layer_norm, neg_, nn
import torch.nn.functional as F

# import numpy as np

from einops import rearrange

from dataset import (
    NUM_STAGES,
    NUM_CHARACTERS,
    MAX_ACTIONSTATE,
    # MAX_SHIELD,
    # MAX_JUMPS,
    # CategoricalActionConverter,
)

# import utils


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


# class QRNN(nn.Module):
#     def __init__(self, input_size, hidden_size, num_layers=1):
#         super().__init__()
#         self.fcs = nn.ModuleList(
#             [nn.Linear(input_size, 3 * hidden_size)]
#             + [nn.Linear(hidden_size, 3 * hidden_size) for _ in range(num_layers - 1)]
#         )
#
#     def init_input_passthrough(self):
#         l = self.fcs[0].bias.shape[0] // 3
#         for i in range(len(self.fcs)):
#             self.fcs[i].bias.data[l : 2 * l].fill_(6.0)
#
#     def init_memory_passthrough(self):
#         l = self.fcs[0].bias.shape[0] // 3
#         for i in range(len(self.fcs)):
#             self.fcs[i].bias.data[l : 2 * l].fill_(-6.0)
#
#     def forward(self, x, h=None, keep_all_hs=False):
#         if h is None:
#             hs = len(self.fcs) * [None]
#         else:
#             assert len(self.fcs) == h.shape[-2], f"{len(self.fcs)} != {h.shape[-2]}"
#             hs = list(h.chunk(len(self.fcs), dim=-2))
#
#         for i, (fc, h) in enumerate(zip(self.fcs, hs)):
#             Z, F, O = fc(x).chunk(3, dim=-1)
#             C = utils.forget_mult(F.sigmoid(), Z.relu(), h)
#             x = O.sigmoid() * C
#             if keep_all_hs:
#                 hs[i] = C
#             else:
#                 hs[i] = C[..., -1, :]
#
#         h = torch.stack(hs, dim=-2)
#         return x, h
#


class GaussianEmbedding(nn.Module):
    def __init__(
        self,
        bounds: Tuple[float, float] = (-1, 1),
        n: int = 16,
        learnable: bool = False,
    ):
        super().__init__()
        bmin = min(bounds)
        bmax = max(bounds)
        bdif = (bmax - bmin) / (n - 3)
        self.means = nn.Parameter(
            torch.linspace(bmin - bdif, bmax + bdif, n, requires_grad=learnable)
        )
        self.log_stds = nn.Parameter(
            (bdif * torch.ones(n, requires_grad=learnable)).log()
        )

    def forward(self, x):
        return (
            torch.distributions.Normal(self.means, self.log_stds.clamp(max=10.0).exp())
            .log_prob(x[..., None])
            .softmax(dim=-1)
        )

    def encode(self, x):
        return self(x)

    def decode(self, x):
        return (x * self.means).sum(dim=-1)

    def sample(self, x, deterministic=False):
        if deterministic:
            return self.decode(x)
        values = torch.distributions.Normal(
            self.means, self.log_stds.clamp(max=10.0).exp()
        ).sample(x.shape[:-1])
        return (x * values).sum(dim=-1)

    def loss(self, x, target):
        return F.mse_loss(self.decode(x), target, reduction="none")


class ObsEmbedding(nn.Module):
    def __init__(
        self,
        emb_dim: int = 64,
        as_emb_dim: int = 64,
        asc_emb_dim: int = 32,
        pos_emb_dim: int = 64,
    ):
        super().__init__()
        self.as_emb = nn.Embedding(MAX_ACTIONSTATE + 1, as_emb_dim)
        self.asc_emb = GaussianEmbedding((-1, 60), asc_emb_dim, learnable=True)
        self.as_asc_emb = Concat(
            FeedForward(as_emb_dim + asc_emb_dim, emb_dim // 2, emb_dim)
        )
        self.percent_emb = GaussianEmbedding((0, 200), emb_dim, learnable=True)
        self.dir_emb = nn.Parameter(0.02 * torch.randn(emb_dim, requires_grad=True))
        self.pos_x_emb = GaussianEmbedding((-150, 150), pos_emb_dim, learnable=True)
        self.pos_y_emb = GaussianEmbedding((-100, 200), pos_emb_dim, learnable=True)
        self.pos_emb = Concat(FeedForward(2 * pos_emb_dim, emb_dim // 2, emb_dim))

    def forward(self, obs):
        as_, asc, percent, dir_, pos_x, pos_y, _, _ = obs.unbind(-1)
        as_asc = self.as_asc_emb(self.as_emb(as_.long()), self.asc_emb(asc))
        percent = self.percent_emb(percent)
        dir_ = dir_[..., None] * self.dir_emb
        pos = self.pos_emb(self.pos_x_emb(pos_x), self.pos_y_emb(pos_y))
        return as_asc, percent, dir_, pos


class MetadataEmbedding(nn.Module):
    def __init__(self, emb_dim: int):
        super().__init__()
        self.stage_embedding = nn.Embedding(NUM_STAGES, emb_dim)
        self.character1_embedding = nn.Embedding(NUM_CHARACTERS, emb_dim)
        self.character2_embedding = nn.Embedding(NUM_CHARACTERS, emb_dim)

    def forward(self, x):
        x = x.long()
        # x is expected to be of size (N, 3)
        stage = self.stage_embedding(x[:, 0])
        character1 = self.character1_embedding(x[:, 1])
        character2 = self.character2_embedding(x[:, 2])
        return stage, character1, character2


class StateEmbedding(nn.Module):
    def __init__(
        self,
        emb_dim: int,
        output_dim: int,
        n_head: int = 2,
        n_layer: int = 2,
        ff_mult: int = 2,
    ):
        super().__init__()
        self.obs_emb = ObsEmbedding(emb_dim)
        self.meta_emb = MetadataEmbedding(emb_dim)
        self.pos_enc = nn.Parameter(0.02 * torch.randn(11, emb_dim, requires_grad=True))
        self.torso = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=emb_dim,
                nhead=n_head,
                dim_feedforward=ff_mult * emb_dim,
                activation=F.gelu,
            ),
            num_layers=n_layer,
        )
        self.proj = nn.Linear(11 * emb_dim, output_dim)

    def forward(self, obs_s, obs_o, meta):
        obs_s = self.obs_emb(obs_s)
        obs_o = self.obs_emb(obs_o)
        meta = tuple([m[...,None,:].expand_as(obs_s[0]) for m in self.meta_emb(meta)])
        x = torch.stack(
            obs_s + obs_o + meta, dim=-2
        )
        x = x + self.pos_enc
        x = self.torso(x.flatten(end_dim=-3)).view(*x.shape)
        return self.proj(torch.cat(x.unbind(dim=-2), dim=-1))


class ActionEncoding(nn.Module):
    def __init__(
        self,
        emb_dim: int = 64,
    ):
        super().__init__()
        self.sx_emb = GaussianEmbedding((-80, 80), emb_dim, learnable=True)
        self.sy_emb = GaussianEmbedding((-80, 80), emb_dim, learnable=True)
        self.cx_emb = GaussianEmbedding((-80, 80), emb_dim, learnable=True)
        self.cy_emb = GaussianEmbedding((-80, 80), emb_dim, learnable=True)
        self.tr_emb = GaussianEmbedding((0, 140), emb_dim, learnable=True)
        self.buttons_emb = nn.Embedding(2 * 7, emb_dim)
        self.register_buffer(
            "buttons_offsets", 2 * torch.arange(7, requires_grad=False)
        )
        self.buttons_mask = (
            torch.arange(7).repeat_interleave(2, dim=-1),
            torch.arange(14),
        )

    def forward(self, action):
        sx, sy, cx, cy, tr, buttons = torch.split(action, [1, 1, 1, 1, 1, 7], dim=-1)
        sx = self.sx_emb(sx[..., 0])
        sy = self.sy_emb(sy[..., 0])
        cx = self.cx_emb(cx[..., 0])
        cy = self.cy_emb(cy[..., 0])
        tr = self.tr_emb(tr[..., 0])
        buttons = self.buttons_emb(buttons.long() + self.buttons_offsets)
        return sx, sy, cx, cy, tr, *buttons.unbind(dim=-2)

    def decode(self, logits):
        sx, sy, cx, cy, tr, buttons = torch.split(logits, [1, 1, 1, 1, 1, 7], dim=-2)
        sx = self.sx_emb.decode(sx)
        sy = self.sy_emb.decode(sy)
        cx = self.cx_emb.decode(cx)
        cy = self.cy_emb.decode(cy)
        tr = self.tr_emb.decode(tr)
        buttons_sim = buttons @ self.buttons_emb.weight.T  # * self.buttons_mask
        buttons = rearrange(
            buttons_sim[..., self.buttons_mask[0], self.buttons_mask[1]],
            "... (N B) -> ... N B",
            B=2,
        )
        return sx, sy, cx, cy, tr, buttons.argmax(dim=-1)

    def sample(self, logits, deterministic=False):
        sx, sy, cx, cy, tr, buttons = torch.split(logits, [1, 1, 1, 1, 1, 7], dim=-2)
        sx = self.sx_emb.sample(sx[..., 0, :], deterministic)
        sy = self.sy_emb.sample(sy[..., 0, :], deterministic)
        cx = self.cx_emb.sample(cx[..., 0, :], deterministic)
        cy = self.cy_emb.sample(cy[..., 0, :], deterministic)
        tr = self.tr_emb.sample(tr[..., 0, :], deterministic)
        buttons_sim = buttons @ self.buttons_emb.weight.T  # * self.buttons_mask
        buttons = rearrange(
            buttons_sim[..., self.buttons_mask[0], self.buttons_mask[1]],
            "... (N B) -> ... N B",
            B=2,
        )
        if deterministic:
            return sx, sy, cx, cy, tr, buttons.argmax(dim=-1)
        buttons = torch.distributions.Categorical(logits=buttons).sample()
        return sx, sy, cx, cy, tr, buttons

    def loss(self, logits, targets):
        (
            sx_logits,
            sy_logits,
            cx_logits,
            cy_logits,
            tr_logits,
            buttons_logits,
        ) = torch.split(logits, [1, 1, 1, 1, 1, 7], dim=-2)
        (
            sx_targets,
            sy_targets,
            cx_targets,
            cy_targets,
            tr_targets,
            buttons_targets,
        ) = torch.split(targets, [1, 1, 1, 1, 1, 7], dim=-1)

        buttons_sim = buttons_logits @ self.buttons_emb.weight.T  # * self.buttons_mask
        buttons_logits = rearrange(
            buttons_sim[..., self.buttons_mask[0], self.buttons_mask[1]],
            "... (N B) -> ... N B",
            B=2,
        )
        loss = 0.0
        loss += self.sx_emb.loss(sx_logits[..., 0, :], sx_targets[..., 0])
        loss += self.sy_emb.loss(sy_logits[..., 0, :], sy_targets[..., 0])
        loss += self.cx_emb.loss(cx_logits[..., 0, :], cx_targets[..., 0])
        loss += self.cy_emb.loss(cy_logits[..., 0, :], cy_targets[..., 0])
        loss += self.tr_emb.loss(tr_logits[..., 0, :], tr_targets[..., 0])
        buttons_loss = F.cross_entropy(
            buttons_logits.flatten(end_dim=-2),
            buttons_targets.long().flatten(),
            reduction="none",
        )
        buttons_loss = buttons_loss.view(*loss.shape, 7)
        loss += buttons_loss.sum(dim=-1)
        return loss


class BCModel(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        emb_dim = 64
        latent_dim = 128
        n_head = 2
        n_layer = 2
        self.state_emb = StateEmbedding(emb_dim, latent_dim)
        self.action_emb = ActionEncoding(emb_dim)
        self.action_proj = nn.Linear(12*emb_dim, latent_dim)
        self.action_unproj = nn.Linear(latent_dim, 12*emb_dim)
        self.torso = nn.Transformer(
            d_model=latent_dim,
            nhead=n_head,
            num_encoder_layers=n_layer,
            num_decoder_layers=n_layer,
            activation=F.gelu,
            batch_first=True,
        )

    def loss(self, actions, obs_s, obs_o, meta):
        actions_target = actions[..., 1:,:]

        states = self.state_emb(obs_s, obs_o, meta)[..., :-1, :]
        prev_actions = self.action_emb(actions[..., :-1, :])
        prev_actions = self.action_proj(torch.cat(prev_actions, dim=-1))
        logits = self.torso(states, prev_actions)
        logits = self.action_unproj(logits).view(*prev_actions.shape[:-1], 12, -1)
        loss = self.action_emb.loss(logits, actions_target)
        return loss

#   def loss(self, actions, states1, states2, metadata, rewards):
#         infos = {}
#         delay = self.config.delay
#         actions_target = actions[..., 1 + delay :, :]
#
#         states = torch.stack([states1, states2], dim=0)[..., :-1, :]
#         states_target_for_predictor = self._unfold_nexts(states, delay)
#
#         prev_actions = self.action_emb(actions[..., :-1, :])
#         states = self.state_emb(states)
#         states = rearrange(states, "P ... D -> ... (P D)")
#         metadata = self.metadata_emb(metadata)
#
#         states_past, _ = self._split_by_delay(states)
#
#         z, h = self.memory(
#             prev_actions,
#             states,
#             utils.extend_as(metadata, states.shape),
#             keep_all_hs=True,
#         )
#
#         adv = torch.zeros_like(rewards[..., 1 + delay :])
#         value = self.critic(z[..., delay:, :])[..., 0]
#         adv = rewards[..., 1 + delay :] - value
#         loss_critic = adv.square().mean()
#         adv = adv.clamp(-1.4, 1.4).detach()
#
#         h, _ = self._split_by_delay(h, dim=-3)
#         z_past, _ = self._split_by_delay(z)
#
#         prev_actions_unfolded = self._unfold_nexts(prev_actions, delay)
#         states_pred = self.predictor.unroll_keep_all(
#             prev_actions_unfolded,
#             states_past,
#             utils.extend_as(metadata, z_past.shape),
#             z_past,
#         )
#
#         z_pred, _ = self.memory(
#             prev_actions_unfolded,
#             states_pred,
#             utils.extend_as(metadata, states_pred.shape),
#             h=h,
#         )
#         z_pred = z_pred[..., -1, :]
#
#         loss_predictor, infos_predictor = self._loss_predictor(
#             states_pred, states_target_for_predictor
#         )
#
#         z_policy = self.policy(z_pred)
#         loss_action, infos_action = self.action_decoder.loss(
#             z_policy, actions_target, advantage=adv
#         )
#
#         infos.update({"loss_critic": loss_critic.detach()})
#         infos.update({k + "_predictor": v for k, v in infos_predictor.items()})
#         infos.update({k + "_action": v for k, v in infos_action.items()})
#         return loss_predictor + loss_action + loss_critic, infos
#


# @dataclass
# class BCModelConfig:
#     memory_dim: int
#     memory_layers: int
#
#     policy_hidden_dims: int | Sequence[int]
#     policy_dim: int
#     delay: int
#
#     controller_decoder_emb_dim: int = 64
#
#     def __post_init__(self):
#         assert self.delay > 0
#
#
# class BCModel(nn.Module):
#     def __init__(self, config: BCModelConfig):
#         super().__init__()
#         self.config = config
#         self.metadata_emb = MetadataEmb()
#         self.action_emb = ActionEmb()
#         self.state_emb = StateEmb()
#         self.memory = Memory(config.memory_dim, config.memory_layers)
#         self.predictor = Predictor(config.memory_dim + MetadataEmb.dim)
#         self.state_decoder = StateDecoder()
#         self.critic = FeedForward(
#             input_dim=config.memory_dim,
#             hidden_dims=config.policy_hidden_dims,
#             output_dim=1,
#         )
#         self.policy = FeedForward(
#             input_dim=config.memory_dim,
#             hidden_dims=config.policy_hidden_dims,
#             output_dim=config.policy_dim,
#             use_layer_norm=True,
#         )
#         self.action_decoder = ControllerDecoder(
#             config.policy_dim, emb_dim=config.controller_decoder_emb_dim
#         )
#
#     def _split_by_delay(self, x, dim=-2):
#         d = self.config.delay
#         x_past, x_future = x.split([x.shape[dim] - d, d], dim=dim)
#         return x_past, x_future
#
#     def _unfold_nexts(self, x, size):
#         return rearrange(
#             x[..., 1:, :].unfold(dimension=-2, size=size, step=1),
#             "... D T -> ... T D",
#         )
#
#     def _loss_predictor(self, z, states_targets):
#         z = rearrange(z, "... (P D) -> P ... D", P=2)
#         return self.state_decoder.loss(z, states_targets)
#
#     def loss(self, actions, states1, states2, metadata, rewards):
#         infos = {}
#         delay = self.config.delay
#         actions_target = actions[..., 1 + delay :, :]
#
#         states = torch.stack([states1, states2], dim=0)[..., :-1, :]
#         states_target_for_predictor = self._unfold_nexts(states, delay)
#
#         prev_actions = self.action_emb(actions[..., :-1, :])
#         states = self.state_emb(states)
#         states = rearrange(states, "P ... D -> ... (P D)")
#         metadata = self.metadata_emb(metadata)
#
#         states_past, _ = self._split_by_delay(states)
#
#         z, h = self.memory(
#             prev_actions,
#             states,
#             utils.extend_as(metadata, states.shape),
#             keep_all_hs=True,
#         )
#
#         adv = torch.zeros_like(rewards[..., 1 + delay :])
#         value = self.critic(z[..., delay:, :])[..., 0]
#         adv = rewards[..., 1 + delay :] - value
#         loss_critic = adv.square().mean()
#         adv = adv.clamp(-1.4, 1.4).detach()
#
#         h, _ = self._split_by_delay(h, dim=-3)
#         z_past, _ = self._split_by_delay(z)
#
#         prev_actions_unfolded = self._unfold_nexts(prev_actions, delay)
#         states_pred = self.predictor.unroll_keep_all(
#             prev_actions_unfolded,
#             states_past,
#             utils.extend_as(metadata, z_past.shape),
#             z_past,
#         )
#
#         z_pred, _ = self.memory(
#             prev_actions_unfolded,
#             states_pred,
#             utils.extend_as(metadata, states_pred.shape),
#             h=h,
#         )
#         z_pred = z_pred[..., -1, :]
#
#         loss_predictor, infos_predictor = self._loss_predictor(
#             states_pred, states_target_for_predictor
#         )
#
#         z_policy = self.policy(z_pred)
#         loss_action, infos_action = self.action_decoder.loss(
#             z_policy, actions_target, advantage=adv
#         )
#
#         infos.update({"loss_critic": loss_critic.detach()})
#         infos.update({k + "_predictor": v for k, v in infos_predictor.items()})
#         infos.update({k + "_action": v for k, v in infos_action.items()})
#         return loss_predictor + loss_action + loss_critic, infos
#
#     @torch.no_grad()
#     def inference(self, actions, state1, state2, metadata, temperature=1.0, topk=0):
#         delay = self.config.delay
#         actions = actions[:, -delay - 1 :, :]
#         state1 = state1[:, -delay - 1 : -delay, :]
#         state2 = state2[:, -delay - 1 : -delay, :]
#
#         action_prev, actions_future = self._split_by_delay(self.action_emb(actions))
#         state = torch.stack([state1, state2], dim=0)
#         state = self.state_emb(state)
#         state = rearrange(state, "P ... D -> ... (P D)")
#         metadata = self.metadata_emb(metadata)
#
#         try:
#             memory_state = self.memory_state
#         except:
#             print("No memory states")
#             memory_state = None
#
#         z, self.memory_state = self.memory(
#             action_prev, state, metadata[..., None, :], h=memory_state
#         )
#         z = z[..., -1, :]
#
#         states_pred = self.predictor.unroll_keep_all(
#             actions_future, state[..., -1, :], metadata, z
#         )
#         z_pred, _ = self.memory(
#             actions_future,
#             states_pred,
#             utils.extend_as(metadata, states_pred.shape),
#             h=self.memory_state,
#         )
#         z_policy = self.policy(z_pred[..., -1:, :])
#
#         act, infos = self.action_decoder.sample(z_policy, temperature, topk)
#         infos = {k: v.cpu().numpy() for k, v in infos.items()}
#         return self.action_decoder.to_int(act).cpu().numpy(), infos
#
#     def save_with_config(self, path):
#         config = self.config
#         s = self.state_dict()
#         s["config"] = config
#         try:
#             torch.save(s, path)
#         except Exception as e:
#             print(e)
#

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

    #
    # config = BCModelConfig(
    #     memory_dim=512,
    #     memory_layers=3,
    #     policy_hidden_dims=1024,
    #     policy_dim=512,
    #     delay=4,
    # )
    # data = MeleeDataloader(
    #     seq_len=15 + 1,
    #     path="./data/mmaped/",
    #     data_types=["actions", "states", "metadata"],
    # )
    #
    # model = BCModel(config)
    #
    # actions, states, metadata, rewards = next(
    #     data.get_samples(
    #         batch_size=2,
    #         device="cpu",
    #         return_rewards=True,
    #         low_discrepancy_index=0,
    #     )
    # )
    # (
    #     actions,
    #     states1,
    #     states2,
    #     metadata,
    #     rewards,
    # ) = prepare_data(actions, states, metadata, rewards)
    #
    # states_future_ema = None  # model.embed_states(states1_future, states2_future)
    #
    # loss, infos = model.loss(actions, states1, states2, metadata, rewards)
    # print(loss)
    # print(infos)
