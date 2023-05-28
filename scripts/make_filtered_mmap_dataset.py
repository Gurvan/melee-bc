import os
from pathlib import Path
from IPython import embed
import numpy as np
from mmap_ninja import numpy as RaggedMmap


def is_damage_state(state):
    return (state >= 0x4B) * (state <= 0x5B)


def is_new_damage(state, timer):
    return (timer == 0) * is_damage_state(state)


path = Path("/media/DATA/Melee/FloatData/")

actions = RaggedMmap.open_existing(path / "mmap_actions")
states = RaggedMmap.open_existing(path / "mmap_states")
metadata = RaggedMmap.open_existing(path / "mmap_metadata")

print(actions.shape, states.shape, metadata.shape)

l = states.shape[0]
n = 500000

print(l // n)

is_d = []
indices0 = []
indices1 = []

for i, j in zip(range(0, l, n), list(range(0, l, n))[1:] + [l]):
    d = is_new_damage(states[i:j, :, 0], states[i:j, :, 1])
    indices0 += list(np.where(d[:, 1])[0])
    indices1 += list(np.where(d[:, 0])[0])
    print(i, j, l, j / l * 100)

idx0 = np.array(indices0)[:, None] + np.arange(65)[None, :] + 20
idx1 = np.array(indices1)[:, None] + np.arange(65)[None, :] + 20

while idx0[-1, -1] >= l:
    idx0 = idx0[:-1]
while idx1[-1, -1] >= l:
    idx1 = idx1[:-1]

os.makedirs(path / "filtered", exist_ok=True)

try:
    a = np.concatenate((actions[idx0], actions[idx1][..., [1, 0], :]))
    np.save(path / "filtered" / "actions", a)
    del a

    s = np.concatenate((states[idx0], states[idx1][..., [1, 0], :]))
    np.save(path / "filtered" / "states", s)
    del s

    m = np.concatenate((metadata[idx0], metadata[idx1][..., [0, 2, 1]]))
    np.save(path / "filtered" / "metadata", m)
    del m
except:
    embed()
