from typing import List
import os
from pathlib import Path
import argparse
import shutil

import numpy as np
from einops import rearrange
from mmap_ninja import numpy as RaggedMmap

actions_size = 12
states_size = 8


def get_files(datapath: Path):
    return [f for f in datapath.iterdir() if f.is_file() and f.suffix == ".npy"]


def get_lengths(files: List[Path]):
    # return [(len(np.load(f)) - 3) // 40 for f in files]
    return [
        (((os.stat(f).st_size - 86) // 2) - 3)
        // (2 * 2 * (actions_size + 1 + states_size))
        for f in files
    ]


def reshape(x):
    metadata = x[None, :3]
    actions_and_states = rearrange(
        x[3:],
        "(time player Z) -> time player Z",
        player=2,
        Z=actions_size + 1 + states_size,
    )
    actions, characters, states = (
        actions_and_states[..., :actions_size],
        actions_and_states[..., actions_size],
        actions_and_states[..., actions_size + 1 :],
    )
    assert states.shape[-1] == states_size
    metadata = np.repeat(metadata[:, :1], len(actions), axis=0)
    metadata = np.concatenate([metadata, characters], axis=-1)
    return actions, states, metadata


def get_actions(x):
    return reshape(x)[0]


def get_states(x):
    return reshape(x)[1]


def get_metadata(x):
    return reshape(x)[2]


def generate(files: List[Path], files_len: List[int], extractor=reshape):
    for f, fl in zip(files, files_len):
        try:
            x = np.load(f, mmap_mode="r")
            yield (*extractor(x), fl)
        except Exception as e:
            # os.remove(f)
            # print(f, e)
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Define the arguments
    parser.add_argument("npy_dir", help="directory containing '.npy' files", required=True)
    parser.add_argument("output_dir", help="directory to store the memory mapped data", required=True)
    
    # Parse the command line arguments
    args = parser.parse_args()

    inpath = Path(args.npy_dir)
    outpath = Path(args.output_dir)

    os.makedirs(outpath, exist_ok=True)


    files = get_files(inpath)
    files_len = get_lengths(files)

    total_len = sum(files_len)

    actions_mmap = RaggedMmap.empty(
        out_dir=outpath / "mmap_actions_tmp",
        dtype="int8",
        shape=(total_len, 2, actions_size),
        order="C",
    )

    states_mmap = RaggedMmap.empty(
        out_dir=outpath / "mmap_states_tmp",
        dtype="float32",
        shape=(total_len, 2, states_size),
        order="C",
    )

    metadata_mmap = RaggedMmap.empty(
        out_dir=outpath / "mmap_metadata_tmp",
        dtype="int16",
        shape=(total_len, 3),
        order="C",
    )

    valid_files_len = []
    start = 0
    end = 0
    for i, (actions, states, metadata, file_len) in enumerate(
        generate(files, files_len)
    ):
        try:
            # assert files_len[i] == len(actions), f"{files_len[i]} != {len(actions)}"
            assert file_len == len(actions), f"{files_len[i]} != {len(actions)}"
            valid_files_len.append(len(actions))
            print(f"{start} / {total_len} | {100 * start/total_len:.2f}%")
            end = start + len(actions)
            actions_mmap[start:end] = actions
            states_mmap[start:end] = states
            metadata_mmap[start:end] = metadata
            start = end
        except Exception as e:
            # print(e)
            continue

        if (i + 1) % 100 == 0:
            actions_mmap.flush()
            states_mmap.flush()
            metadata_mmap.flush()

    actions_mmap.flush()
    states_mmap.flush()
    metadata_mmap.flush()

    file_offsets = np.cumsum([0] + valid_files_len)[:-1]
    np.save(outpath / "file_offsets.npy", file_offsets)

    # Remove excess length from invalid files
    RaggedMmap.from_ndarray(outpath / "mmap_actions", actions_mmap[:end])
    RaggedMmap.from_ndarray(outpath / "mmap_states", states_mmap[:end])
    RaggedMmap.from_ndarray(outpath / "mmap_metadata", metadata_mmap[:end])



    shutil.rmtree(outpath / "mmap_actions_tmp")
    shutil.rmtree(outpath / "mmap_states_tmp")
    shutil.rmtree(outpath / "mmap_metadata_tmp")
