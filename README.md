# Melee Behavioral Cloning

This project aims at experimenting with  behavioral cloning (aka supervised learning) for making a Melee bot.

The goal is to see if simple behavioral cloning can produce non-trivial behavior, both for undelayed and delayed agent (with reaction time).

## Dataset

We use the [`SLP public dataset`](https://drive.google.com/file/d/1VqRECRNL8Zy4BFQVIHvoVGtfjz4fi9KC/view?usp=sharing) made of `Slippi (.slp) replays` (see [Project Slippi](https://github.com/project-slippi/project-slippi)).

The dataset was collected by [altf4](https://github.com/altf4) and contains around 100k replays from various sources.

## Play

In order to see the bot play you need to install [libmelee](https://github.com/altf4/libmelee).

The you can run:

`python play.py --dolphin_dir path_to_dolphin_bin_dir --iso_path path_to_melee_1.02_iso --checkpoint path_to_training_checkpoint`

with:

- `path_to_dolphin_bin_dir` as in libmelee.
- `path_to_training_checkpoint` like `checkpoints/melee-bc-1-ema-demo`.

Example of partially trained agents can be found in the `checkpoints` directory. The numbers in the filenames denotes the delay under which the bot was trained.

Demo bots:
 - `melee-bc-1-ema-demo` trained for about 2 hours on my GPU (RTX 2060 Super), on half the public dataset
 - ~~`melee-bc-16-ema-demo` trained for about 5 hours, on less than half the public dataset~~ (this one doesn't work after training became instable).

Both bots would greatly benefits from longer training, especially the 16 delay one.

Videos of `melee-bc-1-ema-demo` can be found [here](https://www.youtube.com/watch?v=vRCCnAzIiWU&list=PL2bzD8K5QN1vlATggNyPEg6_Jl96A2SSj).


## Train

In order to train of finetune the bot you need to convert the `slp` files to a more efficient format.

Two scripts are provided in the `scripts` directory:

- `parse_slp_to_npy.go` is a `golang` scripts the cinverts the `slp` replays to Nmpuy for `npy`.
- `make_mmap_dataset.py` convert the `npy` files to a memory mapped data format that is very efficient to load and speeds up training a lot.

See the `Scripts` section.


Once you converted the dataset you can train with `python train.py`. I suggest you have a look at the end of `train.py` so you can have a look at the training options (like delay).

## Scripts

`go run parse_slp_to_npy.go -i folder_containing_slp -o output_dir -N number_of_cpu_threads`

This will take a while if you process the entire public dataset. The `slp` files can be in multiple subfolders.
Using multiple threads really speeds things up, but only up until the point where the conversion becomes memory bound.



`python scripts/make_mmap_dataset.py npy_files_dir output_dir`

`npy_file_dir` should be the output folder from the go script, and `output_dir` is the directory from where the data will be loaded by the training script.


## Notes

- You need an NVIDIA GPU to train. By default `fp16` training is used, which requires cards `> 20XX`. You can disable `fp16` training in `train.py`.
- You can increase training and inference speed if you install PyTorch 2 instead of 1.13, as it can compile as specific function in `utils.py` (`forget_mult`). I get about 10% speedup using PyTorch 2.
