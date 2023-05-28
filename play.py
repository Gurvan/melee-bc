import torch
import time
import signal
import sys
import argparse
from pathlib import Path

import torch.multiprocessing as mp
import melee

from melee_utils import (
    StateManager,
    ConsoleMock,
    ControllerMock,
    process_action,
)
from models import BCModel
def is_damage_state(state):
    return (state >= 0x4B) * (state <= 0x5B)



def load_model(path):
    params = torch.load(path, map_location="cpu")
    model = BCModel(params["config"])
    s = {k: v for k, v in params.items() if k != "config"}

    model.load_state_dict(s, strict=True)
    model.eval()
    model = torch.compile(model)
    return model


class Agent:
    def __init__(self, path, device, delay=None, history_length=None):
        self.device = device
        self.model = load_model(path).to(device)
        self.model.share_memory()
        self.num_frames_per_stack = 1
        if delay is None:
            delay = self.model.config.delay
        else:
            self.model.config.delay = delay
        if history_length is None:
            history_length = delay + 1
        # assert (
        #     delay == self.model.config.delay
        # ), f"Supported delays: {self.model.config.delay}"
        print(f"Delay: {delay}")
        assert delay % self.num_frames_per_stack == 0
        assert history_length % self.num_frames_per_stack == 0
        self.delay = delay
        self.history_length = history_length
        self.stack_delay = delay // self.num_frames_per_stack
        self.stack_history_length = history_length // self.num_frames_per_stack
        self.state_manager = StateManager(
            seq_len=history_length,
            delay=delay,
        )

        self.prev_actions = 2 * [self.state_manager.neutral_action]

    def init_queues(self):
        action_queue = mp.Queue()
        return action_queue

    def prepare_input(self, actions, states1, states2, metadata):
        return (
            torch.from_numpy(actions).to(self.device),
            torch.from_numpy(states1).float().to(self.device),
            torch.from_numpy(states2).float().to(self.device),
            torch.from_numpy(metadata).int().to(self.device),
        )

    def update_state(self, gamestate):
        self.state_manager.update_state(gamestate)

    def compute_actions(self, action_queue):
        if self.state_manager.state_counter == 0:
            self.state_manager.state_counter += self.num_frames_per_stack
            context = self.prepare_input(*self.state_manager.get_both())
            t = 1
            if is_damage_state(context[1][-1][0][0]) or is_damage_state(context[2][-1][0][0]):
                t = 0.8
            actions, infos = self.model.inference(*context, temperature=t, topk=0)
            actions = [actions[:, i] for i in range(actions.shape[1])]
            for i, a in enumerate(actions):
                self.state_manager.update_action(a)
                action_queue.put((a, {k: v[:, i] for k, v in infos.items()}))


def init_console(dolphin_path, iso_path):

    console = melee.Console(
        path=str(dolphin_path),
        blocking_input=True,
        online_delay=0,
        fullscreen=False,
        disable_audio=True,
        save_replays=False,
    )
    controller = melee.Controller(
        console=console, port=1, type=melee.ControllerType.STANDARD
    )

    controller_opponent = melee.Controller(
        console=console, port=2, type=melee.ControllerType.STANDARD
    )

    # This isn't necessary, but makes it so that Dolphin will get killed when you ^C
    def signal_handler(sig, frame):
        console.stop()
        print("Shutting down cleanly...")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    # Run the console
    console.run(iso_path=iso_path)

    for i in range(2):
        time.sleep(0.1)
        print("Connecting to console..." + f"  {i}")
        if console.connect():
            print("Console connected")
            break

    print("Connecting controller to console...")
    if not controller.connect():
        print("ERROR: Failed to connect the controller.")
        sys.exit(-1)
    print("Controller connected")

    print("Connecting controller to console...")
    if not controller_opponent.connect():
        print("ERROR: Failed to connect the controller.")
        sys.exit(-1)
    print("Controller connected")

    return console, controller, controller_opponent


def init_console_mock():
    console, controller, controller_opponent = (
        ConsoleMock(),
        ControllerMock(),
        ControllerMock(),
    )
    return console, controller, controller_opponent


def play(
    console,
    controller,
    controller_opponent,
    agent: Agent,
    action_queue: mp.Queue,
):
    # Main loop
    while True:
        # "step" to the next frame
        gamestate = console.step()
        if gamestate is None:
            continue

        # The console object keeps track of how long your bot is taking to process frames
        #   And can warn you if it's taking too long
        if console.processingtime * 1000 > 12:
            print(
                "WARNING: Last frame took "
                + str(console.processingtime * 1000)
                + "ms to process."
            )

        # What menu are we in?
        if gamestate.menu_state in [melee.Menu.IN_GAME, melee.Menu.SUDDEN_DEATH]:
            agent.update_state(gamestate)
            agent.compute_actions(action_queue)
            actions, infos = action_queue.get()

            process_action(controller, actions[0])
            process_action(controller_opponent, actions[1])
        else:
            melee.MenuHelper.menu_helper_simple(
                gamestate,
                controller,
                melee.Character.MARTH,
                melee.Stage.FINAL_DESTINATION,
                "",
                autostart=True,
                swag=False,
            )
            melee.MenuHelper.menu_helper_simple(
                gamestate,
                controller_opponent,
                melee.Character.FOX,
                melee.Stage.FINAL_DESTINATION,
                costume=1,
                autostart=False,
                swag=False,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dolphin_dir",
        metavar="d",
        type=Path,
        help="directory containing the dolphin executable from libmelee",
        required=True,
    )
    parser.add_argument(
        "--iso_path",
        metavar="i",
        type=Path,
        help="path of the Melee 1.02 iso",
        required=True,
    )
    parser.add_argument(
        "--checkpoint",
        metavar="c",
        type=Path,
        help="path to the agent checkpoint",
        required=True,
    )
    parser.add_argument(
        "--gpu", action="store_true", default=False, help="run the agent opf the GPU"
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        default=False,
        help="run on a mock environment, for testing purpose.",
    )
    args = parser.parse_args()

    device = "cuda" if args.gpu else "cpu"
    agent = Agent(args.checkpoint, device)

    if args.mock:
        console, controller, controller_opponent = init_console_mock()
    else:
        console, controller, controller_opponent = init_console(
            args.dolphin_dir, args.iso_path
        )

    action_queue = agent.init_queues()
    play(
        console,
        controller,
        controller_opponent,
        agent,
        action_queue,
    )
