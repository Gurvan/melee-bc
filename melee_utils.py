import time

import numpy as np
import melee

from dataset import (
    NUM_CHARACTERS,
    NUM_STAGES,
    MAX_ACTIONSTATE,
    MAX_SHIELD,
    MAX_JUMPS,
)


def action_numpy2melee(action):
    sx = action[0] / 80.0
    sy = action[1] / 80.0
    cx = action[2] / 80.0
    cy = action[3] / 80.0
    trig = action[4] / 140.0
    z = bool(action[5] > 0)
    l = bool(action[6] > 0)
    r = bool(action[7] > 0)
    a = bool(action[8] > 0)
    b = bool(action[9] > 0)
    x = bool(action[10] > 0)
    y = bool(action[11] > 0)
    return sx, sy, cx, cy, trig, z, l, r, a, b, x, y


def clamp_state(state):
    state[..., 0] = state[..., 0].clip(0, MAX_ACTIONSTATE)
    state[..., 1] = state[..., 1].clip(0, 119.0)
    state[..., 2] = state[..., 2].clip(0, 300.0)
    state[..., 3] = state[..., 3].clip(-1.0, 1.0)
    state[..., 4] = state[..., 4].clip(-275.0, 275.0)
    state[..., 5] = state[..., 5].clip(-170.0, 340.0)
    state[..., 6] = state[..., 6].clip(0, MAX_SHIELD)
    state[..., 7] = state[..., 7].clip(0, MAX_JUMPS)
    return state


# asID | asFrame | Damage | Direction | PosX | PosY | shieldHealth | JumpsRemaining
def player_melee2numpy(pl: melee.PlayerState):
    actionstate = min(pl.action.value, 392)
    actionframe = pl.action_frame
    damage = pl.percent
    direction = 1.0 if pl.facing else -1.0
    pos_x = pl.position.x
    pos_y = pl.position.y
    shield = pl.shield_strength
    jumps = pl.jumps_left

    character = np.clip(pl.character.value, 0, NUM_CHARACTERS - 1)
    state = np.array(
        [
            actionstate,
            actionframe,
            damage,
            direction,
            pos_x,
            pos_y,
            shield,
            jumps,
        ]
    )
    return (
        character,
        clamp_state(state),
    )


INTERNAL_NO_STAGE = 0
INTERNAL_FINAL_DESTINATION = 0x19
INTERNAL_BATTLEFIELD = 0x18
INTERNAL_POKEMON_STADIUM = 0x12
INTERNAL_DREAMLAND = 0x1A
INTERNAL_FOUNTAIN_OF_DREAMS = 0x8
INTERNAL_YOSHIS_STORY = 0x6


def convert_stage(internal_stage):
    if internal_stage == INTERNAL_NO_STAGE:
        return 0
    elif internal_stage == INTERNAL_FINAL_DESTINATION:
        return 0x20
    elif internal_stage == INTERNAL_BATTLEFIELD:
        return 0x1F
    elif internal_stage == INTERNAL_POKEMON_STADIUM:
        return 0x12
    elif internal_stage == INTERNAL_DREAMLAND:
        return 0x1C
    elif internal_stage == INTERNAL_FOUNTAIN_OF_DREAMS:
        return 0x02
    elif internal_stage == INTERNAL_YOSHIS_STORY:
        return 0x08
    else:
        return 0x1F


def gamestate_melee2numpy(gs: melee.GameState, port_self=1, port_opponent=2):
    stage = np.clip(convert_stage(gs.stage), 0, NUM_STAGES - 1)
    character_self, state_self = player_melee2numpy(gs.players[port_self])
    character_opponent, state_opponent = player_melee2numpy(gs.players[port_opponent])
    metadata = np.array([stage, character_self, character_opponent], dtype=np.int32)
    return state_self, state_opponent, metadata


class StateManager:
    def __init__(
        self,
        seq_len,
        delay=0,
        port_self=1,
        port_opponent=2,
    ):
        self.seq_len = seq_len
        self.delay = delay
        self.port_self = port_self
        self.port_opponent = port_opponent
        self.states1 = (self.seq_len - self.delay) * [self.neutral_state]
        self.states2 = (self.seq_len - self.delay) * [self.neutral_state]
        self.metadata = np.zeros(3, dtype=np.int32)
        self.actions1 = self.seq_len * [self.neutral_action]
        self.actions2 = self.seq_len * [self.neutral_action]
        self.state_counter = 0
        self.action_counter = 0

    @property
    def neutral_action(self):
        # return np.array([58, 58, 58, 58, 0, 0])
        return np.array(12 * [0])

    @property
    def neutral_state(self):
        return np.array([0x142, -1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    def update_state(self, gamestate):
        state1, state2, metadata = gamestate_melee2numpy(
            gamestate, self.port_self, self.port_opponent
        )

        self.states1.append(state1)
        self.states2.append(state2)
        self.metadata = metadata

        if len(self.states1) > self.seq_len - self.delay:
            del self.states1[0]
        if len(self.states2) > self.seq_len - self.delay:
            del self.states2[0]

        if self.state_counter > 0:
            self.state_counter -= 1
        # print("update state:", len(self.states1), self.state_counter)

    def update_action(self, actions):
        action1, action2 = actions
        self.actions1.append(action1)
        self.actions2.append(action2)

        if len(self.actions1) > self.seq_len:
            del self.actions1[0]
        if len(self.actions2) > self.seq_len:
            del self.actions2[0]

        if self.action_counter > 0:
            self.action_counter -= 1
        # print("update actions:", len(self.actions1), self.action_counter)

    def get_both(self, reverse=False):
        states1 = np.stack(self.states1 + self.delay * [self.neutral_state], axis=0)
        states2 = np.stack(self.states2 + self.delay * [self.neutral_state], axis=0)
        actions1 = np.stack(self.actions1, axis=0)
        actions2 = np.stack(self.actions2, axis=0)
        metadata = self.metadata
        metadata_flip = np.array([self.metadata[0], self.metadata[2], self.metadata[1]])

        if not reverse:
            states_both1 = np.stack([states1, states2], axis=0)
            states_both2 = np.stack([states2, states1], axis=0)
            actions_both = np.stack([actions1, actions2], axis=0)
            metadata_both = np.stack([metadata, metadata_flip], axis=0)
        else:
            states_both1 = np.stack([states2, states1], axis=0)
            states_both2 = np.stack([states1, states2], axis=0)
            actions_both = np.stack([actions2, actions1], axis=0)
            metadata_both = np.stack([metadata_flip, metadata], axis=0)
        return actions_both, states_both1, states_both2, metadata_both

    def __len__(self):
        return len(self.actions1)


class ConsoleMock:
    def __init__(self):
        self.processingtime = 0

    def step(self):
        gs = melee.GameState()
        gs.players[1] = melee.PlayerState()
        gs.players[2] = melee.PlayerState()
        gs.players[1].character = melee.Character.FOX
        gs.players[2].character = melee.Character.FOX
        gs.players[1].action = melee.Action.STANDING
        gs.players[2].action = melee.Action.STANDING
        gs.menu_state = melee.Menu.IN_GAME
        time.sleep(1 / 60.0)
        return gs


class ControllerMock:
    def release_all(self):
        pass

    def tilt_analog_unit(self, b, sx, sy):
        pass

    def press_shoulder(self, b, trig):
        pass

    def press_button(self, b):
        pass

    def flush(self):
        pass


def process_action(controller: melee.Controller | ControllerMock, action_raw):
    sx, sy, cx, cy, trig, z, l, r, a, b, x, y = action_numpy2melee(
        action_raw  # .detach().cpu()
    )
    controller.release_all()
    controller.tilt_analog_unit(melee.Button.BUTTON_MAIN, sx, sy)
    controller.tilt_analog_unit(melee.Button.BUTTON_C, cx, cy)
    controller.press_shoulder(melee.Button.BUTTON_L, trig)
    if z:
        controller.press_button(melee.Button.BUTTON_Z)
    if l:
        controller.press_button(melee.Button.BUTTON_L)
    if r:
        controller.press_button(melee.Button.BUTTON_R)
    if a:
        controller.press_button(melee.Button.BUTTON_A)
    if b:
        controller.press_button(melee.Button.BUTTON_B)
    if x:
        controller.press_button(melee.Button.BUTTON_X)
    if y:
        controller.press_button(melee.Button.BUTTON_Y)
    controller.flush()
