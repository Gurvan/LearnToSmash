import random
from math import sqrt
from copy import deepcopy
import numpy as np

from ssbm_gym.ssbm_env import BaseEnv, isDying

from .spaces import ActionSpace
from .embed import EmbedGame, DummyEmbedGame

def make_env(frame_limit, options, dummy=False):
    def _init():
        try:
            p2 = options['player2'] == 'ai'
        except:
            p2 = False
        if p2:
            if not dummy:
                print('SelfPlayEnv')
                env = SelfPlayEnv(frame_limit=frame_limit, options=options)
            else:
                print('DummySelfPlayEnv')
                env = DummySelfPlayEnv(frame_limit=frame_limit, options=options)
            return env
        else:
            if not dummy:
                print('Env')
                env = Env(frame_limit=frame_limit, options=options)
            else:
                print('DummyEnv')
                env = DummyEnv(frame_limit=frame_limit, options=options)
            return env
    return _init


def EnvVec(num_envs, frame_limit=1e12, options={}, dummy=False):
    if type(options) == dict:
        return SubprocVecEnv([make_env(frame_limit=frame_limit, options=options, dummy=dummy) for _ in range(num_envs)])
    if type(options) == list:
        assert(len(options) != 0)
        assert(len(options) <= num_envs)
        if len(options) < num_envs:
            k = round(num_envs / len(options)) + 1
            options = (k * options)[:num_envs]

        return SubprocVecEnv([make_env(frame_limit=frame_limit, options=options[i], dummy=dummy) for i in range(num_envs)])


class Env(BaseEnv):
    def __init__(self, **kwargs):
        BaseEnv.__init__(self, **kwargs)
        self._embed_obs = EmbedGame()
        self.player_in_control = [True, True]
        self.fake_reset_frame = 0

    @property
    def action_space(self):
        if self._action_space is not None:
            return self._action_space
        else:
            self._action_space = ActionSpace()
            return self._action_space

    @property
    def observation_space(self):
        if self._observation_space is not None:
            return self._observation_space
        else:
            self._observation_space = self._embed_obs
            return self._embed_obs

    def embed_obs(self, obs):
        return self._embed_obs(obs)

    def compute_reward(self):
        died = [0, 0]
        damage = [0, 0]
        if self.prev_obs is not None:
            for pid in [self.pid, 1-self.pid]:
                # Compute if player is in control to prevent rewarding opponent SDs
                self.player_in_control[pid] = (self.player_in_control[pid] \
                                            or is_in_control(self.obs.players[pid].action_state)) \
                                            and (int(self.obs.players[pid].hitstun_frames_left) <= 0)

                died[pid] = not isDying(self.prev_obs.players[pid]) and isDying(self.obs.players[pid])
                damage[pid] = min(max(0, self.obs.players[pid].percent - self.prev_obs.players[pid].percent), 15)

        r = 1.0 * (died[1-self.pid] * (not self.player_in_control[1-self.pid]) - died[self.pid]) + 0.01 * (damage[1-self.pid] - damage[self.pid])
        return r

    def is_terminal(self):
        return self.obs.frame - self.fake_reset_frame >= self.frame_limit

    def act(self, action):
        return self.action_space.from_index(action)

    def step(self, action):
        if self.obs is not None:
            self.prev_obs = deepcopy(self.obs)
        
        obs = self.api.step([self.act(action)])
        self.obs = obs
        reward = self.compute_reward()
        done = self.is_terminal() or self.obs.players[0].controller.button_START
        if done:
            self.fake_reset_frame = self.obs.frame

        return self.embed_obs(obs), reward, done, None


class DummyEnv():
    def __init__(self, frame_limit = 100000, pid = 0, options = {}):
        self.api = None
        self.frame = 0
        self.frame_limit = frame_limit
        self.pid = pid  # player id
        self.obs = None
        self.prev_obs = None
        self._action_space = None
        self._observation_space = None
        self._embed_obs = DummyEmbedGame()

    @property
    def action_space(self):
        if self._action_space is not None:
            return self._action_space
        else:
            self._action_space = ActionSpace()
            return self._action_space

    @property
    def observation_space(self):
        if self._observation_space is not None:
            return self._observation_space
        else:
            self._observation_space = self._embed_obs
            return self._embed_obs

    def embed_obs(self, obs):
        return self._embed_obs(obs)

    def compute_reward(self):
        return random.gauss(0, 1)

    def is_terminal(self):
        return self.frame >= self.frame_limit

    def reset(self):
        self.frame = 0
        return self._embed_obs(None)

    def act(self, action):
        return self.action_space.from_index(action)

    def step(self, action):
        self.frame += 1
        if self.obs is not None:
            self.prev_obs = deepcopy(self.obs)
        
        action = self.act(action)
        obs = None
        self.obs = obs
        reward = self.compute_reward()
        infos = dict({'frame': self.frame, 'damage/death': [0,0,0,0]})
        done = self.is_terminal()
        if done:
            self.frame = 0

        return self.embed_obs(self.obs), reward, done, infos

    def close(self):
        pass


class SelfPlayEnv(BaseEnv):
    def __init__(self, **kwargs):
        BaseEnv.__init__(self, **kwargs)
        self._embed_obs = EmbedGame()
        self.player_in_control = [True, True]
        self.fake_reset_frame = 0

    @property
    def action_space(self):
        if self._action_space is not None:
            return self._action_space
        else:
            self._action_space = ActionSpace()
            return self._action_space

    @property
    def observation_space(self):
        if self._observation_space is not None:
            return self._observation_space
        else:
            self._observation_space = self._embed_obs
            return self._embed_obs

    def embed_obs(self, obs):
        return self._embed_obs(obs)

    def compute_reward(self):
        died = [0, 0]
        damage = [0, 0]
        if self.prev_obs is not None:
            for pid in [self.pid, 1-self.pid]:
                # Compute if player is in control to prevent rewarding opponent SDs
                self.player_in_control[pid] = (self.player_in_control[pid] \
                                            or is_in_control(self.obs.players[pid].action_state)) \
                                            and (int(self.obs.players[pid].hitstun_frames_left) <= 0)

                died[pid] = not isDying(self.prev_obs.players[pid]) and isDying(self.obs.players[pid])
                damage[pid] = min(max(0, self.obs.players[pid].percent - self.prev_obs.players[pid].percent), 15)

        r0 = 1.0 * (died[1-self.pid] * (not self.player_in_control[1-self.pid]) - died[self.pid])   + 0.01 * (damage[1-self.pid] - damage[self.pid])
        r1 = 1.0 * (  died[self.pid] * (not self.player_in_control[self.pid])   - died[1-self.pid]) + 0.01 * (damage[self.pid] - damage[1-self.pid])
        
        ## Bonus moved in learner.py
        # Add a bonus for proximity to help the agent at the start of learning
        # x0, y0, x1, y1 = self.obs.players[self.pid].x, self.obs.players[self.pid].y, self.obs.players[1-self.pid].x, self.obs.players[1-self.pid].y
        # d = sqrt((x0-x1)**2 + (y0-y1)**2)
        # r0 += 1/60 * 0.03 * 1/(1 + 0.1 * d)
        # r1 += 1/60 * 0.03 * 1/(1 + 0.1 * d)
        return r0, r1

    def is_terminal(self):
        return self.obs.frame - self.fake_reset_frame >= self.frame_limit

    def act(self, action):
        return self.action_space.from_index(action)

    def step(self, actions):
        if self.obs is not None:
            self.prev_obs = deepcopy(self.obs)
        
        obs = self.api.step([self.act(a) for a in actions])
        self.obs = obs
        reward = self.compute_reward()
        done = self.is_terminal() or self.obs.players[0].controller.button_START
        if done:
            self.fake_reset_frame = self.obs.frame

        return self.embed_obs(obs), reward, done, None


class DummySelfPlayEnv():
    def __init__(self, frame_limit = 100000, pid = 0, options = {}):
        self.api = None
        self.frame = 0
        self.frame_limit = frame_limit
        self.pid = pid  # player id
        self.obs = None
        self.prev_obs = None
        self._action_space = None
        self._observation_space = None
        self._embed_obs = DummyEmbedGame()

    @property
    def action_space(self):
        if self._action_space is not None:
            return self._action_space
        else:
            self._action_space = ActionSpace()
            return self._action_space

    @property
    def observation_space(self):
        if self._observation_space is not None:
            return self._observation_space
        else:
            self._observation_space = self._embed_obs
            return self._embed_obs

    def embed_obs(self, obs):
        return self._embed_obs(obs)

    def compute_reward(self):
        return [random.gauss(0, 1), random.gauss(0, 1)]

    def is_terminal(self):
        return self.frame >= self.frame_limit

    def reset(self):
        self.frame = 0
        return self._embed_obs(None)

    def act(self, action):
        return self.action_space.from_index(action)

    def step(self, actions):
        self.frame += 1
        if self.obs is not None:
            self.prev_obs = deepcopy(self.obs)
        
        actions = [self.act(a)  for a in actions]
        obs = None
        self.obs = obs
        reward = self.compute_reward()
        # infos = dict({'frame': self.frame, 'damage/death': [0,0,0,0]})
        done = self.is_terminal()
        if done:
            self.frame = 0

        return self.embed_obs(self.obs), reward, done, None
        # return self.embed_obs(self.obs), reward, done, infos

    def close(self):
        pass



import enum

# From Slippi replay parser
class State(enum.IntEnum):
  # Animation ID ranges
  DAMAGE_START = 0x4B
  DAMAGE_END = 0x5B
  CAPTURE_START = 0xDF
  CAPTURE_END = 0xE8
  GUARD_START = 0xB2
  GUARD_END = 0xB6
  GROUNDED_CONTROL_START = 0xC  # 0xE
  GROUNDED_CONTROL_END = 0x18
  SQUAT_START = 0x27
  SQUAT_END = 0x29
  DOWN_START = 0xB7
  DOWN_END = 0xC6
  TECH_START = 0xC7
  TECH_END = 0xCC
  DYING_START = 0x0
  DYING_END = 0xA
  CONTROLLED_JUMP_START = 0x18
  CONTROLLED_JUMP_END = 0x22
  GROUND_ATTACK_START = 0x2C
  GROUND_ATTACK_END = 0x40
  CLIFF_START = 0xFC
  CLIFF_END = 0x107

  # Animation ID specific
  ROLL_FORWARD = 0xE9
  ROLL_BACKWARD = 0xEA
  SPOT_DODGE = 0xEB
  AIR_DODGE = 0xEC
  ACTION_WAIT = 0xE
  ACTION_DASH = 0x14
  ACTION_KNEE_BEND = 0x18
  GUARD_ON = 0xB2
  TECH_MISS_UP = 0xB7
  TECH_MISS_DOWN = 0xBF
  DASH = 0x14
  TURN = 0x12
  LANDING_FALL_SPECIAL = 0x2B
  JUMP_FORWARD = 0x19
  JUMP_BACKWARD = 0x1A
  FALL_FORWARD = 0x1E
  FALL_BACKWARD = 0x1F
  GRAB = 0xD4

def is_in_control(state):
    ground = state >= State.GROUNDED_CONTROL_START and state <= State.GROUNDED_CONTROL_END
    squat = state >= State.SQUAT_START and state <= State.SQUAT_END
    groundAttack = state > State.GROUND_ATTACK_START and state <= State.GROUND_ATTACK_END
    isGrab = state == State.GRAB
    cliff = state >= State.CLIFF_START and state <= State.CLIFF_END
    return ground or squat or groundAttack or isGrab or cliff



import multiprocessing
import cloudpickle
import pickle

class CloudpickleWrapper(object):
    def __init__(self, var):
        self.var = var

    def __getstate__(self):
        return cloudpickle.dumps(self.var)

    def __setstate__(self, obs):
        self.var = pickle.loads(obs)


def _worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.var()
    while True:
        try:
            cmd, data = remote.recv()
            if cmd == 'step':
                observation, reward, done, info = env.step(data)
                # if done:
                #     # save final observation where user can get it, then reset
                #     # info['terminal_observation'] = observation
                remote.send((observation, reward, done, info))
            elif cmd == 'reset':
                observation = env.reset()
                remote.send(observation)
            elif cmd == 'close':
                remote.close()
                break
            elif cmd == 'get_spaces':
                remote.send((env.observation_space, env.action_space))
            elif cmd == 'env_method':
                method = getattr(env, data[0])
                remote.send(method(*data[1], **data[2]))
            elif cmd == 'get_attr':
                remote.send(getattr(env, data))
            elif cmd == 'set_attr':
                remote.send(setattr(env, data[0], data[1]))
            else:
                raise NotImplementedError
        except EOFError:
            break


class SubprocVecEnv(): 
    def __init__(self, env_fns, start_method='spawn'):
        self.num_envs = len(env_fns)
        self.waiting = False
        self.closed = False

        if start_method is None:
            # Fork is not a thread safe method (see issue #217)
            # but is more user friendly (does not require to wrap the code in
            # a `if __name__ == "__main__":`)
            forkserver_available = 'forkserver' in multiprocessing.get_all_start_methods()
            start_method = 'forkserver' if forkserver_available else 'spawn'
        ctx = multiprocessing.get_context(start_method)

        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(self.num_envs)])
        self.processes = []
        for work_remote, remote, env_fn in zip(self.work_remotes, self.remotes, env_fns):
            args = (work_remote, remote, CloudpickleWrapper(env_fn))
            # daemon=True: if the main process crashes, we should not cause things to hang
            process = ctx.Process(target=_worker, args=args, daemon=True)
            process.start()
            self.processes.append(process)
            work_remote.close()

        self.remotes[0].send(('get_spaces', None))
        self.observation_space, self.action_space = self.remotes[0].recv()

    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True


    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return obs, rews, dones, infos


    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        obs = [remote.recv() for remote in self.remotes]
        return obs


    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for process in self.processes:
            process.join()
        self.closed = True


    def get_attr(self, attr_name, indices=None):
        """Return attribute from vectorized environment (see base class)."""
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(('get_attr', attr_name))
        return [remote.recv() for remote in target_remotes]


    def set_attr(self, attr_name, value, indices=None):
        """Set attribute inside vectorized environments (see base class)."""
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(('set_attr', (attr_name, value)))
        for remote in target_remotes:
            remote.recv()


    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        """Call instance methods of vectorized environments."""
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(('env_method', (method_name, method_args, method_kwargs)))
        return [remote.recv() for remote in target_remotes]


    def _get_target_remotes(self, indices):
        """
        Get the connection object needed to communicate with the wanted
        envs that are in subprocesses.

        :param indices: (None,int,Iterable) refers to indices of envs.
        :return: ([multiprocessing.Connection]) Connection object to communicate between processes.
        """
        indices = self._get_indices(indices)
        return [self.remotes[i] for i in indices]

