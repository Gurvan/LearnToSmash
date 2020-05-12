# -*- coding: utf-8 -*-
from collections import deque
import random
import torch
import numpy as np
from ssbm.envs import EnvVec, Env as SSBMEnv, DummyEnv, SelfPlayEnv as SSBMSelfPlayEnv, DummySelfPlayEnv



default_options = dict(
    speed=0,
    render=False,
    player1='ai',
    player2='cpu',
    char1='falcon',
    char2='falcon',
    cpu2=2,
    stage='battlefield',
)

def Env(args, device, options=None, dummy=False, rank=0):
    if options is None:
        options = default_options
    if options['player2'] in ['ai', 'human']:
        return SelfPlayEnv(args=args, device=device, options=options, dummy=dummy, rank=rank)
    else:
        return CPUEnv(args=args, device=device, options=options, dummy=dummy, rank=rank)


class CPUEnv():
    def __init__(self, args, device, options=None, dummy=False, rank=0):
        try:
            n_actors = args.n_actors_per_worker
        except:
            n_actors = 1
        if options is None:
            options = default_options
        if type(options) == list:
            options = options[:n_actors]
        try:
            duration = args.duration
        except:
            duration = 1e10
        self.device = device
        self.env = EnvVec(n_actors, frame_limit=duration, options=options, dummy=dummy)
        self.action_dim = self.env.action_space.n
        self.act_every = args.act_every 
        self.observation_buffer = deque([], maxlen=args.act_every)
        self.observation_dim = self.env.observation_space.n
        self.action_space = self.env.action_space

    def reset(self):
        if len(self.observation_buffer) == 0:
            observation = self.env.reset()
            observation = torch.tensor(observation, dtype=torch.float32, device=self.device)
            self.observation_buffer.append(observation)
            observation = torch.stack(list(self.observation_buffer), dim=0)
        else:
            observation = self.observation_buffer[-1].unsqueeze(0)
        return observation

    def step(self, action):
        reward, done = np.zeros(self.env.num_envs), False
        for t in range(self.act_every):
            observation, r, done, _ = self.env.step(action)
            observation = torch.tensor(observation, dtype=torch.float32, device=self.device)
            self.observation_buffer.append(observation)
            reward += np.array(r).T.flatten()
            if done[0]:
                break
        reward = np.array([reward])
        observation = torch.stack(list(self.observation_buffer), dim=0)
        return observation, reward, done

    def close(self):
        self.env.close()


class SelfPlayEnv():
    def __init__(self, args, device, options=None, dummy=False, rank=0):
        try:
            n_actors = args.n_actors_per_worker
        except:
            n_actors = 1
        if options is None:
            options = default_options
            options['player1'] = 'ai'
            options['player2'] = 'ai'
            options['char2'] = options['char1']
        if type(options) == list:
            options = options[:n_actors]
            for option in options:
                option['player1'] = 'ai'
                option['player2'] = 'ai'
                option['char2'] = options['char1']
        try:
            duration = args.duration
        except:
            duration = 1e10
        self.device = device
        self.env = EnvVec(n_actors, frame_limit=duration, options=options, dummy=dummy)
        self.action_dim = self.env.action_space.n
        self.act_every = args.act_every 
        self.observation_buffer = deque([], maxlen=args.act_every)
        self.observation_dim = self.env.observation_space.n
        self.action_space = self.env.action_space

    def reset(self):
        if len(self.observation_buffer) == 0:
            observation = self.env.reset()
            observation1 = torch.tensor(observation, dtype=torch.float32, device=self.device)
            observation2 = torch.cat(tuple(reversed(observation1.chunk(2, dim=-1))), dim=-1)
            observation = torch.cat([observation1, observation2], dim=0)
            self.observation_buffer.append(observation)
            observation = torch.stack(list(self.observation_buffer), dim=0)
        else:
            observation = self.observation_buffer[-1].unsqueeze(0)
        return observation

    def step(self, action):
        reward, done = np.zeros(2 * self.env.num_envs), False
        action = np.array(action).reshape(2, -1).T
        for t in range(self.act_every):
            observation, r, done, _ = self.env.step(action)
            observation1 = torch.tensor(observation, dtype=torch.float32, device=self.device)
            observation2 = torch.cat(tuple(reversed(observation1.chunk(2, dim=-1))), dim=-1)
            observation = torch.cat([observation1, observation2], dim=0)
            self.observation_buffer.append(observation)
            reward += np.array(r).T.flatten()
            if done[0]:
                break
        reward = np.array([reward])
        observation = torch.stack(list(self.observation_buffer), dim=0)
        return observation, reward, done

    def close(self):
        self.env.close()


class SimpleEnv():
    def __init__(self, args, device, options=None, dummy=False, rank=0):
        n_actors = 1
        if options is None:
            options = default_options
        if type(options) == list:
            options = options[0]
            options['cpu2'] = rank%8 + 1
            if rank%2 == 0:
                options['iso'] = "/home/gurvan/Melee/ISOs/SSBM, 20XXHP 4.07.UCF.iso"
        try:
            duration = args.duration
        except:
            duration = 1e10
        self.device = device
        if not dummy:
            self.env = SSBMEnv(frame_limit=duration, options=options)
        else:
            self.env = DummyEnv(frame_limit=duration, options=options)
        self.action_dim = self.env.action_space.n
        self.act_every = args.act_every 
        self.observation_buffer = deque([], maxlen=args.act_every)
        self.observation_dim = self.env.observation_space.n


    def reset(self):
        if len(self.observation_buffer) == 0:
            observation = self.env.reset()
            observation = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
            self.observation_buffer.append(observation)
            observation = torch.stack(list(self.observation_buffer), dim=0)
        else:
            observation = self.observation_buffer[-1].unsqueeze(0)
        return observation

    def step(self, action):
        # reward, done = np.array(self.env.num_envs * [0.]), False
        reward, done = [], False
        for t in range(self.act_every):
            observation, r, done, _ = self.env.step(action[0])
            observation = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
            self.observation_buffer.append(observation)
            # reward += r
            reward.append(r)
            if done:
                break
        reward = np.array(reward)[:, None]
        observation = torch.stack(list(self.observation_buffer), dim=0)
        return observation, reward, done

    def close(self):
        self.env.close()


class SelfPlaySimpleEnv():
    def __init__(self, args, device, options=None, dummy=False, rank=None):
        if options is None:
            options = default_options
        if type(options) == list:
            options = options[0]
            options['player1'] = 'ai'
            options['player2'] = 'ai'
            options['char2'] = options['char1']
        try:
            duration = args.duration
        except:
            duration = 1e10
        self.device = device
        if not dummy:
            self.env = SSBMSelfPlayEnv(frame_limit=duration, options=options)
        else:
            self.env = DummySelfPlayEnv(frame_limit=duration, options=options)
        self.action_dim = self.env.action_space.n
        self.act_every = args.act_every 
        self.observation_buffer = deque([], maxlen=args.act_every)
        self.observation_dim = self.env.observation_space.n


    def reset(self):
        if len(self.observation_buffer) == 0:
            observation = self.env.reset()
            observation1 = torch.tensor(observation, dtype=torch.float32, device=self.device)
            observation2 = torch.cat(tuple(reversed(observation1.chunk(2, dim=-1))), dim=-1)
            observation = torch.stack([observation1, observation2], dim=0)
            self.observation_buffer.append(observation)
            observation = torch.stack(list(self.observation_buffer), dim=0)
        else:
            observation = self.observation_buffer[-1].unsqueeze(0)
        return observation

    def step(self, action):
        # reward, done = np.array(2 * [0.]), False
        reward, done = [], False
        for t in range(self.act_every):
            observation, r, done, _ = self.env.step(action)
            observation1 = torch.tensor(observation, dtype=torch.float32, device=self.device)
            observation2 = torch.cat(tuple(reversed(observation1.chunk(2, dim=-1))), dim=-1)
            observation = torch.stack([observation1, observation2], dim=0)
            self.observation_buffer.append(observation)
            # reward += np.array([r, -r])
            reward.append(r)
            if done:
                break
        reward = np.array(reward)#[:, None]
        # print(reward.shape)
        observation = torch.stack(list(self.observation_buffer), dim=0)
        return observation, reward, done

    def close(self):
        self.env.close()




if __name__ == "__main__":
    import numpy as np
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--act-every', default=int(6))
    parser.add_argument('--duration', default=int(100))
    parser.add_argument('--n-actors-per-worker', default=int(2))
    parser.add_argument('--selfplay', action='store_true', default=False)
    parser.add_argument('--dummy', action='store_true', default=False)
    args = parser.parse_args()

    options = dict(
        speed=0,
        render=False,
        player1='ai',
        player2='cpu',
        char1='falcon',
        char2='falcon',
        cpu2=2,
        stage='battlefield',
    )
    if args.selfplay:
        options['player2'] = 'ai'

    env = Env(args, args.device, options=options, dummy=args.dummy)
    obs = env.reset()
    print(obs.squeeze().numpy())
    while True:
        if args.selfplay:
            action = [[env.action_space.sample(), env.action_space.sample()] for _ in range(args.n_actors_per_worker)]
        else:
            action = [env.action_space.sample() for _ in range(args.n_actors_per_worker)]
        obs, reward, done = env.step(action)
        print(obs.squeeze().numpy(), reward, done)
        if done[0]:
            break

