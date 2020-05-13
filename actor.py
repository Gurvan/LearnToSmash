import numpy as np
import torch
from env import Env
from models import Policy

class Actor(object):
    def __init__(self, args, rollout_queue, shared_state_dict, actor_name=None, rank=0):
        self.args = args
        self.rollout_queue = rollout_queue
        self.actor_name = actor_name
        self.rank = rank
        # self.device = 'cpu'  # args.device
        self.device = args.device
        self.env = None
        self.policy = None
        self.memory = None
        self.shared_state_dict = shared_state_dict

    def initialize(self):
        print('Build Environment for {}'.format(self.actor_name))
        if self.env is None:
            self.env = Env(self.args, self.device, options=self.args.options, dummy=self.args.dummy, rank=self.rank)
        self.policy = Policy(self.env.action_dim).to(self.device)
        self.memory = Memory()

    def performing(self):
        torch.manual_seed(self.args.seed + self.rank)
        self.initialize()
        obs = self.env.reset()
        with torch.no_grad():
            while True:
                self.policy.load_state_dict(self.shared_state_dict.state_dict())
                try:
                    self.policy.reset_rnn()
                    obs = self.env.reset()
                except:
                    obs = obs[-1:]
                    print(obs.shape)
                self.memory.observations.append(obs)
                # print(obs.shape)
                for step in range(self.args.num_steps):
                    action, action_log_prob = self.policy(obs)
                    self.memory.actions.append(action)
                    self.memory.actions_log_probs.append(action_log_prob)

                    send_action = action[-1].cpu().numpy()
                    obs, reward, done = self.env.step(send_action)
                    self.memory.observations.append(obs)
                    self.memory.rewards.append(torch.from_numpy(reward.astype(np.float32)))

                    # print("actor", obs.shape, action.shape, action_log_prob.shape, reward.shape)
                action, action_log_prob = self.policy(obs)
                self.memory.actions.append(action[0:-1])
                self.memory.actions_log_probs.append(action_log_prob[0:-1])

                # print(self.rollout_queue.qsize())
                self.rollout_queue.put(self.memory.get_batch())


class Memory:
    def __init__(self):
        self.clear_memory()
    
    def clear_memory(self):
        self.observations = []
        self.actions = []
        self.actions_log_probs = []
        self.rewards = []

    def push(self, observation, action, action_log_prob, reward):
        self.observations.append(observation)
        self.actions.append(action)
        self.actions_log_probs.append(action_log_prob)
        self.rewards.append(torch.from_numpy(reward.astype(np.float32)))

    def get_batch(self):
        observations = torch.cat(self.observations, dim=0).to('cpu')
        actions = torch.cat(self.actions, dim=0).to('cpu')
        actions_log_probs = torch.cat(self.actions_log_probs, dim=0).to('cpu')
        rewards = torch.cat(self.rewards, dim=0).to('cpu')
        self.clear_memory()
        return (observations, actions, actions_log_probs, rewards)

    def __len__(self):
        return len(self.actions)

