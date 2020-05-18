import numpy as np
import torch
from env import Env
from models import Policy, partial_load

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
        self.opponent_is_self = True
        self.shared_state_dict = shared_state_dict

    def initialize(self):
        print('Build Environment for {}'.format(self.actor_name))
        if self.env is None:
            self.env = Env(self.args, self.device, options=self.args.options, dummy=self.args.dummy, rank=self.rank)
        self.policy = Policy(self.env.action_dim).to(self.device)
        self.opponent = Policy(self.env.action_dim).to(self.device)
        self.opponent.eval()
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
                if not self.opponent_is_self:
                    obs, opponent_obs = obs.chunk(2, dim=1)
                self.memory.observations.append(obs)
                # print(obs.shape)
                for step in range(self.args.num_steps):
                    action, action_log_prob = self.policy(obs)
                    if not self.opponent_is_self:
                        with torch.no_grad():
                            opponent_action = self.opponent.act(opponent_obs)
                        actions = torch.cat([action, opponent_action], dim=1)
                        send_action = actions[-1].cpu().numpy()
                    else:
                        send_action = action[-1].cpu().numpy()

                    obs, reward, done = self.env.step(send_action)
                    rewards = torch.from_numpy(reward.astype(np.float32))
                    if not self.opponent_is_self:
                        obs, opponent_obs = obs.chunk(2, dim=1)
                        rewards, _ = rewards.chunk(2, dim=1)

                    self.memory.actions.append(action)
                    self.memory.actions_log_probs.append(action_log_prob)
                    self.memory.observations.append(obs)
                    self.memory.rewards.append(rewards)

                    # print("actor", obs.shape, action.shape, action_log_prob.shape, reward.shape)
                action, action_log_prob = self.policy(obs)
                self.memory.actions.append(action[0:-1])
                self.memory.actions_log_probs.append(action_log_prob[0:-1])

                # print(self.rollout_queue.qsize())
                self.rollout_queue.put(self.memory.get_batch())
                self.opponent_is_self = not reload_opponent(self.opponent, os.path.join(self.args.opponent_dir))


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


import os
import numpy as np
def reload_opponent(opponent, path):
    opponents = []
    try:
        opponents = sorted(os.listdir(str(path)))
        n = len(opponents)
        if n > 1:
            if np.random.random() > 0.5:
                return False
                # print("Reloading latest opponent")
                # partial_load(opponent, path / opponents[-1])
                # return True
            else:
                # o = np.random.choice(opponents[:-1])
                o = np.random.choice(opponents)
                # print("Reloading", o)
                partial_load(opponent, path / np.random.choice(opponents), debug=False)
                return True
    except:
        return False
    return False



