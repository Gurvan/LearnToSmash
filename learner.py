import os
import time
import queue
from copy import deepcopy
from collections import deque

import torch
from torch.optim import Adam

from models import Policy, partial_load
from utils import proximity_bonus


class Learner(object):
    def __init__(self, args, queue_batch, shared_state_dict):
        self.args = args
        self.device = args.device
        self.queue_batch = queue_batch
        self.policy = None
        self.shared_state_dict = shared_state_dict
        self.initialize()

    def initialize(self):
        self.policy = Policy(self.args.action_dim).to(self.device)
        if self.args.load_model is not None:
            partial_load(self.policy, self.args.load_model)
            if self.args.reset_policy:
                self.policy.policy.weight.data.zero_()
                self.policy.policy.bias.data.zero_()
        self.optimizer = Adam(self.policy.parameters(), lr=self.args.lr)
        self.policy.train()
        self.update_state_dict()

    def update_state_dict(self):
        self.shared_state_dict.load_state_dict(self.policy.state_dict())

    def learning(self):
        torch.manual_seed(self.args.seed)
        # self.initialize()
        c_hat = self.args.c_hat
        rho_hat = self.args.rho_hat

        observations, actions, mu_log_probs, rewards = self.queue_batch.get(block=True)
        # print(observations.shape, actions.shape, mu_log_probs.shape, rewards.shape)
        i = 0
        batch_iter = 0
        t = time.perf_counter()
        while True:
            try:
                # Retrain on previous batch if the next one is not ready yet
                if batch_iter < self.args.max_intensity:
                    observations, actions, mu_log_probs, rewards = self.queue_batch.get(block=False)
                    batch_iter = 0
                else:
                    batch_iter = 0
                    observations, actions, mu_log_probs, rewards = self.queue_batch.get(block=True)
                
            except queue.Empty:
                pass
            observations, actions, mu_log_probs, rewards = observations.to(self.device), actions.to(self.device), mu_log_probs.to(self.device), rewards.to(self.device)
            # print(observations.shape, actions.shape, mu_log_probs.shape, rewards.shape)
            batch_iter += 1
            self.optimizer.zero_grad()

            values, pi_log_probs, entropy = self.policy.evaluate_actions(observations, actions)
            # print(values.shape, pi_log_probs.shape, mu_log_probs.shape, entropy.shape)

            is_rate = (pi_log_probs.detach() - mu_log_probs).exp()
            c = is_rate.clamp_max(c_hat)
            rho = is_rate.clamp_max(rho_hat)
            # print(is_rate.shape, c.shape, rho.shape)

            # Optimistic reward
            rewards_ = rewards.exp() - 1.0
            # print(rewards.cpu().numpy())
            # Reward bonus: proximity
            n_steps = (observations.shape[0] - 1) * observations.shape[1] * i  # total number of frames played
            bonus = proximity_bonus(observations, self.args.act_every, alpha=max(0, 0.15 * (1e9 - n_steps) / 1e9))
            rewards_with_bonus = rewards_ + bonus


            ###### V-trace / IMPALA
            # https://arxiv.org/abs/1802.01561
            v, advantages = compute_vtrace(values, rewards_with_bonus, c, rho, self.args.gamma)
            # print(v.shape, advantages.shape)

            value_loss = 0.5 * (v - values).pow(2).sum()
            policy_loss = -(pi_log_probs * advantages).sum()
            entropy_loss = -entropy.sum()
            ######

            # print(value_loss.item(), policy_loss.item(), entropy_loss.item())
            loss = policy_loss + self.args.value_loss_coef * value_loss + self.args.entropy_coef * entropy_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.args.max_grad_norm)
            self.optimizer.step()

            self.update_state_dict()

            if (i % self.args.save_interval == 0) and not self.args.dummy:
                torch.save(self.shared_state_dict.state_dict(), self.args.result_dir / "model.pth")
                torch.save(self.shared_state_dict.state_dict(), self.args.result_dir / '..' / 'latest' / "model.pth")
            
            print(is_rate.mean().item())
            if batch_iter == 1:
                t_ = time.perf_counter()
                i += 1
                n_steps = (observations.shape[0] - 1) * observations.shape[1] * i
                print("Iteration: {} / Time: {:.3f}s / Total frames {} / Value loss {:.3f} / Policy loss {:.3f} / Entropy loss {:.5f} / Total loss {:.3f} / Reward: {:.3f}".format(
                    i,
                    t_ - t,
                    n_steps,
                    value_loss.item() / rho.shape[0],
                    policy_loss.item() / rho.shape[0],
                    entropy_loss.item() / rho.shape[0],
                    loss.item() / rho.shape[0],
                    rewards_.mean().item() * 3600 / self.args.act_every,
                ))
                t = t_

                
                if not self.args.dummy:
                    with open(self.args.result_dir / 'reward.txt', "a") as f:
                        print(n_steps, rewards_.mean().item() * 3600 / self.args.act_every, bonus.mean().item() * 3600 / self.args.act_every, file=f, sep=",")
                    with open(self.args.result_dir / '..' / 'latest' / 'reward.txt', "a") as f:
                        print(n_steps, rewards_.mean().item() * 3600 / self.args.act_every, bonus.mean().item() * 3600 / self.args.act_every, file=f, sep=",")

            # Prevent from replaying the batch if the experience is too much off-policy
            if is_rate.log2().mean().abs() > 0.015:
                batch_iter = self.args.max_intensity
            time.sleep(0.1)


# V-trace
# dsV = ps ( rs + y V(xs1) - V(xs))
# vs = V(xs) + dsV + y cs(vs1 - V(xs1))
# https://arxiv.org/abs/1802.01561
def compute_vtrace(values, rewards, c, rho, discounts):
    # print("values, rewards, c, rho")
    # print(values.shape, rewards.shape, c.shape, rho.shape)
    with torch.no_grad():
        v = [values[-1]]
        for s in reversed(range(values.shape[0]-1)):
            dV = rho[s] * (rewards[s] + discounts * values[s+1] - values[s])
            v.append(values[s] + dV + discounts * c[s] * (v[-1] - values[s+1]))

        v = torch.stack(tuple(reversed(v)), dim=0)
        advantages = rho * (rewards + discounts * v[1:] - values[:-1])
        return v.detach(), advantages.detach()
