import torch
from torch.multiprocessing import Queue
import traceback

class ExperienceBuffer():
    def __init__(self, batch_size):
        self.observations = []
        self.actions = []
        self.actions_log_probs = []
        self.rewards = []

        self.queue_trace = Queue(maxsize=30)
        self.queue_batch = Queue(maxsize=3)
        self.batch_size = batch_size
        self.num_traces = 0

    def listening(self):
        try:
            while True:
                trace = self.queue_trace.get(block=True)
                # print("trace", trace[0].shape, trace[1].shape, trace[2].shape, trace[3].shape)

                self.observations.append(trace[0])
                self.actions.append(trace[1])
                self.actions_log_probs.append(trace[2])
                self.rewards.append(trace[3])

                self.num_traces += trace[0].shape[1]
                # print(self.num_traces)

                if self.num_traces >= self.batch_size:
                    self.num_traces -= self.batch_size
                    observations_batch, observations_remain = torch.cat(self.observations, dim=1).split([self.batch_size, self.num_traces], dim=1)
                    actions_batch, actions_remain = torch.cat(self.actions, dim=1).split([self.batch_size, self.num_traces], dim=1)
                    actions_log_probs_batch, actions_log_probs_remain = torch.cat(self.actions_log_probs, dim=1).split([self.batch_size, self.num_traces], dim=1)
                    rewards_batch, rewards_remain = torch.cat(self.rewards, dim=1).split([self.batch_size, self.num_traces], dim=1)

                    self.observations = [observations_remain]
                    self.actions = [actions_remain]
                    self.actions_log_probs = [actions_log_probs_remain]
                    self.rewards = [rewards_remain]

                    # print("batch", observations_batch.shape, actions_batch.shape, actions_log_probs_batch.shape, rewards_batch.shape)
                    # print("remain", observations_remain.shape, actions_remain.shape, actions_log_probs_remain.shape, rewards_remain.shape)

                    self.produce_batch(observations_batch, actions_batch, actions_log_probs_batch, rewards_batch)
        except Exception as e:
            tb = traceback.format_exc()
            print(e)
            print(tb)
            self.observations = []
            self.actions = []
            self.actions_log_probs = []
            self.rewards = []

    def produce_batch(self, observations, actions, actions_log_probs, rewards):
        self.queue_batch.put((
            observations,
            actions,
            actions_log_probs,
            rewards,
            ))
