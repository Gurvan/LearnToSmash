import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal, Independent
import numpy as np


class ObservationEmbedding(nn.Module):
    def __init__(self, observations_dim=16):
        super(ObservationEmbedding, self).__init__()
        self.actionstate_embedding_size = 32
        self.actionstate_embedding = nn.Embedding(400, self.actionstate_embedding_size)
        self.actionstate_embedding.weight.data.zero_()
        self.output_dim = 2 * (self.actionstate_embedding_size + int(observations_dim/2) - 1)

    def embed(self, x):
        a, s = x.split([1, x.shape[-1] - 1], dim=-1)
        actionstate = self.actionstate_embedding(a.long().clamp(0, 399)).squeeze(-2)
        return torch.cat([actionstate, s], dim=-1)

    def forward(self, x, return_info=False):
        agent, opponent = x.chunk(2, dim=-1)
        agent = self.embed(agent)
        opponent = self.embed(opponent)
        if return_info:
            # returns embeddding + [facing, in_air] for the agent
            return torch.cat([agent, opponent], dim=-1), agent.T[[-3, -1]].T
        else:
            return torch.cat([agent, opponent], dim=-1)


# action_state, x, y, percent, shield_size, facing, jump_used, in_air
class ObservationDecoder(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.ssbm_actionstate_dim = 400
        self.ssbm_logits = nn.Linear(embedding_dim, 2 * (self.ssbm_actionstate_dim + 7))
        # self._ssbm_actionstate = nn.Linear(hidden, self._ssbm_actionstate_dim)
        # self._ssbm_physic = nn.Linear(hidden, 2)
        # self._ssbm_percent = nn.Linear(hidden, 1)
        # self._ssbm_shield = nn.Linear(hidden, 1)
        # self._ssbm_facing = nn.Linear(hidden, 1)
        # self._ssbm_jump_used = nn.Linear(hidden, 1)
        # self._ssbm_in_air = nn.Linear(hidden, 1)

    def forward(self, x):
        agent, opponent = self.ssbm_logits(x).chunk(2, dim=-1)
        actionstate, physic, percent, shield, facing, jump_used, in_air = agent.split([self.ssbm_actionstate_dim, 2, 1, 1, 1, 1, 1], dim=-1)
        percent = F.softplus(percent)
        shield = F.softplus(shield)
        agent_dict = dict(
            actionstate=actionstate,
            physic=physic,
            percent=percent,
            shield=shield,
            facing=facing,
            jump_used=jump_used,
            in_air=in_air,
        )
        actionstate, physic, percent, shield, facing, jump_used, in_air = opponent.split([self.ssbm_actionstate_dim, 2, 1, 1, 1, 1, 1], dim=-1)
        percent = F.softplus(percent)
        shield = F.softplus(shield)
        opponent_dict = dict(
            actionstate=actionstate,
            physic=physic,
            percent=percent,
            shield=shield,
            facing=facing,
            jump_used=jump_used,
            in_air=in_air,
        )
        return agent_dict, opponent_dict

    def split_observation(self, observation):
        agent, opponent = observation.chunk(2, dim=-1)
        actionstate, physic, percent, shield, facing, jump_used, in_air = agent.split([1,2,1,1,1,1,1], dim=-1)
        agent_dict = dict(
            actionstate=actionstate,
            physic=physic,
            percent=percent,
            shield=shield,
            facing=facing,
            jump_used=jump_used,
            in_air=in_air,
        )
        actionstate, physic, percent, shield, facing, jump_used, in_air = opponent.split([1,2,1,1,1,1,1], dim=-1)
        opponent_dict = dict(
            actionstate=actionstate,
            physic=physic,
            percent=percent,
            shield=shield,
            facing=facing,
            jump_used=jump_used,
            in_air=in_air,
        )
        return agent_dict, opponent_dict

    def compute_loss(self, agent_pred, opponent_pred, observation):
        agent_target, opponent_target = self.split_observation(observation)

        loss = 0
        loss += F.cross_entropy(agent_pred['actionstate'].view(-1, self.ssbm_actionstate_dim), agent_target['actionstate'].long().view(-1)).sum()
        loss += F.mse_loss(agent_pred['physic'], agent_target['physic']).sum()
        loss += F.mse_loss(agent_pred['percent'], agent_target['percent']).sum()
        loss += F.mse_loss(agent_pred['shield'], agent_target['shield']).sum()
        loss += F.binary_cross_entropy_with_logits(agent_pred['facing'], agent_target['facing']).sum()
        loss += F.binary_cross_entropy_with_logits(agent_pred['jump_used'], agent_target['jump_used']).sum()
        loss += F.binary_cross_entropy_with_logits(agent_pred['in_air'], agent_target['in_air']).sum()
        
        loss += F.cross_entropy(opponent_pred['actionstate'].view(-1, self.ssbm_actionstate_dim), opponent_target['actionstate'].long().view(-1)).sum()
        loss += F.mse_loss(opponent_pred['physic'], opponent_target['physic']).sum()
        loss += F.mse_loss(opponent_pred['percent'], opponent_target['percent']).sum()
        loss += F.mse_loss(opponent_pred['shield'], opponent_target['shield']).sum()
        loss += F.binary_cross_entropy_with_logits(opponent_pred['facing'], opponent_target['facing']).sum()
        loss += F.binary_cross_entropy_with_logits(opponent_pred['jump_used'], opponent_target['jump_used']).sum()
        loss += F.binary_cross_entropy_with_logits(opponent_pred['in_air'], opponent_target['in_air']).sum()
        return loss


class Policy(nn.Module):
    def __init__(self, action_dim, hidden_dim=256, rnn_hidden_dim=512, policy_hidden_dim=128):
        super().__init__()
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.rnn_hidden_dim = rnn_hidden_dim
        self.policy_hidden_dim = policy_hidden_dim
        self.action_embedding_dim = 10

        self.observation_embedding = ObservationEmbedding()
        self.core = nn.Sequential(
            nn.Linear(self.observation_embedding.output_dim, hidden_dim),
            nn.ReLU(True),
            ResidualBlock(hidden_dim, 128),
        )
        self.action_embedding = nn.Embedding(action_dim, self.action_embedding_dim)
        self.rnn = nn.GRU(self.action_embedding_dim + hidden_dim, rnn_hidden_dim)
        self.post_rnn_film = FiLM(rnn_hidden_dim, 2)

        self.policy = nn.Sequential(
            nn.Linear(rnn_hidden_dim, policy_hidden_dim),
            nn.ELU(),
            nn.Linear(policy_hidden_dim, action_dim),
        )
        self.value = nn.Sequential(
            nn.Linear(rnn_hidden_dim, policy_hidden_dim),
            nn.ELU(),
            nn.Linear(policy_hidden_dim, 1),
        )

        self.simcore = SimCore(rnn_hidden_dim, self.action_embedding_dim)

        self.rnn_hidden = None
        self.prev_action = None

    def _forward(self, observation, rnn_hidden):
        obs, c = self.observation_embedding(observation, return_info=True)
        x = self.core(obs)
        if self.prev_action is None:
            action = torch.zeros(1, x.shape[1], self.action_embedding_dim).to(x.device)
        else:
            action = self.action_embedding(self.prev_action.to(x.device))
        action = action.repeat((int(x.shape[0] / action.shape[0]), 1, 1))
        y = torch.cat([action, x], dim=-1)

        h, rnn_hidden = self.rnn(y, rnn_hidden)
        h = self.post_rnn_film(h, c)
        logits = self.policy(h[-1:])
        dist = Categorical(logits=logits)
        return dist, rnn_hidden

    def forward(self, observation):
        dist, self.rnn_hidden = self._forward(observation, self.rnn_hidden)
        action = dist.sample()
        self.prev_action = action
        action_log_prob = dist.log_prob(action)
        return action.detach(), action_log_prob.detach()

    def act(self, observation):
        dist, self.rnn_hidden = self._forward(observation, self.rnn_hidden)
        action = dist.sample()
        self.prev_action = action
        return action.detach()

    def evaluate_actions(self, observations, actions, rnn_hidden=None):
        n = int((observations.shape[0] - 1) / (actions.shape[0] - 1))
        obs, c = self.observation_embedding(observations, return_info=True)
        x = self.core(obs)
        prev_actions = self.get_prev_actions(actions, n).squeeze(-1)
        y = torch.cat([prev_actions, x], dim=-1)

        h, _ = self.rnn(y, rnn_hidden)
        beliefs = h[:-1]
        h = h[list(range(0, observations.shape[0], n))]
        c = c[list(range(0, observations.shape[0], n))]
        h = self.post_rnn_film(h, c)
        values = self.value(h).squeeze(-1)
        logits = self.policy(h[:-1])
        dist = Categorical(logits=logits)
        actions_log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return values, actions_log_probs, entropy, beliefs, prev_actions[1:], dist.logits

    def get_prev_actions(self, actions, n):
        first_prev_action = torch.zeros((1, actions.shape[1], self.action_embedding_dim), device=actions.device)
        next_prev_actions = self.action_embedding(actions.repeat_interleave(n, dim=0))
        prev_actions = torch.cat([first_prev_action, next_prev_actions], dim=0)
        return prev_actions

    def reset_rnn(self):
        self.rnn_hidden = None
        self.prev_action = None


class Teacher(nn.Module):
    def __init__(self, action_dim, hidden_dim=64, rnn_hidden_dim=256, rnn_layers=1):
        super().__init__()
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.rnn_hidden_dim = rnn_hidden_dim
        self.rnn_layers = rnn_layers

        self.observation_embedding = ObservationEmbedding()
        self.core = nn.Sequential(
            nn.Linear(self.observation_embedding.output_dim, hidden_dim),
            nn.ReLU(True),
            ResidualBlock(hidden_dim, 32),
        )
        self.rnn = nn.GRU(action_dim + hidden_dim, rnn_hidden_dim, rnn_layers)
        self.post_rnn_film = FiLM(rnn_hidden_dim, 2)

        self.policy = nn.Linear(rnn_hidden_dim, action_dim)
        self.value = nn.Linear(rnn_hidden_dim, 1)

        self.rnn_hidden = None
        self.prev_action = None

    def compute_actions_probs(self, observations, actions, rnn_hidden=None):
        n = int((observations.shape[0] - 1) / (actions.shape[0] - 1))
        obs, c = self.observation_embedding(observations, return_info=True)
        x = self.core(obs)
        prev_actions = self.get_prev_actions(actions, n)
        y = torch.cat([prev_actions, x], dim=-1)

        h, _ = self.rnn(y, rnn_hidden)
        h = h[list(range(0, observations.shape[0], n))]
        c = c[list(range(0, observations.shape[0], n))]
        h = self.post_rnn_film(h, c)
        logits = self.policy(h[:-1])
        dist = Categorical(logits=logits)
        return dist.probs.detach()

    def get_prev_actions(self, actions, n):
        first_prev_action = torch.zeros((1, actions.shape[1], self.action_dim), device=actions.device)
        next_prev_actions = torch.nn.functional.one_hot(actions.repeat_interleave(n, dim=0), self.action_dim).float()
        prev_actions = torch.cat([first_prev_action, next_prev_actions], dim=0)
        return prev_actions

    def reset_rnn(self):
        self.rnn_hidden = None
        self.prev_action = None


class SimCore(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.Lo = 30
        self.Nt = 6
        self.Ng = 4

        self.rnn = nn.GRU(action_dim, state_dim)
        self.predecoder = ResidualBlock(state_dim, 64)
        self.decoder = ObservationDecoder(embedding_dim=self.state_dim)

    def compute_loss(self, observations, actions, beliefs):
        # print("obs, actions, beliefs")
        # print(observations.shape, actions.shape, beliefs.shape)
        Lu = observations.shape[0] - 1 - self.Lo
        loss = 0
        for _ in range(self.Nt):
            i = np.random.randint(Lu)
            K = np.random.choice(self.Lo, self.Ng, replace=False)
            state = beliefs[i:i+1]
            states = []
            # Unroll
            # print(max(K))
            for j in range(max(K)+1):
                action = actions[i + j:i + j + 1]
                # print(action, state)
                # print(action.shape, len(state), state.shape)
                _, state = self.rnn(action, state)
                if j in K:
                    states.append(state)
            states = torch.cat(states, dim=0)
            # print(states.shape, observations[K + i + 1].shape)
            agent_pred, opponent_pred = self.decoder(self.predecoder(states))
            loss += self.decoder.compute_loss(agent_pred, opponent_pred, observations[K + i + 1])
            # print("loss", loss)

        # print("loss", loss.shape)
        return loss


class ResidualBlock(nn.Module):
    """
    https://arxiv.org/abs/1806.10909
    """
    def __init__(self, data_dim, hidden_dim):
        super().__init__()
        self.data_dim = data_dim
        self.hidden_dim = hidden_dim
        self.mlp = nn.Sequential(
            nn.Linear(data_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, data_dim),
            nn.ReLU(True),
        )

    def forward(self, x):
        return x + self.mlp(x)


class FiLM(nn.Module):
    """
    https://distill.pub/2018/feature-wise-transformations/
    """
    def __init__(self, input_dim, conditioning_dim):
        super().__init__()
        self.input_dim = input_dim
        self.conditioning_dim = conditioning_dim
        self.film = nn.Linear(conditioning_dim, 2 * input_dim)

    def forward(self, x, c):
        gamma, beta = self.film(c).chunk(2, dim=-1)
        return gamma * x + beta


def partial_load(model, path, debug=True):
    old_dict = torch.load(path, map_location='cpu')
    model_dict = model.state_dict()
    if debug:
        print("Non-matching keys: ", {k for k, _ in old_dict.items() if not (k in model_dict and model_dict[k].shape == old_dict[k].shape)})
    old_dict = {k: v for k, v in old_dict.items() if k in model_dict and model_dict[k].shape == old_dict[k].shape}
    model_dict.update(old_dict)
    model.load_state_dict(model_dict)