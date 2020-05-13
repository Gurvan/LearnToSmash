import torch

# Give bonus reward when the characters are close. It help at the beginning of training.
def proximity_bonus(observation, act_every, alpha=0.1, agent_x_index=1, agent_y_index=2, opponent_x_index=9, opponent_y_index=10):
    index = torch.tensor([agent_x_index, agent_y_index, opponent_x_index, opponent_y_index]).long()
    agent_x, agent_y, opponent_x, opponent_y = observation.index_select(dim=-1, index=index).split(1, dim=-1)
    d = ((agent_x - opponent_x).pow(2) + (agent_y - opponent_y).pow(2)).sqrt().squeeze(-1)
    bonus = 1/60 * alpha * 1/(1 + alpha * d)

    n = int(observation.shape[0] / act_every)
    bonus = bonus[1:].view(n, act_every, -1).sum(1)
    return bonus