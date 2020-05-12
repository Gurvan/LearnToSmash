import os
import sys
import argparse
from os.path import join

import numpy as np
import torch

from env import SelfPlaySimpleEnv as Env
from models import Policy

options = dict(
    iso="/home/gurvan/Melee/ISOs/SSBM, 20XXHP 4.07.UCF.iso",
    speed=1,
    render=True,
    player1='ai',
    player2='ai',
    char1='falcon',
    char2='falcon',
    cpu2=3,
    stage='battlefield',
)

def main():
    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--act-every', type=int, default=3, help='act every N frames (default: 3)')
    parser.add_argument('--duration', type=int, default=1800, help='duration in frames (default: 1800 = 30 seconds)')
    parser.add_argument('--dummy', action='store_true', default=False, help='Dummy env')
    parser.add_argument('--human', action='store_true', default=False, help='P2 is human')
    parser.add_argument('--load-model', type=str, default='results/latest/model.pth')
    args = parser.parse_args()

    args.max_rollout_length = args.duration


    if args.human:
        options["player2"] = "human"
    
    env = Env(args, 'cpu', options=options, dummy=args.dummy)
    observation_dim = env.observation_dim
    action_dim = env.action_dim

    print(action_dim)

    actor = Policy(action_dim)
    actor_state_dict = torch.load(args.load_model , map_location='cpu')
    actor.load_state_dict(actor_state_dict, strict=False)

    
    obs = env.reset()
    t = 0
    r = 0
    # actions = [0,0]
    actions = 0
    with torch.no_grad():
        while True:
            try:
                action = actor.act(obs)
                action = action[0].cpu().numpy()
                obs, reward, done = env.step(action)
                if done:
                    env.close()
                    break
 
                if t >= args.duration:
                    env.close()
                    break

            except Exception as e:
                print(e)
                env.close()
                raise
                break
            t += 1

if __name__ == '__main__':
    main()
