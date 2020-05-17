import time
import pathlib
import argparse
from datetime import datetime

import torch
import torch.multiprocessing as mp

from models import Policy, partial_load
from actor import Actor
from learner import Learner
from experience_buffer import ExperienceBuffer

from env import Env

filedir = pathlib.Path(__file__).parent.resolve()


options = dict(
    speed=0,
    render=False,
    player1='ai',
    player2='ai',
    char1='falcon',
    char2='falcon',
    stage='battlefield',
)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment_id', type=str, default=str(datetime.now()), help='Experiment ID')
    parser.add_argument('--lr', type=float, default=3e-5, help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-steps', type=int, default=300, help='Number of frames played per actor per rollout')
    parser.add_argument('--n-actors-per-worker', type=int, default=4, help='Number of actors per worker')
    parser.add_argument('--n-workers', type=int, default=2, help='Number worker (1 pool of actors sharing GPU computations)')
    parser.add_argument('--act-every', type=int, default=3, help='Send action every N frames')
    parser.add_argument('--seed', type=int, default=2020, help='Random seed')
    parser.add_argument('--max-intensity', type=int, default=3, help="Maximum number of times an experience can be used for learning")
    parser.add_argument('--c-hat', type=float, default=1.0, help="IMPALA hyper-parameters")
    parser.add_argument('--rho-hat', type=float, default=1.0, help="IMPALA hyper-parameters")
    parser.add_argument('--gamma', type=float, default=0.997, help="IMPALA hyper-parameters")
    parser.add_argument('--entropy-coef', type=float, default=0.0025, help="IMPALA hyper-parameters")
    parser.add_argument('--value-loss-coef', type=float, default=0.5, help="IMPALA hyper-parameters")
    parser.add_argument('--max-grad-norm', type=float, default=0.5)
    parser.add_argument('--save-interval', type=int, default=10, help="Save model every N epochs")
    parser.add_argument('--dummy', action="store_true", default=False, help="Used for testing implementation with dummy environments")
    parser.add_argument('--no-cuda', action="store_true", default=False, help="Disables CUDA")
    parser.add_argument('--load-model', type=str, default=None, help="Load pretrained agent")
    parser.add_argument('--load-teacher', type=str, default=None, help="Load teacher agent")
    parser.add_argument('--reset-policy', action="store_true", default=False, help="Reset the policy part of a pretained agent. Useful for training a new character using a model pretrained on another character.")
    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    args.device = torch.device('cuda' if use_cuda else 'cpu')

    args.gamma = args.gamma**args.act_every
    args.num_steps = int(args.num_steps / args.act_every)
    args.options = options

    try:
        mp.set_start_method('forkserver', force=True)
        # print("forkserver init")
    except RuntimeError:
        pass

    processes = []
    
    # Buffer used for sharing rollouts from actors to learner
    experience_buffer = ExperienceBuffer(args.batch_size)
    p = mp.Process(target=experience_buffer.listening)
    p.start()
    processes.append(p)

    # Getting action dim and observation dim from Env
    env = Env(args, device='cpu', options=options, dummy=True)
    observation_dim = env.observation_dim
    args.action_dim = env.action_dim
    env.close()
    print('Observation Space: {} / Action Dim: {}'.format(observation_dim, args.action_dim))

    # Initializing shared memory used between workers and learner that contains the actor parameters
    shared_state_dict = Policy(args.action_dim)
    if args.load_model is not None:
        partial_load(shared_state_dict, args.load_model)
        if args.reset_policy:
            shared_state_dict.policy.weight.data.zero_()
            shared_state_dict.policy.bias.data.zero_()
    shared_state_dict = shared_state_dict.share_memory()

    # Creating learner
    learner = Learner(args, experience_buffer.queue_batch, shared_state_dict)

    # Creating actors
    actors = []
    for i in range(args.n_workers):
        actor_name = 'actor_' + str(i)
        actor = Actor(args, experience_buffer.queue_trace, shared_state_dict, actor_name, rank=i)
        actors.append(actor)

    # Creating results dir
    if not args.dummy:
        args.result_dir = filedir / 'results' / args.experiment_id
        args.result_dir.mkdir(parents=True, exist_ok=True)
        (filedir / 'results' / 'latest').mkdir(parents=True, exist_ok=True)
        try:
            (filedir / 'results' / 'latest' / 'reward.txt').unlink()
        except:
            pass
        print(args, file=open(args.result_dir / "info.txt", mode='w'))


    try:
        print('Run processes')
        for rank, actor in enumerate(actors):
            p = mp.Process(target=actor.performing)
            p.start()
            processes.append(p)

        try:
            learner.learning()
        except Exception as e:
            print(e)
            raise e

        for p in processes:
            p.join()
    except Exception as e:
        print(e)
        for p in processes:
            p.terminate()
        for p in processes:
            p.join()

    finally:
        if not args.dummy:
            torch.save(shared_state_dict.state_dict(), args.result_dir / "model.pth")
            torch.save(shared_state_dict.state_dict(), args.result_dir / '..' / 'latest' / "model.pth")
            import subprocess
            time.sleep(0.1)
            subprocess.Popen(["pkill", "dolphin-emu"])
            time.sleep(0.1)
            subprocess.Popen(["pkill", "dolphin-emu"])
