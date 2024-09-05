import argparse
import logging
import os
from copy import deepcopy
from datetime import datetime
from functools import partial

import gymnasium
import numpy as np
import torch
from pettingzoo.classic import tictactoe_v3
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.env.pettingzoo_env import PettingZooEnv
from tianshou.highlevel.logger import LoggerFactoryDefault
from tianshou.policy import BasePolicy, DQNPolicy, MultiAgentPolicyManager, RandomPolicy
from tianshou.trainer import OffpolicyTrainer
from tianshou.utils.net.common import Net


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--task", type=str, default="TicTacToe-v3")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--learned-go-first", action="store_true")
    parser.add_argument(
        "--win-rate",
        type=float,
        default=0.6,
        help="Expected win rate. Note: optimal policy can achieve 0.7",
    )
    parser.add_argument(
        "--hidden-sizes",
        type=int,
        nargs="*",
        default=[128, 128, 128, 128],
        help="Hidden layer sizes of DQN.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.9,
        help="Discount factor. Note: a small gamma favors earlier win.",
    )
    parser.add_argument(
        "--td-step", type=int, default=3, help="N-step in multi-step TD target."
    )
    parser.add_argument("--target-update-freq", type=int, default=320)
    parser.add_argument(
        "--buffer-size", type=int, default=20000, help="Size of replay buffer."
    )
    parser.add_argument(
        "--epsilon-train",
        type=float,
        default=0.1,
        help="Value for epsilon-greedy in training.",
    )
    parser.add_argument(
        "--epsilon-test",
        type=float,
        default=0.05,
        help="Value for epsilon-greedy in testing.",
    )
    parser.add_argument(
        "--load-opponent",
        action="store_true",
        help="If load the opponent policy from file.",
    )
    parser.add_argument(
        "--train-env-num",
        type=int,
        default=10,
        help="Number of training environments the agent interacts with in parallel.",
    )
    parser.add_argument(
        "--test-env-num",
        type=int,
        default=10,
        help="Number of testing environments the agent interacts with in parallel.",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=50,
        help="Number of total epochs in training process.",
    )
    parser.add_argument("--step-per-epoch", type=int, default=1000)
    parser.add_argument("--step-per-collect", type=int, default=10)
    parser.add_argument("--update-per-step", type=float, default=0.1)
    parser.add_argument("--episode-per-test", type=int, default=100)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--logger", type=str, default="tensorboard", choices=["tensorboard", "wandb"]
    )
    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument("--wandb-project", type=str, default="tianshou-tictactoe")
    parser.add_argument("--resume-id", type=str, default=None)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--watch-episode-num", type=int, default=1)
    parser.add_argument("--render-fps", type=int, default=60)

    return parser.parse_known_args()[0]


def configure_log_path(args: argparse.Namespace) -> argparse.Namespace:
    args.now = datetime.now().strftime("%Y%m%d-%H%M%S")
    args.log_path = os.path.join(args.logdir, args.now)
    if args.logger == "wandb":
        os.makedirs(os.path.join(args.log_path, "wandb"), mode=755, exist_ok=True)
    return args


def get_env(render_mode: str | None = None) -> PettingZooEnv:
    return PettingZooEnv(tictactoe_v3.env(render_mode=render_mode))


def add_env_info(args: argparse.Namespace) -> argparse.Namespace:
    env = get_env()
    args.state_space = (
        env.observation_space["observation"]
        if isinstance(env.observation_space, gymnasium.spaces.Dict)
        else env.observation_space
    )  # `env.observation_space` may has `action_mask` key
    args.state_shape = args.state_space.shape or int(args.state_space.n)
    args.action_space = env.action_space
    args.action_shape = args.action_space.shape or int(args.action_space.n)
    return args


def get_policy(
    args: argparse.Namespace,
    policy_learn: BasePolicy | None = None,
    policy_opponent: BasePolicy | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    use_best: bool = False,
):
    if policy_learn is None:
        net = Net(
            state_shape=args.state_shape,
            action_shape=args.action_shape,
            hidden_sizes=args.hidden_sizes,
            device=args.device,
        ).to(args.device)

        if optimizer is None:
            optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate)

        policy_learn = DQNPolicy(
            model=net,
            optim=optimizer,
            action_space=args.action_space,
            discount_factor=args.gamma,
            estimation_step=args.td_step,
            target_update_freq=args.target_update_freq,
        )

    if use_best:
        path = os.path.join(args.log_path, "best.pth")
        policy_learn.load_state_dict(torch.load(path, map_location=args.device))
        logging.info(f"Load best policy from {path}")

    if policy_opponent is None:
        if args.load_opponent:
            path = os.path.join(args.log_path, "opponent.pth")
            policy_opponent = deepcopy(policy_learn)
            policy_opponent.load_state_dict(torch.load(path))
            logging.info(f"Load opponent policy from {path}")
        else:
            policy_opponent = RandomPolicy(action_space=args.action_space)
            logging.info("No opponent policy given, use random policy.")

    if args.learned_go_first:
        policies = [policy_learn, policy_opponent]
    else:
        policies = [policy_opponent, policy_learn]
    env = get_env()
    ma_policy = MultiAgentPolicyManager(policies=policies, env=env)

    return ma_policy, optimizer, env.agents


def train(
    args: argparse.Namespace,
    policy_learn: BasePolicy | None = None,
    policy_opponent: BasePolicy | None = None,
    optimizer: torch.optim.Optimizer | None = None,
):
    train_envs = DummyVectorEnv([get_env for _ in range(args.train_env_num)])
    test_envs = DummyVectorEnv([get_env for _ in range(args.test_env_num)])

    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)

    # policy
    ma_policy, optim, agents = get_policy(
        args=args,
        policy_learn=policy_learn,
        policy_opponent=policy_opponent,
        optimizer=optimizer,
    )

    # collector
    train_collector = Collector(
        ma_policy,
        train_envs,
        VectorReplayBuffer(args.buffer_size, args.train_env_num),
        exploration_noise=True,
    )
    test_collector = Collector(ma_policy, test_envs, exploration_noise=True)

    # logger
    logger_factory = LoggerFactoryDefault()
    if args.logger == "wandb":
        logger_factory.logger_type = "wandb"
        logger_factory.wandb_project = args.wandb_project
    else:
        logger_factory.logger_type = "tensorboard"

    logger = logger_factory.create_logger(
        log_dir=args.log_path,
        experiment_name=os.path.join(args.task, str(args.seed), args.now),
        run_id=args.resume_id,
        config_dict=vars(args),
    )

    # train
    def train_fn(num_epoch: int, step_idx: int) -> None:
        agent_id = 0 if args.learned_go_first else 1
        ma_policy.policies[agents[agent_id]].set_eps(args.epsilon_train)

    def test_fn(num_epoch: int, step_idx: int) -> None:
        agent_id = 0 if args.learned_go_first else 1
        ma_policy.policies[agents[agent_id]].set_eps(args.epsilon_test)

    def stop_fn(mean_rewards: float) -> bool:
        return mean_rewards > args.win_rate

    def save_best_fn(policy: BasePolicy) -> None:
        path = os.path.join(args.log_path, "best.pth")
        agent_id = 0 if args.learned_go_first else 1
        torch.save(ma_policy.policies[agents[agent_id]].state_dict(), path)
        logging.info(f"Save best policy to {path}")

    def reward_metric(rewards: np.ndarray) -> np.ndarray:
        agent_id = 0 if args.learned_go_first else 1
        return rewards[:, agent_id]

    result = OffpolicyTrainer(
        policy=ma_policy,
        max_epoch=args.num_epochs,
        batch_size=args.batch_size,
        train_collector=train_collector,
        test_collector=test_collector,
        step_per_epoch=args.step_per_epoch,
        episode_per_test=args.episode_per_test,
        update_per_step=args.update_per_step,
        step_per_collect=args.step_per_collect,
        train_fn=train_fn,
        test_fn=test_fn,
        stop_fn=stop_fn,
        save_best_fn=save_best_fn,
        reward_metric=reward_metric,
        logger=logger,
        resume_from_log=args.resume_id is not None,
        test_in_train=False,
    ).run()

    return result, ma_policy.policies[agents[0 if args.learned_go_first else 1]]


def watch(
    args: argparse.Namespace,
    policy_learn: BasePolicy | None = None,
    policy_opponent: BasePolicy | None = None,
    use_best: bool = False,
):
    env = DummyVectorEnv([partial(get_env, render_mode="human")])
    ma_policy, optimizer, agents = get_policy(
        args=args,
        policy_learn=policy_learn,
        policy_opponent=policy_opponent,
        use_best=use_best,
    )
    ma_policy.policies[agents[0 if args.learned_go_first else 1]].set_eps(
        args.epsilon_test
    )
    collector = Collector(ma_policy, env, exploration_noise=True)
    result = collector.collect(
        n_episode=args.watch_episode_num,
        render=1 / args.render_fps,
        reset_before_collect=True,
    )

    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    args = get_args()
    args = configure_log_path(args)
    args = add_env_info(args)

    result, policy = train(args)

    if args.watch:
        result = watch(args, policy_learn=policy, use_best=True)
