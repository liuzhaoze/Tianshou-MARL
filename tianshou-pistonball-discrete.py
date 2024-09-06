import argparse
import os
from datetime import datetime
from functools import partial

import gymnasium
import numpy as np
import torch
from pettingzoo.butterfly import pistonball_v6
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv, PettingZooEnv
from tianshou.highlevel.logger import LoggerFactoryDefault
from tianshou.policy import BasePolicy, DQNPolicy, MultiAgentPolicyManager
from tianshou.trainer import OffpolicyTrainer
from tianshou.utils.net.common import Net


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    # environment
    parser.add_argument("--task", type=str, default="PistonBall-v6")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--piston-num",
        type=int,
        default=3,
        help="Number of pistons(agents) in the environment.",
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

    # policy
    parser.add_argument("--hidden-sizes", type=int, nargs="*", default=[64, 64])
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.9,
    )
    parser.add_argument("--td-step", type=int, default=100)
    parser.add_argument("--target-update-freq", type=int, default=320)
    parser.add_argument("--epsilon-train", type=float, default=0.1)
    parser.add_argument("--epsilon-test", type=float, default=0.05)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )

    # training
    parser.add_argument("--buffer-size", type=int, default=2000)
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=3,
        help="Number of total epochs in training process.",
    )
    parser.add_argument("--step-per-epoch", type=int, default=500)
    parser.add_argument("--step-per-collect", type=int, default=10)
    parser.add_argument("--episode-per-test", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--update-per-step", type=float, default=0.1)

    # log
    parser.add_argument(
        "--logger", type=str, default="tensorboard", choices=["tensorboard", "wandb"]
    )
    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument("--wandb-project", type=str, default="tianshou-pistonball")
    parser.add_argument("--resume-id", type=str, default=None)

    # watch
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--watch-only", action="store_true")
    parser.add_argument("--watch-episode-num", type=int, default=1)
    parser.add_argument("--render-fps", type=int, default=60)
    parser.add_argument("--model-path", type=str, default=None)

    return parser.parse_known_args()[0]


def configure_log_path(args: argparse.Namespace) -> argparse.Namespace:
    args.now = datetime.now().strftime("%Y%m%d-%H%M%S")
    args.log_path = os.path.join(args.logdir, args.now)
    if args.logger == "wandb":
        os.makedirs(os.path.join(args.log_path, "wandb"), mode=755, exist_ok=True)
    return args


def get_env(args: argparse.Namespace, render_mode: str | None = None) -> PettingZooEnv:
    return PettingZooEnv(
        pistonball_v6.env(
            continuous=False, n_pistons=args.piston_num, render_mode=render_mode
        )
    )


def add_env_info(args: argparse.Namespace) -> argparse.Namespace:
    env = get_env(args)
    args.state_space = (
        env.observation_space["observation"]
        if isinstance(env.observation_space, gymnasium.spaces.Dict)
        else env.observation_space
    )  # `env.observation_space` may has `action_mask` key
    args.state_shape = args.state_space.shape or int(args.state_space.n)
    args.action_space = env.action_space
    args.action_shape = args.action_space.shape or int(args.action_space.n)
    args.agents = env.agents
    return args


def get_policy(
    args: argparse.Namespace,
    policies: list[BasePolicy] | None = None,
    optimizers: list[torch.optim.Optimizer] | None = None,
    use_best: bool = False,
):
    if policies is None:
        policies = []
        optimizers = []
        for _ in range(args.piston_num):
            net = Net(
                state_shape=args.state_shape,
                action_shape=args.action_shape,
                hidden_sizes=args.hidden_sizes,
                device=args.device,
            ).to(args.device)
            optim = torch.optim.Adam(net.parameters(), lr=args.learning_rate)
            policy = DQNPolicy(
                model=net,
                optim=optim,
                action_space=args.action_space,
                discount_factor=args.gamma,
                estimation_step=args.td_step,
                target_update_freq=args.target_update_freq,
            )
            policies.append(policy)
            optimizers.append(optim)

    if use_best:
        path = args.model_path or os.path.join(args.log_path, "best.pth")
        parameters = torch.load(path, map_location=args.device)
        [p.load_state_dict(parameters[a]) for a, p in zip(args.agents, policies)]
        print(f"Load best policy from {path}")

    ma_policy = MultiAgentPolicyManager(policies=policies, env=get_env(args))
    return ma_policy, optimizers


def train(
    args: argparse.Namespace,
    policies: list[BasePolicy] | None = None,
    optimizers: list[torch.optim.Optimizer] | None = None,
):
    train_envs = DummyVectorEnv(
        [partial(get_env, args) for _ in range(args.train_env_num)]
    )
    test_envs = DummyVectorEnv(
        [partial(get_env, args) for _ in range(args.test_env_num)]
    )

    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)

    # policy
    ma_policy, optims = get_policy(args, policies=policies, optimizers=optimizers)

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
        [p.set_eps(args.epsilon_train) for p in ma_policy.policies.values()]

    def test_fn(epoch: int, env_step: int | None) -> None:
        [p.set_eps(args.epsilon_test) for p in ma_policy.policies.values()]

    def stop_fn(mean_rewards: float) -> bool:
        return False

    def save_best_fn(policy: MultiAgentPolicyManager | BasePolicy) -> None:
        path = os.path.join(args.log_path, "best.pth")
        parameters = {a: p.state_dict() for a, p in policy.policies.items()}
        torch.save(parameters, path)
        print(f"Save best policy to {path}")

    def reward_metric(rewards: np.ndarray) -> np.ndarray:
        return rewards[:, 0]

    result = OffpolicyTrainer(
        policy=ma_policy,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=args.num_epochs,
        step_per_epoch=args.step_per_epoch,
        step_per_collect=args.step_per_collect,
        episode_per_test=args.episode_per_test,
        batch_size=args.batch_size,
        train_fn=train_fn,
        test_fn=test_fn,
        stop_fn=stop_fn,
        save_best_fn=save_best_fn,
        update_per_step=args.update_per_step,
        logger=logger,
        test_in_train=False,
        reward_metric=reward_metric,
    ).run()

    return result, ma_policy


def watch(
    args: argparse.Namespace,
    policies: list[BasePolicy] | None = None,
    use_best: bool = False,
):
    env = DummyVectorEnv([partial(get_env, args, render_mode="human")])
    ma_policy, optims = get_policy(args, policies=policies, use_best=use_best)
    [p.set_eps(args.epsilon_test) for p in ma_policy.policies.values()]
    collector = Collector(ma_policy, env, exploration_noise=True)
    result = collector.collect(
        n_episode=args.watch_episode_num,
        render=1 / args.render_fps,
        reset_before_collect=True,
    )

    return result


if __name__ == "__main__":
    args = get_args()
    args = configure_log_path(args)
    args = add_env_info(args)

    if not args.watch_only:
        result, ma_policy = train(args)
    else:
        ma_policy, _ = get_policy(args)

    if args.watch or args.watch_only:
        result = watch(args, list(ma_policy.policies.values()), use_best=True)
