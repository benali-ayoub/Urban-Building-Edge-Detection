"""
raining script for the DQN agent.
"""

import numpy as np
from typing import List, Tuple
from pathlib import Path
import cv2
from tqdm import tqdm

from src.rl.edge_agent import DQN
from src.rl.replay_memory import *
from .environment import PreprocessingEnv
from .dqn_agent import DQNAgent
import random
from sklearn.model_selection import train_test_split

import torch
import torch.nn.functional as F


def load_dataset(data_dir: str) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Load image and ground truth pairs."""
    data_dir = Path(data_dir)
    image_files = sorted(data_dir.glob("images_inter/*.png"))
    gt_files = sorted(data_dir.glob("gt_edge_inter/*.png"))

    dataset = []
    for img_path, gt_path in zip(image_files, gt_files):
        img = cv2.imread(str(img_path))
        gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
        dataset.append((img, gt))

    return dataset


def train_agent(
    data_dir: str,
    epochs: int = 100,
    batch_size: int = 32,
    save_path: str = "models/preprocessing_agent.pth",
    eval_interval: int = 5,
):
    dataset = load_dataset(data_dir)
    train_set, val_set = train_test_split(dataset, test_size=0.2)

    env = PreprocessingEnv(train_set[0][0], train_set[0][1])
    agent = DQNAgent(
        state_size=env.observation_space.shape[0],
        action_sizes={name: space.n for name, space in env.action_space.spaces.items()},
    )

    best_val_reward = float("-inf")
    patience = 10
    no_improve = 0

    for epoch in range(epochs):
        # Training
        agent.train()
        total_reward = 0
        losses = []

        random.shuffle(train_set)
        for img, gt in tqdm(train_set, desc=f"Epoch {epoch + 1}/{epochs}"):
            env = PreprocessingEnv(img, gt)
            state = env.reset()
            episode_reward = 0

            for step in range(5):
                action = agent.act(state)
                next_state, reward, done, info = env.step(action)
                agent.remember(state, action, reward, next_state, done)
                loss = agent.replay(batch_size)

                if loss is not None:
                    losses.append(loss)

                state = next_state
                episode_reward += reward

            total_reward += episode_reward / 5

        avg_train_reward = total_reward / len(train_set)

        # Validation
        if epoch % eval_interval == 0:
            agent.eval()
            val_rewards = []

            for img, gt in val_set:
                env = PreprocessingEnv(img, gt)
                state = env.reset()
                val_reward = 0

                for _ in range(5):
                    action = agent.act(state)
                    next_state, reward, done, _ = env.step(action)
                    val_reward += reward
                    state = next_state

                val_rewards.append(val_reward / 5)

            avg_val_reward = np.mean(val_rewards)

            print(f"Epoch {epoch + 1}")
            print(f"  Training reward: {avg_train_reward:.4f}")
            print(f"  Validation reward: {avg_val_reward:.4f}")
            print(f"  Average loss: {np.mean(losses):.4f}")
            print(f"  Epsilon: {agent.epsilon:.4f}")

            if avg_val_reward > best_val_reward:
                best_val_reward = avg_val_reward
                agent.save(save_path)
                print(
                    f"  New best model saved with validation reward {best_val_reward:.4f}"
                )
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    print("Early stopping triggered")
                    break

        if epoch % 3 == 0:
            agent.update_target_network()

    return agent


def get_optimal_parameters(
    agent_path: str, image: np.ndarray, num_samples: int = 5
) -> dict:
    """Get optimal preprocessing parameters for an image."""
    # Create environment without ground truth for inference
    env = PreprocessingEnv(image)

    # Load agent
    agent = DQNAgent(
        state_size=env.observation_space.shape[0],
        action_sizes={name: space.n for name, space in env.action_space.spaces.items()},
    )
    agent.load(agent_path)
    agent.eval()  # Set to evaluation mode

    best_params = None
    best_reward = float("-inf")

    # Try multiple times to get best parameters
    for _ in range(num_samples):
        state = env.reset()
        action = agent.act(state)
        next_state, reward, done, info = env.step(action)

        parameters = {
            "bilateral_d": action["bilateral_d"] + 1,
            "bilateral_sigma": action["bilateral_sigma"] + 1,
            "clahe_clip": (action["clahe_clip"] + 1) / 10.0,
            "clahe_grid": 4 + (action["clahe_grid"] * 4),
            "gaussian_kernel": 2 * action["gaussian_kernel"] + 3,
            "threshold_value": action["threshold_value"] + 100,
        }

        # Process image with current parameters
        processed = env._apply_preprocessing(action)
        current_reward = env._calculate_reward(processed)

        if current_reward > best_reward:
            best_reward = current_reward
            best_params = parameters.copy()

    print(f"Best reward achieved: {best_reward:.4f}")
    return best_params


def train_edge_dqn(env, num_episodes=6, save_dir="saved_models"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create save directory if it doesn't exist
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Initialize networks and optimizer
    policy_net = DQN(
        env.observation_space.shape[0],
        env.observation_space.shape[1],
        env.action_space.nvec.prod(),
    ).to(device)
    target_net = DQN(
        env.observation_space.shape[0],
        env.observation_space.shape[1],
        env.action_space.nvec.prod(),
    ).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = torch.optim.Adam(policy_net.parameters())
    memory = ReplayMemory(10000)
    strategy = EpsilonGreedyStrategy(1.0, 0.05, 0.001)

    batch_size = 32
    gamma = 0.999
    target_update = 10

    # Track best model
    best_reward = float("-inf")
    rewards_history = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        state = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(device)
        total_reward = 0

        while True:
            # Select action
            sample = random.random()
            eps_threshold = strategy.get_exploration_rate(episode)
            if sample > eps_threshold:
                with torch.no_grad():
                    action = policy_net(state).max(1)[1].view(1, 1)
            else:
                action = torch.tensor(
                    [[random.randrange(env.action_space.nvec.prod())]], device=device
                )

            # Convert flat action back to multi-discrete
            action_tuple = np.unravel_index(action.item(), env.action_space.nvec)

            # Take action
            next_state, reward, done, _, _ = env.step(action_tuple)
            total_reward += reward

            next_state = (
                torch.FloatTensor(next_state).unsqueeze(0).unsqueeze(0).to(device)
            )
            reward = torch.tensor([reward], device=device)

            # Store transition
            memory.push(state, action, next_state, reward)
            state = next_state

            # Perform optimization
            if len(memory) >= batch_size:
                transitions = memory.sample(batch_size)
                batch = Transition(*zip(*transitions))

                non_final_mask = torch.tensor(
                    tuple(map(lambda s: s is not None, batch.next_state)), device=device
                )
                non_final_next_states = torch.cat(
                    [s for s in batch.next_state if s is not None]
                )

                state_batch = torch.cat(batch.state)
                action_batch = torch.cat(batch.action)
                reward_batch = torch.cat(batch.reward)

                state_action_values = policy_net(state_batch).gather(1, action_batch)

                next_state_values = torch.zeros(batch_size, device=device)
                next_state_values[non_final_mask] = (
                    target_net(non_final_next_states).max(1)[0].detach()
                )
                expected_state_action_values = (
                    next_state_values * gamma
                ) + reward_batch

                # Compute loss and optimize
                loss = F.smooth_l1_loss(
                    state_action_values, expected_state_action_values.unsqueeze(1)
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if done:
                break

        # Update target network
        if episode % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())

        # Save rewards history
        rewards_history.append(total_reward)

        # Save best model
        if total_reward > best_reward:
            best_reward = total_reward
            torch.save(
                {
                    "episode": episode,
                    "model_state_dict": policy_net.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "reward": total_reward,
                },
                save_dir / "best_model.pth",
            )

        # Save checkpoint every 100 episodes
        if episode % 100 == 0:
            torch.save(
                {
                    "episode": episode,
                    "model_state_dict": policy_net.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "reward": total_reward,
                    "rewards_history": rewards_history,
                },
                save_dir / f"checkpoint_episode_{episode}.pth",
            )

        print(
            f"Episode {episode}: Total reward: {total_reward}, Best reward: {best_reward}"
        )

    # Save final model
    torch.save(
        {
            "episode": num_episodes - 1,
            "model_state_dict": policy_net.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "reward": total_reward,
            "rewards_history": rewards_history,
        },
        save_dir / "final_model.pth",
    )

    # Save rewards history
    np.save(save_dir / "rewards_history.npy", np.array(rewards_history))

    return policy_net, rewards_history
