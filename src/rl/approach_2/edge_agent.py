import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import os
from datetime import datetime
import matplotlib.pyplot as plt
from src.rl.approach_2.edge_env import DQN


class EdgeDetectionAgent:
    def __init__(self, env, memory_size=20000):
        try:
            self.env = env
            self.memory = deque(maxlen=memory_size)
            self.batch_size = 32
            self.gamma = 0.99
            self.epsilon = 1.0
            self.epsilon_min = 0.05
            self.epsilon_decay = 0.995
            self.tau = 0.001
            self.learning_rate = 0.0001

            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"Using device: {self.device}")

            self.n_actions = env.action_space.shape[0]

            self.model = DQN(env.observation_space.shape, self.n_actions).to(
                self.device
            )
            self.target_model = DQN(env.observation_space.shape, self.n_actions).to(
                self.device
            )
            self.target_model.load_state_dict(self.model.state_dict())

            self.optimizer = optim.AdamW(
                self.model.parameters(), lr=self.learning_rate, weight_decay=0.01
            )

            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="max", factor=0.5, patience=30, verbose=True
            )

            self.action_low = env.action_space.low
            self.action_high = env.action_space.high
            self.action_mean = (self.action_high + self.action_low) / 2
            self.action_std = (self.action_high - self.action_low) / 4

        except Exception as e:
            print(f"Error initializing agent: {str(e)}")
            raise

    def act(self, state):
        try:
            if random.random() < self.epsilon:
                # Smart random sampling around mean
                action = np.random.normal(self.action_mean, self.action_std)
                return np.clip(action, self.action_low, self.action_high)

            with torch.no_grad():
                state = (
                    torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(self.device)
                )
                actions = self.model(state).cpu().numpy()[0]
                return np.clip(actions, self.action_low, self.action_high)

        except Exception as e:
            print(f"Error in act: {str(e)}")
            return self.action_mean

    def remember(self, state, action, reward, next_state, done):
        try:
            state_small = cv2.resize(state, (32, 32))
            next_state_small = cv2.resize(next_state, (32, 32))
            self.memory.append((state_small, action, reward, next_state_small, done))
        except Exception as e:
            print(f"Error in remember: {str(e)}")

    def replay(self):
        try:
            if len(self.memory) < self.batch_size:
                return None

            batch = random.sample(self.memory, self.batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)

            states = torch.FloatTensor(np.array(states)).unsqueeze(1).to(self.device)
            next_states = (
                torch.FloatTensor(np.array(next_states)).unsqueeze(1).to(self.device)
            )
            actions = torch.FloatTensor(np.array(actions)).to(self.device)
            rewards = torch.FloatTensor(rewards).to(self.device)
            dones = torch.FloatTensor(dones).to(self.device)

            current_q = self.model(states)
            next_q = self.target_model(next_states).detach()

            target_q = rewards + (1 - dones) * self.gamma * torch.max(next_q, dim=1)[0]

            loss = nn.SmoothL1Loss()(
                current_q, target_q.unsqueeze(1).repeat(1, self.n_actions)
            )

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            self._soft_update()

            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

            return loss.item()

        except Exception as e:
            print(f"Error in replay: {str(e)}")
            return None

    def _soft_update(self):
        try:
            for target_param, param in zip(
                self.target_model.parameters(), self.model.parameters()
            ):
                target_param.data.copy_(
                    self.tau * param.data + (1.0 - self.tau) * target_param.data
                )
        except Exception as e:
            print(f"Error in soft update: {str(e)}")


def save_agent(agent, save_dir, timestamp=None):
    try:
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create save directory if it doesn't exist
        save_path = os.path.join(save_dir, "saved_agents")
        os.makedirs(save_path, exist_ok=True)

        # Save the model state dictionaries
        model_save = {
            "model_state": agent.model.state_dict(),
            "target_model_state": agent.target_model.state_dict(),
            "optimizer_state": agent.optimizer.state_dict(),
            "epsilon": agent.epsilon,
            "action_mean": agent.action_mean,
            "action_std": agent.action_std,
            "action_low": agent.action_low,
            "action_high": agent.action_high,
        }

        model_path = os.path.join(save_path, f"edge_detection_agent_{timestamp}.pth")
        torch.save(model_save, model_path)

        print(f"\nAgent saved successfully to: {model_path}")
        return model_path

    except Exception as e:
        print(f"Error saving agent: {str(e)}")
        return None


def load_agent(env, model_path):
    agent = EdgeDetectionAgent(env)

    # Load saved state
    checkpoint = torch.load(model_path, map_location=agent.device)

    # Load model states
    agent.model.load_state_dict(checkpoint["model_state"])
    agent.target_model.load_state_dict(checkpoint["target_model_state"])
    agent.optimizer.load_state_dict(checkpoint["optimizer_state"])

    # Load other parameters
    agent.epsilon = checkpoint["epsilon"]
    agent.action_mean = checkpoint["action_mean"]
    agent.action_std = checkpoint["action_std"]
    agent.action_low = checkpoint["action_low"]
    agent.action_high = checkpoint["action_high"]

    print(f"\nAgent loaded successfully from: {model_path}")
    return agent


def visualize_results(inputs, results, truths, episode, save_dir):
    try:
        n_images = len(inputs)
        fig = plt.figure(figsize=(20, 5 * n_images))

        for i in range(n_images):
            # Original input
            ax1 = plt.subplot(n_images, 4, i * 4 + 1)
            ax1.imshow(inputs[i], cmap="gray", vmin=0, vmax=255)
            ax1.set_title(f"Input {i+1}")
            ax1.axis("off")

            # Inverted input
            ax2 = plt.subplot(n_images, 4, i * 4 + 2)
            ax2.imshow(cv2.bitwise_not(inputs[i]), cmap="gray", vmin=0, vmax=255)
            ax2.set_title(f"Inverted {i+1}")
            ax2.axis("off")

            # Edge detection result
            ax3 = plt.subplot(n_images, 4, i * 4 + 3)
            if results[i] is not None:
                ax3.imshow(results[i], cmap="gray", vmin=0, vmax=255)
            else:
                ax3.imshow(np.zeros_like(inputs[i]), cmap="gray", vmin=0, vmax=255)
            ax3.set_title(f"Edge Detection {i+1}")
            ax3.axis("off")

            # Ground truth
            ax4 = plt.subplot(n_images, 4, i * 4 + 4)
            ax4.imshow(truths[i], cmap="gray", vmin=0, vmax=255)
            ax4.set_title(f"Ground Truth {i+1}")
            ax4.axis("off")

        plt.tight_layout()

        # Save the figure
        save_path = os.path.join(save_dir, f"progress_{episode:04d}.png")
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close(fig)

        # Optional: Display the saved image to verify
        if episode % 100 == 0:  # Show every 100th image
            saved_img = cv2.imread(save_path)
            if saved_img is not None:
                plt.figure(figsize=(20, 5 * n_images))
                plt.imshow(cv2.cvtColor(saved_img, cv2.COLOR_BGR2RGB))
                plt.axis("off")
                plt.show()
                plt.close()

    except Exception as e:
        print(f"Error in visualization: {str(e)}")
        plt.close("all")  # Close all figures in case of error
