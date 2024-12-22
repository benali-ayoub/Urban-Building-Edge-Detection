"""
DQ agent for hyperparameter optimization.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
from typing import Dict, List, Tuple
from torch.optim.lr_scheduler import ReduceLROnPlateau


class DQNetwork(nn.Module):
    def __init__(self, state_size: int, action_sizes: Dict[str, int]):
        super().__init__()
        self.state_size = state_size
        self.action_sizes = action_sizes
        
        # Simpler architecture with correct input size
        self.shared = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU()
        )
        
        # Output heads
        self.action_heads = nn.ModuleDict({
            name: nn.Linear(256, size) 
            for name, size in action_sizes.items()
        })
    
    def forward(self, state: torch.Tensor) -> Dict[str, torch.Tensor]:
        shared_features = self.shared(state)
        return {
            name: head(shared_features) 
            for name, head in self.action_heads.items()
        }
        
class DQNAgent:
   def __init__(self, state_size: int, action_sizes: Dict[str, int],
                 learning_rate: float = 0.001,
                 gamma: float = 0.95,
                 epsilon: float = 1.0,
                 epsilon_min: float = 0.01,
                 epsilon_decay: float = 0.995,
                 memory_size: int = 10000):
       self.state_size = state_size
       self.action_sizes = action_sizes
       self.gamma = gamma
       self.epsilon = epsilon
       self.epsilon_min = epsilon_min
       self.epsilon_decay = epsilon_decay
       self.training = True
       
       self.memory = deque(maxlen=memory_size)
       self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
       
       # Initialize networks
       self.model = DQNetwork(state_size, action_sizes).to(self.device).float()
       self.target_model = DQNetwork(state_size, action_sizes).to(self.device).float()
       self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
       
       # Initial sync of target network
       self.update_target_network()
       self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=5,
            verbose=True
        )

   def train(self):
      """Set the agent to training mode."""
      self.training = True
      self.model.train()
            
   def eval(self):
      """Set the agent to evaluation mode."""
      self.training = False
      self.model.eval()
   
   def update_target_network(self):
       """Copy weights from main model to target model."""
       self.target_model.load_state_dict(self.model.state_dict())
   
   def remember(
       self,
       state: np.ndarray,
       action: Dict[str, int],
       reward: float,
       next_state: np.ndarray,
       done: bool
   ):
       self.memory.append((state, action, reward, next_state, done))
   
   def act(self, state: np.ndarray) -> Dict[str, int]:
        """
        Select an action based on the current state.
        
        Args:
            state: Current state observation
            
        Returns:
            Dictionary of actions for each parameter
        """
        # Use epsilon-greedy only during training
        if self.training and random.random() < self.epsilon:
            return {
                name: random.randrange(size)
                for name, size in self.action_sizes.items()
            }
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action_values = self.model(state_tensor)
            
            if self.training:
                # During training, use regular argmax
                return {
                    name: int(values.argmax().item())
                    for name, values in action_values.items()
                }
            else:
                # During evaluation, use softmax with temperature
                temperature = 0.5
                actions = {}
                for name, values in action_values.items():
                    probs = torch.softmax(values / temperature, dim=1)
                    actions[name] = int(probs.argmax().item())
                return actions
   
   def replay(self, batch_size: int):
        if len(self.memory) < batch_size:
            return None
            
        minibatch = random.sample(self.memory, batch_size)
        total_loss = 0
        
        for state, action, reward, next_state, done in minibatch:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
            reward_tensor = torch.FloatTensor([reward]).to(self.device)
            
            # Calculate target value
            if not done:
                next_action_values = self.target_model(next_state_tensor)
                next_max = sum(values.max() for values in next_action_values.values())
                target = reward_tensor + self.gamma * next_max
            else:
                target = reward_tensor
                
            # Get current Q-values
            current_action_values = self.model(state_tensor)
            batch_loss = torch.zeros(1, dtype=torch.float32, device=self.device)
            
            # Calculate loss for each action
            for name, values in current_action_values.items():
                q_value = values[0][action[name]]
                # Reshape tensors to ensure proper dimensions
                q_value = q_value.view(1)
                target = target.view(1)
                batch_loss += torch.nn.functional.mse_loss(q_value, target)
            
            total_loss += batch_loss
        
        # Calculate average loss
        avg_loss = total_loss / batch_size
        
        # Backpropagation
        self.optimizer.zero_grad()
        avg_loss.backward()
        self.optimizer.step()
        
        # Decay epsilon
        if self.training and self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return avg_loss.item()
   
   def save(self, filepath: str):
       torch.save({
           'model_state_dict': self.model.state_dict(),
           'optimizer_state_dict': self.optimizer.state_dict(),
           'epsilon': self.epsilon
       }, filepath)
   
   def load(self, filepath: str):
       checkpoint = torch.load(filepath)
       self.model.load_state_dict(checkpoint['model_state_dict'])
       self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
       self.epsilon = checkpoint['epsilon']
       self.update_target_network()