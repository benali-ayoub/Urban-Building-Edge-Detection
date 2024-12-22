import gym
from gym import spaces
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import json
from datetime import datetime

from src.rl.approach_2.edge_agent import EdgeDetectionAgent, save_agent, visualize_results
from src.rl.approach_2.edge_env import EdgeDetectionEnv

def train_edge_detection(input_images, ground_truths, episodes=2000, save_dir='edge_detection_results'):
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(save_dir, exist_ok=True)
        
        # Create visualization directory
        vis_dir = os.path.join(save_dir, 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)
        
        # Initialize with first image
        env = EdgeDetectionEnv(input_images[0], ground_truths[0])
        agent = EdgeDetectionAgent(env)
        
        num_images = len(input_images)
        best_rewards = [-float('inf')] * num_images
        best_results = [None] * num_images
        best_metrics = [None] * num_images
        
        print("\nStarting Edge Detection Training...")
        with tqdm(total=episodes * num_images, desc='Training') as pbar:
            for episode in range(episodes):
                # Cycle through all images
                for img_idx in range(num_images):
                    env.input_image = input_images[img_idx]
                    env.ground_truth = ground_truths[img_idx]
                    env.input_image_inv = cv2.bitwise_not(input_images[img_idx])
                    
                    state = env.reset()
                    total_reward = 0
                    
                    for step in range(5):
                        action = agent.act(state)
                        next_state, reward, done, info = env.step(action)
                        
                        agent.remember(state, action, reward, next_state, done)
                        loss = agent.replay()
                        
                        total_reward += reward
                        state = next_state
                        
                        if done:
                            break
                    
                    if total_reward > best_rewards[img_idx]:
                        best_rewards[img_idx] = total_reward
                        best_results[img_idx] = env.best_result.copy()
                        best_metrics[img_idx] = env.best_metrics.copy()
                    
                    pbar.update(1)
                    pbar.set_postfix({
                        'reward': f'{total_reward:.3f}',
                        'epsilon': f'{agent.epsilon:.3f}'
                    })
                
                # Save agent periodically
                if episode % 100 == 0:
                    save_agent(agent, save_dir, f"{timestamp}_episode_{episode}")
                    
                # Visualize progress periodically
                if episode % 20 == 0:
                    plt.close('all')  # Close any existing figures
                    visualize_results(input_images, best_results, ground_truths, 
                                   episode, vis_dir)
        
        # Save final agent
        final_model_path = save_agent(agent, save_dir, f"{timestamp}_final")
        
        # Save training results
        results = {
            'timestamp': timestamp,
            'best_rewards': best_rewards,
            'best_metrics': best_metrics,
            'final_model_path': final_model_path
        }
        
        with open(os.path.join(save_dir, f'training_results_{timestamp}.json'), 'w') as f:
            json.dump(results, f, indent=4)
        
        return best_results, best_rewards, best_metrics, final_model_path

    except Exception as e:
        print(f"Error in training: {str(e)}")
        plt.close('all')
        return None, None, None, None