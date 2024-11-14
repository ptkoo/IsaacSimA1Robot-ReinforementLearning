from isaacsim import SimulationApp
 # Initialize simulation
simulation_app = SimulationApp({"headless": False})  # Run headless during training

import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import numpy as np
import carb
from omni.isaac.core import World
from omni.isaac.core.utils.prims import define_prim, get_prim_at_path
from omni.isaac.nucleus import get_assets_root_path
from omni.isaac.quadruped.robots import Unitree
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.utils.types import ArticulationAction
import matplotlib.pyplot as plt

import random

# Check if GPU is available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
EPISODES = 1000
GAMMA = 0.99
LR_ACTOR = 1e-4
LR_CRITIC = 5e-4
BATCH_SIZE = 256
MEMORY_SIZE = BATCH_SIZE * 4
TAU = 0.005
MAX_STEP = 1000

class Actor(nn.Module):
    def __init__(self, state_size, action_size):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_size)
        
    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        # Output joint torques
        actions = torch.tanh(self.fc3(x))
        return actions

class Critic(nn.Module):
    def __init__(self, state_size, action_size):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_size + action_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)
        
    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        value = self.fc3(x)
        return value

class OUActionNoise:
    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None, noise_scale=5.0):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.noise_scale = noise_scale
        self.reset()

    def __call__(self):
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape) * self.noise_scale
        )
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x_initial if self.x_initial is not None else np.zeros_like(self.mean)

class A1Environment:
    def __init__(self, physics_dt, render_dt):
        self.world = World(stage_units_in_meters=1.0, physics_dt=physics_dt, rendering_dt=render_dt)
        
        assets_root_path = get_assets_root_path()
        if assets_root_path is None:
            carb.log_error("Could not find Isaac Sim assets folder")
            
        # Create warehouse environment
        prim = get_prim_at_path("/World/Warehouse")
        if not prim.IsValid():
            prim = define_prim("/World/Warehouse", "Xform")
            asset_path = assets_root_path + "/Isaac/Environments/"
            prim.GetReferences().AddReference(asset_path)
            
        # Create A1 robot
        self.a1 = self.world.scene.add(
            Unitree(
                prim_path="/World/A1",
                name="A1",
                position=np.array([0, 0, 0.40]),
                physics_dt=physics_dt,
                model="A1"
            )
        )
        
        # Wait for robot to be fully initialized
        # self.world.scene.play()
        self.world.step(render=True)
        
        # Define observation and action spaces
        self.state_size = 27  # Joint positions (12) + velocities (12) + robot height + robot orientation(x , y)
        self.action_size = 12  # Joint torques for all 12 joints
        
        self.target_height = np.random.uniform(0.4, 0.6)  # Target standing height in meters

        # Define crouching joint positions
        self.default_dof_pos = {
            # Front Left Leg
            "FL_hip_joint": 0.0,    # Neutral hip position
            "FL_thigh_joint": 0.8,  # Thigh bent forward
            "FL_calf_joint": 0.2,  # Calf bent back
            
            # Front Right Leg
            "FR_hip_joint": 0.0,    # Neutral hip position
            "FR_thigh_joint": 0.8,  # Thigh bent forward
            "FR_calf_joint": -0.2,  # Calf bent back
            
            # Rear Left Leg
            "RL_hip_joint": 0,    # Neutral hip position
            "RL_thigh_joint": 0,  # Thigh bent forward
            "RL_calf_joint": 0,  # Calf bent back
            
            # Rear Right Leg
            "RR_hip_joint": 0,    # Neutral hip position
            "RR_thigh_joint": 0,  # Thigh bent forward
            "RR_calf_joint": 0   # Calf bent back
        }

        # Initialize joint indices dictionary
        self.joint_indices = {}
        for joint_name in self.default_dof_pos.keys():
            self.joint_indices[joint_name] = self.a1.get_dof_index(joint_name)

    def reset(self):
        # Reset the world
        self.world.reset()
        
        # Set target height
        self.target_height = np.random.uniform(0.4, 0.6)
        
        # Create arrays for all joint positions
        num_dofs = self.a1.num_dof
        default_pos = np.zeros(num_dofs)
        
        # Set default positions for known joints
        for joint_name, pos in self.default_dof_pos.items():
            idx = self.joint_indices[joint_name]
            default_pos[idx] = pos
            
        # Apply positions
        self.a1.set_joint_positions(default_pos)
        
        # PD control parameters - reduced gains
        kp = 0.9  # Reduced from 100.0
        kd = 0.08  # Reduced from 10.0
        
        # Step simulation multiple times to let physics settle
        for _ in range(50):  # Increased number of settling steps
            # Get current joint states
            current_pos = self.a1.get_joint_positions()
            current_vel = self.a1.get_joint_velocities()
            
            # Calculate PD control torques
            position_error = default_pos - current_pos
            velocity_error = 0 - current_vel  # Target velocity is 0
            
            # Calculate torques using PD control
            stiffness_action = kp * position_error + kd * velocity_error
            
            # Clip torques to max values
            max_torque = 33.5  # Maximum torque for A1 joints
            stiffness_action = np.clip(stiffness_action, -max_torque, max_torque)
            
            # Apply torques to all joints
            self.a1.set_joint_efforts(stiffness_action)
            self.world.step(render=True)
        
        return self._get_observation()

    def step(self, action):
        # Ensure action is numpy array
        action = np.array(action, dtype=np.float32)
        
        # Scale action from [-1, 1] to actual torque range
        max_torque = 33.5  # Maximum torque for A1 joints
        scaled_action = action * max_torque
        
        # PD control parameters
        kp = 0.9  # Position gain
        kd = 0.08  # Velocity gain
        
        # Get current joint states
        current_pos = self.a1.get_joint_positions()
        current_vel = self.a1.get_joint_velocities()
        
        # Get target positions from default pose
        target_pos = np.zeros(self.a1.num_dof)
        for joint_name, pos in self.default_dof_pos.items():
            idx = self.joint_indices[joint_name]
            target_pos[idx] = pos
        
        # Calculate PD control torques
        position_error = target_pos - current_pos
        velocity_error = 0 - current_vel  # Target velocity is 0
        
        # Combine PD control with action torques
        pd_torques = kp * position_error + kd * velocity_error
        total_torques = scaled_action + pd_torques
        
        # Clip final torques to max values
        total_torques = np.clip(total_torques, -max_torque, max_torque)
        
        # Apply total torques to all joints
        self.a1.set_joint_efforts(total_torques)
        
        # Step physics multiple times per control step for stability
        for _ in range(5):  # Substeps for stability
            self.world.step(render=True)
        
        # Get new state
        state = self._get_observation()
        
        # Calculate reward
        reward = self._calculate_reward(state)
        
        # Check if episode is done
        done = self._is_done(state)
        
        return state, reward, done

    def _get_observation(self):
        # Get robot state
        joint_positions = self.a1.get_joint_positions()
        joint_velocities = self.a1.get_joint_velocities()
        robot_position = self.a1.get_world_pose()[0]
        robot_orientation = self.a1.get_world_pose()[1]
        
        # Calculate roll (x) and pitch (y) from quaternion
        sinr_cosp = 2 * (robot_orientation[0] * robot_orientation[1] + robot_orientation[2] * robot_orientation[3])
        cosr_cosp = 1 - 2 * (robot_orientation[1] * robot_orientation[1] + robot_orientation[2] * robot_orientation[2])
        roll = np.arctan2(sinr_cosp, cosr_cosp)  # x-axis orientation

        sinp = 2 * (robot_orientation[0] * robot_orientation[2] - robot_orientation[3] * robot_orientation[1])
        pitch = np.arcsin(np.clip(sinp, -1.0, 1.0))  # y-axis orientation
        
        
        
        # Combine into state vector
        state = np.concatenate([
            joint_positions,      # 12 values
            joint_velocities,     # 12 values
            [robot_position[2]],  # Height (1 value)
            [roll],              # X-axis orientation (1 value)
            [pitch],             # Y-axis orientation (1 value)
        ])
        
        return state
            
    def _calculate_reward(self, state):
        # Extract state information
        joint_positions = state[:12]
        joint_velocities = state[12:24]
        height = state[24]  # Robot height
        roll = state[25]   # X-axis orientation
        pitch = state[26]  # Y-axis orientation
        
        # Get previous height if available, otherwise use current height
        prev_height = getattr(self, 'prev_height', height)
        self.prev_height = height  # Store current height for next step
        
        # 1. Height-related rewards
        height_progress = height - prev_height  # Positive when moving up
        target_height_error = abs(height - self.target_height)
        
        # Strongly reward upward movement, especially when below target
        if height < self.target_height:
            height_progress_reward = 5.0 * height_progress if height_progress > 0 else 2.0 * height_progress
        else:
            # Less reward/penalty when above target height
            height_progress_reward = 2.0 * height_progress if height_progress > 0 else 3.0 * height_progress
        
        # Basic height maintenance reward
        height_maintenance_reward = -1.0 * target_height_error
        
        # 2. Enhanced orientation rewards - exponential penalty for non-horizontal orientation
        orientation_reward = (
            -2.0 * (np.exp(abs(roll)) - 1) +    # Exponential penalty for roll
            -2.0 * (np.exp(abs(pitch)) - 1)      # Exponential penalty for pitch
        )
        
        # 3. Ground contact stability - reward even distribution of ground contact
        foot_positions = [
            joint_positions[2],  # FL foot
            joint_positions[5],  # FR foot
            joint_positions[8],  # RL foot
            joint_positions[11]  # RR foot
        ]
        foot_height_variance = np.var(foot_positions)
        ground_contact_reward = -2.0 * foot_height_variance
        
        # 4. Movement efficiency
        # Penalize rapid movements more when closer to target height
        height_ratio = min(height / self.target_height, 1.0)
        velocity_penalty = -0.05 * np.sum(np.square(joint_velocities)) * height_ratio
        
        # 5. Joint limit avoidance - softer penalty when moving up
        joint_limit_penalty = -0.02 * np.sum(np.square(joint_positions)) * height_ratio
        
        # Combine rewards with adjusted weights
        reward = (
            height_progress_reward * 1.5 +      # Strongest weight for upward progress
            height_maintenance_reward * 1.0 +    # Medium weight for height maintenance
            orientation_reward * 1.2 +          # Strong weight for staying horizontal
            ground_contact_reward * 0.8 +       # Medium weight for stability
            velocity_penalty * 0.3 +            # Lower weight for efficiency
            joint_limit_penalty * 0.3           # Lower weight for joint limits
        )
        
        # Success bonus with graduated rewards
        if height >= self.target_height * 0.8:  # Starting from 80% of target height
            progress_ratio = min((height - self.target_height * 0.8) / (self.target_height * 0.2), 1.0)
            if abs(roll) < 0.2 and abs(pitch) < 0.2:  # Relatively flat
                stability_bonus = 10.0 * progress_ratio
                reward += stability_bonus
                
                # Extra bonus for perfect stance
                if (abs(roll) < 0.1 and 
                    abs(pitch) < 0.1 and 
                    target_height_error < 0.02):
                    reward += 20.0
        
        # Extreme failure penalties
        if height < 0.1:  # Too low
            reward -= 50.0
        if abs(roll) > 0.5 or abs(pitch) > 0.5:  # Too tilted
            reward -= 30.0
        
        return reward
        
    def _is_done(self, state):
        height = state[24]  # Updated index for height
        roll = state[25]    # X-axis orientation
        pitch = state[26]   # Y-axis orientation
        
        # End episode if robot falls or achieves goal
        if height < 0.1:  # Robot has fallen
            print("robot fallen")
            return True
        if (abs(height - self.target_height) < 0.02 and 
            abs(roll) < 0.1 and 
            abs(pitch) < 0.1):
             # Robot has achieved target pose
            print("target pose reached")
            return True
        if abs(roll) > 1 or abs(pitch) > 1 :  # Robot is too tilted in any direction
            print("robot too tilted")
            return True
            
        return False

class DDPG:
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim).to(device)
        self.critic = Critic(state_dim, action_dim).to(device)
        self.target_actor = Actor(state_dim, action_dim).to(device)
        self.target_critic = Critic(state_dim, action_dim).to(device)
        
        # Copy weights
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LR_ACTOR)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=LR_CRITIC)
        
        self.max_action = max_action
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.noise = OUActionNoise(mean=np.zeros(action_dim), std_deviation=0.2)

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
    def select_action(self, state):
        state = torch.FloatTensor(state).to(device)
        action = self.actor(state).cpu().data.numpy()
        action += self.noise()
        return np.clip(action, -self.max_action, self.max_action)
        
    def train(self):
        if len(self.memory) < BATCH_SIZE:
            return
            
        # Sample from memory
        batch = random.sample(self.memory, BATCH_SIZE)
        state, action, reward, next_state, done = zip(*batch)
        
        state = torch.FloatTensor(np.array(state)).to(device)
        action = torch.FloatTensor(np.array(action)).to(device)
        reward = torch.FloatTensor(np.array(reward)).unsqueeze(1).to(device)
        next_state = torch.FloatTensor(np.array(next_state)).to(device)
        done = torch.FloatTensor(np.array(done)).unsqueeze(1).to(device)
        
        # Critic update
        target_actions = self.target_actor(next_state)
        target_q = self.target_critic(next_state, target_actions)
        target_q = reward + (1 - done) * GAMMA * target_q
        current_q = self.critic(state, action)
        critic_loss = nn.MSELoss()(current_q, target_q.detach())
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Actor update
        actor_loss = -self.critic(state, self.actor(state)).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Update target networks
        for param, target_param in zip(self.actor.parameters(), self.target_actor.parameters()):
            target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)
            
        for param, target_param in zip(self.critic.parameters(), self.target_critic.parameters()):
            target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)

def main():
   
    
    # Create environment
    env = A1Environment(physics_dt=1/400.0, render_dt=1/60.0)
    
    # Initialize agent
    state_dim = env.state_size
    action_dim = env.action_size
    max_action = 1.0  # Normalized torque commands
    agent = DDPG(state_dim, action_dim, max_action)
    
    # Training loop
    normalized_rewards =[]
    rewards = []
    for episode in range(EPISODES):
        state = env.reset()
        agent.noise.reset()
        episode_reward = 0
        
        for step in range(MAX_STEP):
            # Select action
            action = agent.select_action(state)
            
            # Step environment
            next_state, reward, done = env.step(action)
           
            # Store transition
            agent.store_transition(state, action, reward, next_state, done)
            
            # Train agent
            agent.train()
            
            episode_reward += reward
            state = next_state
            
            if done:
                break

        rewards.append(episode_reward)
        # Calculate running statistics
        reward_mean = np.mean(rewards)
        reward_std = np.std(rewards)
        
        # Normalize the episode reward to be around 0
        normalized_reward = (episode_reward - reward_mean) / (reward_std + 1e-8)
        normalized_rewards.append(normalized_reward)

        if episode % 10 == 0:
            print(f"Episode {episode}, Reward: {normalized_reward}")

         # Update the simulation world with rendering
        # env._world.step(render=True)  # Call the step function with rendering

    # Plot the rewards after training
    plt.figure(figsize=(10, 6))
    plt.plot(normalized_rewards, label='Normalized Rewards')
   
    plt.xlabel('Episode')
    plt.ylabel('Normalized Reward')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_rewards.png')
    plt.show()

    # Save trained model
    torch.save(agent.actor.state_dict(), "a1_stand_actor.pth")
    torch.save(agent.critic.state_dict(), "a1_stand_critic.pth")
    
    simulation_app.close()

if __name__ == "__main__":
    main()
