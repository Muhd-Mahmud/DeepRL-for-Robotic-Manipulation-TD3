import  time
import os
import numpy as np
import gymnasium as gym
import torch
import robosuite as suite

from robosuite.wrappers import GymWrapper
from network import CriticNetwork, ActorNetwork


if __name__ == "__main__":

    # ==========================================
    # 1. CREATE CHECKPOINT DIRECTORY
    # ==========================================

    os.makedirs("tmp/td3", exist_ok=True)

    # ==========================================
    # 2. SELECT ENVIRONMENT
    # ==========================================

    env_name = "Door"

    # ==========================================
    # 3. LOAD PANDA CONTROLLER
    # ==========================================

    controller_config = suite.load_composite_controller_config(
        robot="Panda"
    )

    # Load joint velocity controller
    joint_velocity_config = suite.load_part_controller_config(
        default_controller="JOINT_VELOCITY"
    )

    # Preserve Panda gripper configuration
    gripper_config = controller_config["body_parts"]["right"].get(
        "gripper", {"type": "GRIP"}
    )

    # Add gripper to joint velocity configuration
    joint_velocity_config["gripper"] = gripper_config

    # Replace default arm controller
    controller_config["body_parts"]["right"] = (
        joint_velocity_config
    )

    # Verify controller configuration
    print("Composite controller:", controller_config["type"])

    print(
        "Arm controller:",
        controller_config["body_parts"]["right"]["type"]
    )

    assert controller_config["type"] == "BASIC"
    assert (
        controller_config["body_parts"]["right"]["type"]
        == "JOINT_VELOCITY"
    )

    # ==========================================
    # 4. CREATE ROBOSUITE ENVIRONMENT
    # ==========================================

    env = suite.make(
        env_name,
        robots=["Panda"],
        controller_configs=controller_config,
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        horizon=300,
        reward_shaping=True,
        control_freq=20,
    )

    print("Environment created successfully!")

    # ==========================================
    # 5. APPLY GYM WRAPPER
    # ==========================================

    env = GymWrapper(env)

    # ==========================================
    # 6. GET OBSERVATION AND ACTION DIMENSIONS
    # ==========================================

    input_dims = env.observation_space.shape
    n_actions = env.action_space.shape[0]

    print("Observation dimensions:", input_dims)
    print("Action dimensions:", n_actions)

    # ==========================================
    # 7. INITIALIZE CRITIC NETWORK
    # ==========================================

    critic_network = CriticNetwork(
        input_dims=input_dims,
        n_actions=n_actions,
        name="critic_1",
        checkpoint_dir="tmp/td3"
    )

    print("Critic Network initialized successfully!")

    # ==========================================
    # 8. INITIALIZE ACTOR NETWORK
    # ==========================================

    actor_network = ActorNetwork(
        input_dims=input_dims,
        n_actions=n_actions,
        name="actor",
        checkpoint_dir="tmp/td3"
    )

    print("Actor Network initialized successfully!")

    # ==========================================
    # 9. TEST ACTOR AND CRITIC NETWORKS
    # ==========================================

    # Get a real observation
    observation, _ = env.reset()

    # Convert observation to tensor
    state = torch.from_numpy(
        np.asarray(observation, dtype=np.float32).reshape(1, -1)
    ).to(actor_network.device)

    # Actor generates an action
    with torch.no_grad():
        action = actor_network(state)

    # Critic evaluates the action
    with torch.no_grad():
        q_value = critic_network(
            state.to(critic_network.device),
            action.to(critic_network.device)
        )

    print("Actor output shape:", action.shape)
    print("Critic output shape:", q_value.shape)
    print("Predicted Q-value:", q_value.item())

    # ==========================================
    # 10. TEST SIMULATION
    # ==========================================

    # Convert actor output to NumPy
    test_action = action.squeeze(0).cpu().numpy()

    # Execute action in environment
    next_observation, reward, terminated, truncated, _ = (
        env.step(test_action)
    )

    print("Simulation step successful!")
    print("Reward:", reward)
    print("Terminated:", terminated)
    print("Truncated:", truncated)

    # ==========================================
    # 11. CLOSE ENVIRONMENT
    # ==========================================

    env.close()

    print("All initialization tests completed!")

