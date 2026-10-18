"""TD3 training entry point for the robosuite Panda Door task.

Copy this file into your project as main.py once td3Agent.py is installed.
"""

import os
import time
import robosuite as suite
from robosuite.wrappers import GymWrapper
from torch.utils.tensorboard import SummaryWriter

from td3_agent import TD3Agent


NUM_EPISODES = 10          # A short integration test, NOT full training
EPISODE_STEPS = 300       # Matches the Door environment horizon
RANDOM_STEPS = 1_000      # Collect initial random experiences
BATCH_SIZE = 128


def make_environment():
    controller_config = suite.load_composite_controller_config(robot="Panda")
    joint_velocity_config = suite.load_part_controller_config(
        default_controller="JOINT_VELOCITY"
    )

    # Use the key path that worked for your installed robosuite version.
    gripper_config = controller_config["body_parts"]["right"].get(
        "gripper", {"type": "GRIP"}
    )
    joint_velocity_config["gripper"] = gripper_config
    controller_config["body_parts"]["right"] = joint_velocity_config

    print("Composite controller:", controller_config["type"])
    print("Arm controller:", controller_config["body_parts"]["right"]["type"])

    env = suite.make(
        "Door",
        robots=["Panda"],
        controller_configs=controller_config,
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        horizon=EPISODE_STEPS,
        reward_shaping=True,
        control_freq=20,
    )
    return GymWrapper(env)


def main():
    os.makedirs("tmp/td3", exist_ok=True)
    env = make_environment()
    writer = SummaryWriter(log_dir="runs/td3_panda_door")

    try:
        input_dims = env.observation_space.shape
        n_actions = env.action_space.shape[0]
        print("Observation dimensions:", input_dims)
        print("Action dimensions:", n_actions)

        agent = TD3Agent(
            input_dims=input_dims,
            n_actions=n_actions,
            action_low=env.action_space.low,
            action_high=env.action_space.high,
            batch_size=BATCH_SIZE,
            checkpoint_dir="tmp/td3",
        )

        total_steps = 0
        for episode in range(NUM_EPISODES):
            state, _ = env.reset()
            episode_reward = 0.0
            last_losses = None

            for step in range(EPISODE_STEPS):
                if total_steps < RANDOM_STEPS:
                    action = env.action_space.sample()
                else:
                    action = agent.choose_action(state, explore=True)

                next_state, reward, terminated, truncated, _ = env.step(action)

                # Store TERMINATED, not (terminated or truncated), to preserve
                # TD bootstrapping across time-limit episode endings.
                agent.remember(state, action, reward, next_state, terminated)

                if total_steps >= RANDOM_STEPS:
                    last_losses = agent.learn()
                    if last_losses is not None:
                        writer.add_scalar("loss/critic", last_losses["critic_loss"], total_steps)
                        if last_losses["actor_loss"] is not None:
                            writer.add_scalar("loss/actor", last_losses["actor_loss"], total_steps)

                episode_reward += float(reward)
                total_steps += 1
                state = next_state
                if terminated or truncated:
                    break

            writer.add_scalar("train/episode_reward", episode_reward, episode)
            print(
                f"Episode {episode + 1}/{NUM_EPISODES} | "
                f"Reward: {episode_reward:.3f} | "
                f"Steps: {step + 1} | "
                f"Replay size: {len(agent.memory)}"
            )

            if (episode + 1) % 5 == 0:
                agent.save_models()

        agent.save_models()
        print("TD3 training test completed. Network weights saved in tmp/td3/")
    finally:
        writer.close()
        env.close()