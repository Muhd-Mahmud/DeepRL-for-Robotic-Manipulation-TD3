
import time
import os
import gymnasium as gym
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import robosuite as suite
from robosuite.wrappers import GymWrapper

if __name__ == "__main__":

    os.makedirs("tmp/td3", exist_ok=True)

    env_name = "Door"

    # Load Panda's default composite controller
    controller_config = suite.load_composite_controller_config(
        robot="Panda"
    )

    # Load joint velocity control for the Panda arm
    joint_velocity_config = suite.load_part_controller_config(
        default_controller="JOINT_VELOCITY"
    )

    # Replace only the right arm controller
    gripper_config = controller_config["body_parts"]["right"].get(
        "gripper", {"type": "GRIP"}
    )

    controller_config["body_parts"]["right"] = {
        **joint_velocity_config,
        "gripper": gripper_config,
    }

    # Verify configuration before creating environment
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

    # Create environment
    env = suite.make(
        env_name,
        robots=["Panda"],
        controller_configs=controller_config,
        has_renderer=True,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        render_camera="frontview",
        horizon=300,
        reward_shaping=True,
        control_freq=20,
    )

    env = GymWrapper(env)

    env.close()
