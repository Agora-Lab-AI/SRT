"""
Example script demonstrating forward pass usage of the Surgical Robot Transformer.
"""

import torch
from loguru import logger
from srt.main import (
    SurgicalRobotTransformer,
    ModelConfig,
    RobotObservation,
)


def run_forward_pass():
    # Initialize model and config
    config = ModelConfig()
    model = SurgicalRobotTransformer(config)
    model.eval()  # Set to evaluation mode

    # Create sample camera images (simulating robot observations)
    # Normally these would come from your robot's cameras
    sample_image = torch.zeros((3, 224, 224))  # [C, H, W] format

    # Create observation object containing all camera views
    observation = RobotObservation(
        stereo_left=sample_image,
        stereo_right=sample_image,
        wrist_left=sample_image,
        wrist_right=sample_image,
    )

    # Perform forward pass
    with torch.no_grad():
        try:
            action = model(observation)

            # Extract predicted actions
            left_pos = action.left_pos.numpy()  # [3] - xyz position
            left_rot = action.left_rot.numpy()  # [6] - 6D rotation
            left_grip = (
                action.left_gripper.numpy()
            )  # [1] - gripper angle

            right_pos = action.right_pos.numpy()  # [3]
            right_rot = action.right_rot.numpy()  # [6]
            right_grip = action.right_gripper.numpy()  # [1]

            logger.info(f"Left arm position: {left_pos}")
            logger.info(f"Left arm rotation: {left_rot}")
            logger.info(f"Left gripper angle: {left_grip}")

            logger.info(f"Right arm position: {right_pos}")
            logger.info(f"Right arm rotation: {right_rot}")
            logger.info(f"Right gripper angle: {right_grip}")

            return action

        except Exception as e:
            logger.error(f"Error during forward pass: {str(e)}")
            raise


if __name__ == "__main__":
    # Set up logging
    logger.add("srt_inference.log")
    logger.info("Starting SRT forward pass example")

    action = run_forward_pass()

    logger.info("Forward pass completed successfully")
