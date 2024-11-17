"""
Surgical Robot Transformer (SRT) - Production Implementation
A PyTorch implementation of the paper "Surgical Robot Transformer (SRT):
Imitation Learning for Surgical Tasks"

Author: Assistant
License: MIT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from dataclasses import dataclass
from loguru import logger
import timm

# Configure logging
logger.add("srt.log", rotation="500 MB")


@dataclass
class ModelConfig:
    """Configuration for the SRT model."""

    # Image processing
    image_size: Tuple[int, int] = (224, 224)
    num_channels: int = 3

    # Transformer architecture
    num_encoder_layers: int = 4
    num_decoder_layers: int = 7
    hidden_dim: int = 512
    num_heads: int = 8
    feedforward_dim: int = 3200
    dropout: float = 0.1

    # Training
    batch_size: int = 8
    learning_rate: float = 1e-5
    beta: float = 10.0
    chunk_size: int = 100


@dataclass
class RobotObservation:
    """Robot observation containing camera views."""

    stereo_left: torch.Tensor  # [C, H, W]
    stereo_right: torch.Tensor  # [C, H, W]
    wrist_left: torch.Tensor  # [C, H, W]
    wrist_right: torch.Tensor  # [C, H, W]

    def to_tensor(self) -> torch.Tensor:
        """Convert to batch tensor."""
        return torch.stack(
            [
                self.stereo_left,
                self.stereo_right,
                self.wrist_left,
                self.wrist_right,
            ]
        )


@dataclass
class RobotAction:
    """Robot action for both arms."""

    left_pos: torch.Tensor  # [3]
    left_rot: torch.Tensor  # [6]
    left_gripper: torch.Tensor  # [1]
    right_pos: torch.Tensor  # [3]
    right_rot: torch.Tensor  # [6]
    right_gripper: torch.Tensor  # [1]

    def to_tensor(self) -> torch.Tensor:
        """Convert to single tensor."""
        return torch.cat(
            [
                self.left_pos,
                self.left_rot,
                self.left_gripper,
                self.right_pos,
                self.right_rot,
                self.right_gripper,
            ]
        )

    @staticmethod
    def from_tensor(tensor: torch.Tensor) -> "RobotAction":
        """Create from flat tensor."""
        return RobotAction(
            left_pos=tensor[..., :3],
            left_rot=tensor[..., 3:9],
            left_gripper=tensor[..., 9:10],
            right_pos=tensor[..., 10:13],
            right_rot=tensor[..., 13:19],
            right_gripper=tensor[..., 19:20],
        )

class ImageEncoder(nn.Module):
    """Encodes multiple camera views into latent space using EVA Large."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Initialize EVA Large backbone
        self.backbone = timm.create_model(
            'eva_large_patch14_336',
            pretrained=True,
            num_classes=0,  # Remove classification head
            global_pool='avg',  # Use average pooling
        )
        
        # Freeze backbone parameters (optional, can be controlled via config)
        if getattr(config, 'freeze_backbone', True):
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Projection head to match transformer dimensions
        self.projection = nn.Sequential(
            nn.Linear(1024, config.hidden_dim * 2),  # EVA Large outputs 1024 features
            nn.LayerNorm(config.hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.Dropout(config.dropout)
        )
        
        # Initialize projection weights
        self._init_weights()
        
        # Image normalization parameters for EVA
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _init_weights(self):
        """Initialize the weights of the projection layers."""
        for m in self.projection.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def preprocess_images(self, images: torch.Tensor) -> torch.Tensor:
        """Normalize and preprocess images for EVA model."""
        # Ensure images are float and in range [0, 1]
        if images.dtype == torch.uint8:
            images = images.float() / 255.0
        
        # Normalize with ImageNet statistics
        images = (images - self.mean) / self.std
        
        # Resize if needed
        if images.shape[-2:] != (336, 336):  # EVA Large expected input size
            images = F.interpolate(
                images,
                size=(336, 336),
                mode='bilinear',
                align_corners=False
            )
        
        return images

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: [B, 4, C, H, W] tensor of camera views
                   B: batch size
                   4: number of views (stereo_left, stereo_right, wrist_left, wrist_right)
                   C: channels (3 for RGB)
                   H, W: height and width

        Returns:
            [B, 4, D] encoded features where D is config.hidden_dim
        """
        B = images.shape[0]
        features = []
        
        # Process each camera view separately
        for i in range(4):
            # Extract and preprocess single view
            view = images[:, i]  # [B, C, H, W]
            view = self.preprocess_images(view)
            
            # Extract features through EVA backbone
            with torch.cuda.amp.autocast(enabled=True):  # Enable AMP for efficiency
                x = self.backbone(view)  # [B, 1024]
            
            # Project to transformer dimension
            x = self.projection(x)  # [B, hidden_dim]
            features.append(x)

        # Stack all views
        features = torch.stack(features, dim=1)  # [B, 4, hidden_dim]
        
        return features

    def get_output_dim(self) -> int:
        """Return the output dimension of the encoder."""
        return self.config.hidden_dim

class TransformerBlock(nn.Module):
    """Standard transformer encoder/decoder block."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            config.hidden_dim,
            config.num_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(config.hidden_dim)
        self.norm2 = nn.LayerNorm(config.hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_dim, config.feedforward_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.feedforward_dim, config.hidden_dim),
            nn.Dropout(config.dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        memory: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, L, D] input sequence
            memory: Optional [B, M, D] memory sequence for cross-attention
            mask: Optional attention mask

        Returns:
            [B, L, D] processed sequence
        """
        # Self/cross attention
        q = k = x if memory is None else memory
        attn_out, _ = self.attention(x, q, k, attn_mask=mask)
        x = x + attn_out
        x = self.norm1(x)

        # MLP
        x = x + self.mlp(x)
        x = self.norm2(x)

        return x


class ActionPredictor(nn.Module):
    """Predicts robot actions from encoded states."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, 20),  # 2 * (3 + 6 + 1) = 20
        )

    def forward(self, x: torch.Tensor) -> RobotAction:
        """
        Args:
            x: [B, D] encoded state

        Returns:
            Predicted robot action
        """
        action = self.mlp(x)
        return RobotAction.from_tensor(action)


class SurgicalRobotTransformer(nn.Module):
    """Main SRT model implementing the full architecture."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        logger.info(f"Initializing SRT model with config: {config}")

        # Image encoder
        self.image_encoder = ImageEncoder(config)

        # Transformer encoder
        self.encoder_layers = nn.ModuleList(
            [
                TransformerBlock(config)
                for _ in range(config.num_encoder_layers)
            ]
        )

        # Transformer decoder
        self.decoder_layers = nn.ModuleList(
            [
                TransformerBlock(config)
                for _ in range(config.num_decoder_layers)
            ]
        )

        # Action predictor
        self.action_predictor = ActionPredictor(config)

        logger.info("Model initialized successfully")

    def encode_images(self, obs: RobotObservation) -> torch.Tensor:
        """Encode camera observations."""
        images = obs.to_tensor().unsqueeze(0)  # Add batch dim
        return self.image_encoder(images)

    def forward(
        self, obs: RobotObservation, chunk_size: Optional[int] = None
    ) -> RobotAction:
        """
        Forward pass predicting actions from observations.

        Args:
            obs: Robot observation containing camera views
            chunk_size: Optional chunk size for processing long sequences

        Returns:
            Predicted robot action
        """
        # Encode images
        encoded = self.encode_images(obs)
        B = encoded.shape[0]

        # Process with transformer encoder
        memory = encoded
        for layer in self.encoder_layers:
            memory = layer(memory)

        # Decode with transformer decoder
        x = memory.mean(dim=1)  # Pool sequence
        for layer in self.decoder_layers:
            x = layer(x.unsqueeze(1), memory=memory).squeeze(1)

        # Predict action
        action = self.action_predictor(x)

        return action

    def compute_loss(
        self, pred_action: RobotAction, target_action: RobotAction
    ) -> torch.Tensor:
        """Compute loss between predicted and target actions."""
        pred = pred_action.to_tensor()
        target = target_action.to_tensor()

        # MSE loss on positions and gripper
        pos_loss = F.mse_loss(
            torch.cat([pred[..., :3], pred[..., 10:13]], dim=-1),
            torch.cat([target[..., :3], target[..., 10:13]], dim=-1),
        )

        # Rotation loss using 6D representation
        rot_loss = F.mse_loss(
            torch.cat([pred[..., 3:9], pred[..., 13:19]], dim=-1),
            torch.cat([target[..., 3:9], target[..., 13:19]], dim=-1),
        )

        # Gripper loss
        grip_loss = F.mse_loss(
            torch.cat([pred[..., 9:10], pred[..., 19:20]], dim=-1),
            torch.cat(
                [target[..., 9:10], target[..., 19:20]], dim=-1
            ),
        )

        # Combine losses
        total_loss = (
            pos_loss + self.config.beta * rot_loss + grip_loss
        )

        return total_loss


def normalize_rotation_6d(
    rot_matrix_6d: torch.Tensor,
) -> torch.Tensor:
    """Convert 6D rotation to orthonormal matrix."""
    x = rot_matrix_6d[..., :3]
    y = rot_matrix_6d[..., 3:6]

    # Gram-Schmidt
    x = F.normalize(x, dim=-1)
    z = torch.cross(x, y, dim=-1)
    z = F.normalize(z, dim=-1)
    y = torch.cross(z, x, dim=-1)

    rotation = torch.stack([x, y, z], dim=-1)
    return rotation
