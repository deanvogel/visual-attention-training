import torch
import gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torchvision.transforms import Resize
import numpy as np
import cv2
from gym import ObservationWrapper

class ResizeObservation(ObservationWrapper):
    r"""Downsample the image observation to a square image."""

    def __init__(self, env, shape):
        super().__init__(env)
        if isinstance(shape, int):
            shape = (shape, shape)
        assert all(x > 0 for x in shape), shape

        self.shape = tuple(shape)

        obs_shape = self.shape + self.observation_space.shape[2:]
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        observation = cv2.resize(
            observation, self.shape[::-1], interpolation=cv2.INTER_AREA
        )
        if observation.ndim == 2:
            observation = np.expand_dims(observation, -1)
        return observation

class VANFeatureExtractionWrapper(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim=256, model_f=None, img_size=224, **kwargs):
        super(VANFeatureExtractionWrapper, self).__init__(observation_space, features_dim)

        in_chans = observation_space.shape[0]

        # self.transforms = Resize((img_size, img_size))
        self.model = model_f(img_size=img_size, in_chans=in_chans, num_classes=features_dim, **kwargs)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # observations = self.transforms(observations)
        return self.model(observations)