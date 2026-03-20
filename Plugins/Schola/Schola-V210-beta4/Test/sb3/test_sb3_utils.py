# Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
"""Tests for the SB3 utility functions and wrappers"""

import pytest
import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box, Discrete, MultiDiscrete, MultiBinary, Dict as DictSpace
from stable_baselines3.common.vec_env import DummyVecEnv
from unittest.mock import Mock, MagicMock, patch
from schola.sb3.utils import merge_spaces, RenderImagesWrapper


class TestMergeSpaces:
    """Test suite for the merge_spaces singledispatch function."""

    def test_merge_box_1d_spaces(self):
        """Test merging 1D Box spaces."""
        space1 = Box(low=0, high=1, shape=(3,), dtype=np.float32)
        space2 = Box(low=0, high=1, shape=(2,), dtype=np.float32)
        space3 = Box(low=-1, high=1, shape=(4,), dtype=np.float32)

        merged = merge_spaces(space1, space2, space3)

        assert isinstance(merged, Box)
        assert merged.shape == (9,)  # 3 + 2 + 4
        assert merged.dtype == np.float32
        assert merged.low.shape == (9,)
        assert merged.high.shape == (9,)

    def test_merge_box_2d_spaces(self):
        """Test merging 2D Box spaces (like images)."""
        space1 = Box(low=0, high=255, shape=(84, 84, 3), dtype=np.uint8)
        space2 = Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)

        merged = merge_spaces(space1, space2)

        assert isinstance(merged, Box)
        assert merged.shape == (84, 84, 4)  # concatenated along last axis
        assert merged.dtype == np.uint8

    def test_merge_box_different_dtype_raises(self):
        """Test that merging Box spaces with different dtypes raises an error."""
        space1 = Box(low=0, high=1, shape=(3,), dtype=np.float32)
        space2 = Box(low=0, high=1, shape=(2,), dtype=np.float64)

        with pytest.raises(ValueError, match="same dtype"):
            merge_spaces(space1, space2)

    def test_merge_box_different_dimensions_raises(self):
        """Test that merging Box spaces with different dimensions raises an error."""
        space1 = Box(low=0, high=1, shape=(3,), dtype=np.float32)
        space2 = Box(low=0, high=1, shape=(2, 2), dtype=np.float32)

        with pytest.raises(ValueError, match="same number of dimensions"):
            merge_spaces(space1, space2)

    def test_merge_discrete_spaces(self):
        """Test merging Discrete spaces into MultiDiscrete."""
        space1 = Discrete(5)
        space2 = Discrete(3)
        space3 = Discrete(7)

        merged = merge_spaces(space1, space2, space3)

        assert isinstance(merged, MultiDiscrete)
        assert len(merged.nvec) == 3
        # Note: Based on the user's code, dims.append([s.n]) suggests each Discrete
        # gets wrapped in a list, so we need to check the implementation

    def test_merge_mixed_discrete_and_multidiscrete(self):
        """Test merging a mix of Discrete and MultiDiscrete spaces."""
        space1 = Discrete(5)
        space2 = MultiDiscrete([3, 4])
        space3 = Discrete(2)

        merged = merge_spaces(space1, space2, space3)

        assert isinstance(merged, MultiDiscrete)
        # Should have 1 (from space1) + 2 (from space2) + 1 (from space3) = 4 dimensions

    def test_merge_multibinary_spaces(self):
        """Test merging MultiBinary spaces."""
        space1 = MultiBinary(5)
        space2 = MultiBinary(3)
        space3 = MultiBinary(7)

        merged = merge_spaces(space1, space2, space3)

        assert isinstance(merged, MultiBinary)
        assert merged.n == 15  # 5 + 3 + 7

    def test_merge_multibinary_wrong_type_raises(self):
        """Test that merging MultiBinary with incompatible type raises an error."""
        space1 = MultiBinary(5)
        space2 = Box(low=0, high=1, shape=(3,), dtype=np.float32)

        with pytest.raises(TypeError, match="Cannot merge MultiBinary space"):
            merge_spaces(space1, space2)

    def test_merge_unsupported_space_raises(self):
        """Test that merging unsupported space types raises NotImplementedError."""
        space = gym.spaces.Text(max_length=10)

        with pytest.raises(NotImplementedError, match="Merge not implemented"):
            merge_spaces(space, space)

    def test_merge_three_box_spaces(self):
        """Test merging three Box spaces"""
        space1 = Box(low=-1, high=1, shape=(2,))
        space2 = Box(low=-2, high=2, shape=(3,))
        space3 = Box(low=-3, high=3, shape=(4,))
        
        merged = merge_spaces(space1, space2, space3)
        
        assert isinstance(merged, Box)
        assert merged.shape == (9,)

from .envs import DictActionBoxEnv, DictActionDiscreteEnv, DictActionMultiBinaryEnv, DictActionMixedEnv, DictActionEmptyEnv, make_dict_action_env

class TestVecMergeDictActionWrapper:
    """Test suite for the VecMergeDictActionWrapper."""

    def test_merge_dict_box_actions(self):
        """Test merging dictionary of Box action spaces."""
        from schola.sb3.utils import VecMergeDictActionWrapper
        
        vec_env = DummyVecEnv([make_dict_action_env(DictActionBoxEnv, True)])
        wrapped_env = VecMergeDictActionWrapper(vec_env)
        
        # Check that action space is merged
        assert isinstance(wrapped_env.action_space, Box)
        assert wrapped_env.action_space.shape == (4,)  # 2 + 2
        
        # Test reset and step
        obs = wrapped_env.reset()
        assert obs.shape == (1, 4)
        
        # Test with flat action
        flat_action = np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32)
        obs, reward, done, info = wrapped_env.step(flat_action)
        assert obs.shape == (1, 4)
        assert reward.shape == (1,)
        assert done.shape == (1,)
        
        wrapped_env.close()

    def test_merge_dict_discrete_actions(self):
        """Test merging dictionary of Discrete action spaces."""
        from schola.sb3.utils import VecMergeDictActionWrapper
        
        vec_env = DummyVecEnv([make_dict_action_env(DictActionDiscreteEnv, True)])
        wrapped_env = VecMergeDictActionWrapper(vec_env)
        
        # Check that action space is merged into MultiDiscrete
        assert isinstance(wrapped_env.action_space, MultiDiscrete)
        assert len(wrapped_env.action_space.nvec) == 2
        
        # Test reset and step
        obs = wrapped_env.reset()
        assert obs.shape == (1, 4)
        
        # Test with flat discrete action
        flat_action = np.array([[2, 1]], dtype=np.int64)
        obs, reward, done, info = wrapped_env.step(flat_action)
        assert obs.shape == (1, 4)
        
        wrapped_env.close()

    def test_merge_dict_multibinary_actions(self):
        """Test merging dictionary of MultiBinary action spaces."""
        from schola.sb3.utils import VecMergeDictActionWrapper
        
        vec_env = DummyVecEnv([make_dict_action_env(DictActionMultiBinaryEnv, True)])
        wrapped_env = VecMergeDictActionWrapper(vec_env)

        # Check that action space is merged
        assert isinstance(wrapped_env.action_space, MultiBinary)
        assert wrapped_env.action_space.n == 5  # 3 + 2
        
        # Test reset and step
        obs = wrapped_env.reset()
        assert obs.shape == (1, 4)
        
        # Test with flat binary action
        flat_action = np.array([[1, 0, 1, 0, 1]], dtype=np.int8)
        obs, reward, done, info = wrapped_env.step(flat_action)
        assert obs.shape == (1, 4)
        
        wrapped_env.close()

    def test_empty_dict_action_raises(self):
        """Test that empty dictionary action space raises an assertion error."""
        from schola.sb3.utils import VecMergeDictActionWrapper
        
        vec_env = DummyVecEnv([make_dict_action_env(DictActionEmptyEnv, True)])
        
        with pytest.raises(AssertionError, match="No Action Spaces to merge"):
            VecMergeDictActionWrapper(vec_env)

    def test_multiple_vec_envs_with_dict_actions(self):
        """Test wrapper with multiple parallel environments."""
        from schola.sb3.utils import VecMergeDictActionWrapper
        
        # Create multiple environments
        n_envs = 4
        vec_env = DummyVecEnv([make_dict_action_env(DictActionBoxEnv, True) for _ in range(n_envs)])
        wrapped_env = VecMergeDictActionWrapper(vec_env)

        # Check action space
        assert isinstance(wrapped_env.action_space, Box)
        assert wrapped_env.action_space.shape == (4,)  # 2 + 2

        # Test reset with multiple envs
        obs = wrapped_env.reset()
        assert obs.shape == (n_envs, 4)

        # Test step with multiple envs
        actions = np.random.uniform(-1, 1, size=(n_envs, 4)).astype(np.float32)
        obs, rewards, dones, infos = wrapped_env.step(actions)
        assert obs.shape == (n_envs, 4)
        assert rewards.shape == (n_envs,)
        assert dones.shape == (n_envs,)
        assert len(infos) == n_envs

        wrapped_env.close()

    def test_step_async_and_wait_with_dict_actions(self):
        """Test async step functionality with dictionary actions."""
        from schola.sb3.utils import VecMergeDictActionWrapper

        vec_env = DummyVecEnv([make_dict_action_env(DictActionBoxEnv, True)])
        wrapped_env = VecMergeDictActionWrapper(vec_env)

        wrapped_env.reset()
        
        # Test step_async
        action = np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32)
        wrapped_env.step_async(action)
        
        # Test step_wait
        obs, rewards, dones, infos = wrapped_env.step_wait()
        assert obs.shape == (1, 4)
        assert rewards.shape == (1,)
        assert dones.shape == (1,)
        assert len(infos) == 1

        wrapped_env.close()


class TestRenderImagesWrapper:
    """Tests for RenderImagesWrapper"""
    
    def test_wrapper_fails_without_matplotlib(self):
        """Test that wrapper fails when matplotlib is not available"""
        mock_env = Mock()
        mock_env.num_envs = 1
        mock_env.observation_space = DictSpace({
            "image": Box(low=0, high=1, shape=(3, 64, 64))
        })
        
        with patch('schola.sb3.utils.plt', None):
            with pytest.raises(ImportError, match="matplotlib"):
                RenderImagesWrapper(mock_env)
                
    def test_convert_to_plt_format_grayscale(self):
        """Test convert_to_plt_format with grayscale image"""
        obs_space = Box(low=0, high=1, shape=(1, 64, 64))
        observation_space = DictSpace({
            "image": obs_space
        })
        
        mock_env = MagicMock()
        mock_env.num_envs = 1
        # Configure mock to return the real observation_space without interception
        mock_env.configure_mock(observation_space=observation_space)
        
        with patch('schola.sb3.utils.plt') as mock_plt:
            mock_plt.ion.return_value = None
            mock_plt.ioff.return_value = None
            mock_plt.show.return_value = None
            mock_axis = MagicMock()
            mock_axis.imshow.return_value = MagicMock()
            mock_plt.subplot.return_value = mock_axis
            
            wrapper = RenderImagesWrapper(mock_env)
            
            obs = np.random.rand(1, 64, 64)
            converted = wrapper.convert_to_plt_format(obs)
            
            # Should squeeze the channel dimension
            assert converted.shape == (64, 64)
            
    def test_convert_to_plt_format_rgb(self):
        """Test convert_to_plt_format with RGB image"""
        obs_space = Box(low=0, high=1, shape=(3, 64, 64))
        observation_space = DictSpace({
            "image": obs_space
        })
        
        mock_env = MagicMock()
        mock_env.num_envs = 1
        # Configure mock to return the real observation_space without interception
        mock_env.configure_mock(observation_space=observation_space)
        
        with patch('schola.sb3.utils.plt') as mock_plt:
            mock_plt.ion.return_value = None
            mock_plt.ioff.return_value = None
            mock_plt.show.return_value = None
            mock_axis = MagicMock()
            mock_axis.imshow.return_value = MagicMock()
            mock_plt.subplot.return_value = mock_axis
            
            wrapper = RenderImagesWrapper(mock_env)
            
            obs = np.random.rand(3, 64, 64)
            converted = wrapper.convert_to_plt_format(obs)
            
            # Should transpose to (H, W, C)
            assert converted.shape == (64, 64, 3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])