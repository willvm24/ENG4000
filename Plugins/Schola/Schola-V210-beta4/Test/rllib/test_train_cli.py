# Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
"""Tests for the rllib cli"""


from copy import deepcopy
import logging
from cyclopts import App
import pytest
from pathlib import Path
from schola.scripts.common.settings import UnrealExecutableSimulatorArgs
from schola.scripts.rllib.train import app as train_app, RLlibTrainCommand
from schola.scripts.rllib.settings import (
    APPOSettings,
    RLlibScriptArgs,
    PPOSettings,
    SACSettings,
    IMPALASettings,
    TrainingSettings,
)
from schola.scripts.common.settings import (
    UnrealEditorSimulatorArgs,
    UnrealProjectSimulatorArgs,
    ActivationFunctionEnum,
)


@pytest.fixture
def mock_main(mocker):
    """Mock the main training function to prevent actual training."""
    return mocker.patch("schola.scripts.rllib.train.main")


@pytest.fixture
def mock_app(mock_main):
    """Build a fresh app with mocked main (no global injection)."""
    app = App(name="train", help="Train a Model using ray")
    logger = logging.getLogger(__name__)
    app = RLlibTrainCommand(app, RLlibScriptArgs, mock_main, logger).make()
    return app


def test_ppo_default_arguments(mock_app, mock_main):
    """Test PPO command with default arguments."""
    mock_app.meta(["ppo"], result_action="return_value")
    
    # Verify main was called once
    mock_main.assert_called_once()
    
    # Extract the arguments passed to main
    args: RLlibScriptArgs = mock_main.call_args[0][0]
    
    # Verify it's the correct dataclass
    assert isinstance(args, RLlibScriptArgs)
    
    # Verify algorithm settings
    assert isinstance(args.algorithm_settings, PPOSettings)
    assert args.algorithm_settings.gae_lambda == 0.95
    assert args.algorithm_settings.clip_param == 0.2
    assert args.algorithm_settings.use_gae is True
    
    # Verify default training settings
    assert isinstance(args.training_settings, TrainingSettings)
    assert args.training_settings.timesteps == 3000
    assert args.training_settings.learning_rate == 0.0003
    assert args.training_settings.gamma == 0.99
    
    # Verify default simulator is editor and num_simulators is fixed at 1
    assert isinstance(args.environment_settings.simulator, UnrealEditorSimulatorArgs)
    assert args.environment_settings.simulator.num_simulators == 1


def test_ppo_custom_training_parameters(mock_app, mock_main):
    """Test PPO command with custom training parameters."""
    mock_app.meta([
        "ppo",
        "--training-settings.timesteps", "10000",
        "--training-settings.learning-rate", "0.001",
        "--training-settings.gamma", "0.95",
        "--training-settings.minibatch-size", "64",
        "--training-settings.train-batch-size-per-learner", "256",
        "--training-settings.num-sgd-iter", "10"
    ], result_action="return_value")
    
    mock_main.assert_called_once()
    args: RLlibScriptArgs = mock_main.call_args[0][0]
    
    # Verify custom training settings
    assert args.training_settings.timesteps == 10000
    assert args.training_settings.learning_rate == 0.001
    assert args.training_settings.gamma == 0.95
    assert args.training_settings.minibatch_size == 64
    assert args.training_settings.train_batch_size_per_learner == 256
    assert args.training_settings.num_epochs == 10


def test_ppo_custom_algorithm_parameters(mock_app, mock_main):
    """Test PPO command with custom PPO-specific parameters."""
    mock_app.meta([
        "ppo",
        "--gae-lambda", "0.90",
        "--clip-param", "0.3",
        "--no-use-gae"
    ], result_action="return_value")
    
    mock_main.assert_called_once()
    args: RLlibScriptArgs = mock_main.call_args[0][0]
    
    # Verify custom PPO settings
    assert isinstance(args.algorithm_settings, PPOSettings)
    assert args.algorithm_settings.gae_lambda == 0.90
    assert args.algorithm_settings.clip_param == 0.3
    assert args.algorithm_settings.use_gae is False


def test_sac_default_arguments(mock_app, mock_main):
    """Test SAC command with default arguments."""
    mock_app.meta(["sac"], result_action="return_value")
    
    mock_main.assert_called_once()
    args: RLlibScriptArgs = mock_main.call_args[0][0]
    
    # Verify it's using SAC settings
    assert isinstance(args.algorithm_settings, SACSettings)
    assert args.algorithm_settings.tau == 0.005
    assert args.algorithm_settings.target_entropy == "auto"
    assert args.algorithm_settings.initial_alpha == 1.0
    assert args.algorithm_settings.n_step == 1
    assert args.algorithm_settings.twin_q is True


def test_sac_custom_parameters(mock_app, mock_main):
    """Test SAC command with custom SAC-specific parameters."""
    mock_app.meta([
        "sac",
        "--tau", "0.01",
        "--initial-alpha", "0.5",
        "--n-step", "3",
        "--no-twin-q"
    ], result_action="return_value")
    
    mock_main.assert_called_once()
    args: RLlibScriptArgs = mock_main.call_args[0][0]
    
    # Verify custom SAC settings
    assert isinstance(args.algorithm_settings, SACSettings)
    assert args.algorithm_settings.tau == 0.01
    assert args.algorithm_settings.initial_alpha == 0.5
    assert args.algorithm_settings.n_step == 3
    assert args.algorithm_settings.twin_q is False


def test_impala_default_arguments(mock_app, mock_main):
    """Test IMPALA command with default arguments."""
    mock_app.meta(["impala"], result_action="return_value")
    
    mock_main.assert_called_once()
    args: RLlibScriptArgs = mock_main.call_args[0][0]
    
    # Verify it's using IMPALA settings
    assert isinstance(args.algorithm_settings, IMPALASettings)
    assert args.algorithm_settings.vtrace is True
    assert args.algorithm_settings.vtrace_clip_rho_threshold == 1.0
    assert args.algorithm_settings.vtrace_clip_pg_rho_threshold == 1.0


def test_impala_custom_parameters(mock_app, mock_main):
    """Test IMPALA command with custom IMPALA-specific parameters."""
    mock_app.meta([
        "impala",
        "--no-vtrace",
        "--vtrace-clip-rho-threshold", "2.0",
        "--vtrace-clip-pg-rho-threshold", "1.5"
    ], result_action="return_value")
    
    mock_main.assert_called_once()
    args: RLlibScriptArgs = mock_main.call_args[0][0]
    
    # Verify custom IMPALA settings
    assert isinstance(args.algorithm_settings, IMPALASettings)
    assert args.algorithm_settings.vtrace is False
    assert args.algorithm_settings.vtrace_clip_rho_threshold == 2.0
    assert args.algorithm_settings.vtrace_clip_pg_rho_threshold == 1.5


def test_resource_settings(mock_app, mock_main):
    """Test resource allocation parameters."""
    mock_app.meta([
        "ppo",
        "--resource-settings.num-gpus", "2",
        "--resource-settings.num-cpus", "8",
        "--resource-settings.num-learners", "4",
        "--resource-settings.num-cpus-per-learner", "2",
        "--resource-settings.num-gpus-per-learner", "1"
    ], result_action="return_value")
    
    mock_main.assert_called_once()
    args: RLlibScriptArgs = mock_main.call_args[0][0]
    
    # Verify resource settings
    assert args.resource_settings.num_gpus == 2
    assert args.resource_settings.num_cpus == 8
    assert args.resource_settings.num_learners == 4
    assert args.resource_settings.num_cpus_per_learner == 2
    assert args.resource_settings.num_gpus_per_learner == 1


def test_network_architecture_settings(mock_app, mock_main):
    """Test network architecture parameters."""
    mock_app.meta([
        "ppo",
        "--network-architecture-settings.activation", "TanH",
        "--network-architecture-settings.use-lstm",
        "--network-architecture-settings.lstm-cell-size", "128",
        "--network-architecture-settings.max-seq-len", "10",
        "--network-architecture-settings.fcnet-hiddens", "256", "256"
    ], result_action="return_value")
    
    mock_main.assert_called_once()
    args: RLlibScriptArgs = mock_main.call_args[0][0]
    
    # Verify network architecture settings
    assert args.network_architecture_settings.activation == ActivationFunctionEnum.TanH, "Activation should be TanH"
    assert args.network_architecture_settings.use_lstm is True, "Use LSTM should be True"
    assert args.network_architecture_settings.lstm_cell_size == 128, "LSTM cell size should be 128"
    assert args.network_architecture_settings.max_seq_len == 10, "max_seq_len should be 10"
    assert args.network_architecture_settings.fcnet_hiddens == [256, 256], "FCNet hiddens should be [256, 256]"


def test_logging_settings(mock_app, mock_main):
    """Test logging verbosity parameters."""
    mock_app.meta([
        "ppo",
        "--logging-settings.schola-verbosity", "2",
        "--logging-settings.rllib-verbosity", "3"
    ], result_action="return_value")
    
    mock_main.assert_called_once()
    args: RLlibScriptArgs = mock_main.call_args[0][0]
    
    # Verify logging settings
    assert args.logging_settings.schola_verbosity == 2
    assert args.logging_settings.rllib_verbosity == 3


def test_checkpoint_settings(mock_app, mock_main, tmp_path):
    """Test checkpoint configuration parameters."""
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    mock_app.meta([
        "ppo",
        "--checkpoint-settings.checkpoint-dir", str(checkpoint_dir),
        "--checkpoint-settings.save-freq", "1000",
        "--checkpoint-settings.enable-checkpoints",
        "--checkpoint-settings.save-final-policy",
        "--checkpoint-settings.export-onnx"
    ], result_action="return_value")
    
    mock_main.assert_called_once()
    args: RLlibScriptArgs = mock_main.call_args[0][0]
    
    # Verify checkpoint settings
    assert args.checkpoint_settings.checkpoint_dir == checkpoint_dir
    assert args.checkpoint_settings.save_freq == 1000
    assert args.checkpoint_settings.enable_checkpoints is True
    assert args.checkpoint_settings.save_final_policy is True
    assert args.checkpoint_settings.export_onnx is True


def test_ppo_with_executable_simulator(mock_app, mock_main, tmp_path):
    """Test executable simulator type is correctly parsed."""
    executable_path = tmp_path / "UnrealGame.exe"
    executable_path.touch()  # Create fake executable
    mock_app.meta([
        "ppo",
        'executable',
        "--executable-path", str(executable_path),
        "--headless"
    ], result_action="return_value")
    
    mock_main.assert_called_once()
    args: RLlibScriptArgs = mock_main.call_args[0][0]
    
    assert isinstance(args.environment_settings.simulator, UnrealExecutableSimulatorArgs)
    assert args.environment_settings.simulator.executable_path == executable_path


def test_executable_num_simulators_parsed(mock_app, mock_main, tmp_path):
    """Test num_simulators is parsed for executable simulator."""
    executable_path = tmp_path / "UnrealGame.exe"
    executable_path.touch()
    mock_app.meta([
        "ppo",
        "executable",
        "--executable-path", str(executable_path),
        "--num-simulators", "4",
    ], result_action="return_value")
    mock_main.assert_called_once()
    args: RLlibScriptArgs = mock_main.call_args[0][0]
    assert isinstance(args.environment_settings.simulator, UnrealExecutableSimulatorArgs)
    assert args.environment_settings.simulator.num_simulators == 4


def test_project_num_simulators_parsed(mock_app, mock_main, tmp_path):
    """Test num_simulators is parsed for project simulator."""
    uproject = tmp_path / "MyGame.uproject"
    uproject.write_text("{}")
    mock_app.meta([
        "ppo",
        "project",
        str(uproject),
        "--num-simulators", "2",
    ], result_action="return_value")
    mock_main.assert_called_once()
    args: RLlibScriptArgs = mock_main.call_args[0][0]
    assert isinstance(args.environment_settings.simulator, UnrealProjectSimulatorArgs)
    assert args.environment_settings.simulator.num_simulators == 2
    

def test_protocol_settings(mock_app, mock_main):
    """Test protocol configuration parameters."""
    mock_app.meta([
        "ppo",
        "--protocol.port", "12345"
    ], result_action="return_value")
    
    mock_main.assert_called_once()
    args: RLlibScriptArgs = mock_main.call_args[0][0]
    
    # Verify protocol settings
    assert args.environment_settings.protocol.port == 12345


def test_multiple_algorithms_return_different_settings(mock_app, mock_main):
    """Test that different algorithm commands create different settings."""
    mock_app.meta(["ppo"], result_action="return_value")
    ppo_args: RLlibScriptArgs = deepcopy(mock_main.call_args[0][0])
    
    mock_main.reset_mock()
    mock_app.meta(["sac"], result_action="return_value")
    sac_args: RLlibScriptArgs = deepcopy(mock_main.call_args[0][0])
    
    mock_main.reset_mock()
    mock_app.meta(["impala"], result_action="return_value")
    impala_args: RLlibScriptArgs = deepcopy(mock_main.call_args[0][0])

    mock_main.reset_mock()
    mock_app.meta(["appo"], result_action="return_value")
    appo_args: RLlibScriptArgs = deepcopy(mock_main.call_args[0][0])
    
    # Verify different algorithm types are set correctly
    assert isinstance(ppo_args.algorithm_settings, PPOSettings)
    assert isinstance(sac_args.algorithm_settings, SACSettings)
    assert isinstance(impala_args.algorithm_settings, IMPALASettings)
    assert isinstance(appo_args.algorithm_settings, APPOSettings)


def test_complex_configuration(mock_app, mock_main, tmp_path):
    """Test a complex configuration with many parameters."""
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    mock_app.meta([
        "ppo",
        # Training settings
        "--training-settings.timesteps", "50000",
        "--training-settings.learning-rate", "0.0005",
        "--training-settings.gamma", "0.98",
        "--training-settings.minibatch-size", "32",
        "--training-settings.train-batch-size-per-learner", "128",
        "--training-settings.num-sgd-iter", "8",
        # PPO settings
        "--gae-lambda", "0.92",
        "--clip-param", "0.25",
        # Resource settings
        "--resource-settings.num-gpus", "1",
        "--resource-settings.num-cpus", "4",
        # Network architecture
        "--network-architecture-settings.activation", "ReLU",
        # Logging
        "--logging-settings.schola-verbosity", "1",
        "--logging-settings.rllib-verbosity", "2",
        # Checkpoint settings
        "--checkpoint-settings.checkpoint-dir", str(checkpoint_dir),
        "--checkpoint-settings.save-freq", "5000",
        "--checkpoint-settings.enable-checkpoints",
        # Protocol settings
        "--protocol.port", "50051"
    ], result_action="return_value")
    
    mock_main.assert_called_once()
    args: RLlibScriptArgs = mock_main.call_args[0][0]
    
    # Verify all settings were applied correctly
    assert args.training_settings.timesteps == 50000
    assert args.training_settings.learning_rate == 0.0005
    assert args.training_settings.gamma == 0.98
    assert args.algorithm_settings.gae_lambda == 0.92 # type: ignore
    assert args.algorithm_settings.clip_param == 0.25 # type: ignore
    assert args.resource_settings.num_gpus == 1
    assert args.resource_settings.num_cpus == 4
    assert args.network_architecture_settings.activation == ActivationFunctionEnum.ReLU
    assert args.logging_settings.schola_verbosity == 1
    assert args.logging_settings.rllib_verbosity == 2
    assert args.checkpoint_settings.save_freq == 5000
    assert args.environment_settings.protocol.port == 50051


@pytest.mark.xdist_group(name="ray-cluster")
def test_train_cli_with_unreal_editor(make_vec_env_server, make_env, ray_cluster, tmp_path):
    checkpoint_dir = tmp_path / "ckpt"
    checkpoint_dir.mkdir()
    env_server_port = make_vec_env_server([make_env("CartPole-v1", i) for i in range(2)])
    
    train_app.meta([
        "ppo",
        # Training settings
        "--training-settings.timesteps", "5000",
        "--training-settings.learning-rate", "0.0005",
        "--training-settings.gamma", "0.98",
        "--training-settings.minibatch-size", "32",
        "--training-settings.train-batch-size-per-learner", "128",
        "--training-settings.num-epochs", "8",
        # PPO settings
        "--gae-lambda", "0.92",
        "--clip-param", "0.25",
        # Resource settings
        "--resource-settings.using-cluster",
        # Network architecture
        "--network-architecture-settings.activation", "ReLU",
        # Logging
        "--logging-settings.schola-verbosity", "2",
        "--logging-settings.rllib-verbosity", "2",
        # Checkpoint settings
        "--checkpoint-settings.checkpoint-dir", str(checkpoint_dir),
        "--checkpoint-settings.no-enable-checkpoints",
        "--checkpoint-settings.no-export-onnx",
        # Protocol settings
        "--protocol.port", f"{env_server_port}",
    ], result_action="return_value")