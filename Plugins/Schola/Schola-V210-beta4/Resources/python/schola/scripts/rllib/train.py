# Copyright (c) 2024-2025 Advanced Micro Devices, Inc. All Rights Reserved.

"""
Script to train an rllib model using Schola.
"""
import logging

from pathlib import Path
from typing import Annotated, Any, Callable, Dict, Literal, Optional, Type, Union, cast
import traceback


from schola.scripts.common.settings import (
    UnrealEditorSimulatorArgs,
    UnrealExecutableSimulatorArgs,
    UnrealProjectSimulatorArgs,
    get_activation_function,
)
from schola.scripts.common.train_template import MetaTrainCommand

from schola.scripts.rllib.settings import (
    APPOSettings,
    RLlibScriptArgs,
    PPOSettings,
    SACSettings,
    IMPALASettings,
)

from cyclopts import App, Parameter, Group

# Logging setup
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
logger = logging.getLogger(__name__)


app = App(name="train", help="Train a Model using ray")


# forward declare here for type hinting with no load
def main(args: RLlibScriptArgs) -> "ray.tune.ExperimentAnalysis":
    """
    Main function for launching training with ray.

    Parameters
    ----------
    args : RLlibArgs
        The arguments for the script as a dataclass

    Returns
    -------
    tune.ExperimentAnalysis
        The results of the training
    """
    # Import ray and rllib dependencies lazily when the command is actually executed
    import ray
    from ray import air, tune
    from ray.rllib.algorithms.algorithm import Algorithm
    from ray.rllib.policy.policy import PolicySpec
    from ray.rllib.algorithms.ppo import PPOConfig
    from ray.rllib.algorithms.sac.sac import SACConfig
    from ray.rllib.algorithms.appo.appo import APPOConfig
    from ray.rllib.algorithms.impala.impala import IMPALAConfig
    from ray.tune.registry import register_env
    from schola.rllib.export import export_onnx_from_policy
    from ray.rllib.policy.policy import Policy
    from ray.rllib.core.rl_module.rl_module import RLModuleSpec, RLModule
    from ray.rllib.connectors.env_to_module import FlattenObservations
    from schola.rllib.env import RayVecEnv
    from schola.rllib.env_runner import ScholaEnvRunner
    from schola.core.protocols.protobuf.gRPC import gRPCProtocol
    from schola.core.simulators.unreal.editor import UnrealEditor
    from schola.core.simulators.unreal.executable import UnrealExecutable
    from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
    from ray.rllib.connectors.connector_v2 import ConnectorV2

    sim_args = args.environment_settings.simulator
    protocol_args = args.environment_settings.protocol
    n_sim = sim_args.num_simulators
    # Run locally if we are only running one simulator
    num_env_runners = 0 if n_sim == 1 else n_sim
    # Collect agent display names using a temporary environment instance (builds project if needed)
    primary_sim = sim_args.make()
    tmp_env = RayVecEnv(
        protocol_args.make(),
        primary_sim,
        verbosity=args.logging_settings.schola_verbosity,
    )
    # Note: We no longer wrap with DictToBoxWrapper - flattening is handled by ConnectorV2
    try:
        # Get agent IDs from the first environment
        agent_names = {agent_id: agent_id for agent_id in tmp_env.possible_agents}
    finally:
        tmp_env.close()

    # Clusters configure resources automatically
    if not args.resource_settings.using_cluster:
        ray.init(
            num_cpus=args.resource_settings.num_cpus,
            num_gpus=args.resource_settings.num_gpus,
        )
    else:
        if args.resource_settings.num_cpus > 1:
            logger.warning(
                "--resource-settings.num-cpus is a non-default value, but the script is connecting to an existing cluster. This parameter will be ignored."
            )
        if args.resource_settings.num_gpus > 0:
            logger.warning(
                "--resource-settings.num-gpus is a non-default value, but the script is connecting to an existing cluster. This parameter will be ignored."
            )

    # Get activation function for model config
    activation_fn = get_activation_function(
        args.network_architecture_settings.activation
    )

    # make a new variable to get typing information
    algorithm_config: AlgorithmConfig = args.algorithm_settings.rllib_config()

    # Use NEW API stack with RayEnv/RayVecEnv (new stack interface)
    # Auto-assignment: RayEnv for local runner (num_env_runners=0), RayVecEnv for remote runners
    config: Union[PPOConfig, SACConfig, APPOConfig, IMPALAConfig] = (
        algorithm_config.api_stack(
            enable_rl_module_and_learner=True,  # Enable new stack
            enable_env_runner_and_connector_v2=True,  # Enable EnvRunner
        )
        .environment(
            env_config={
                "protocol": gRPCProtocol,
                "protocol_args": {
                    "url": args.environment_settings.protocol.url,
                    "port": args.environment_settings.protocol.port,
                },
                "simulator": (
                    UnrealEditor
                    if isinstance(primary_sim, UnrealEditor)
                    else UnrealExecutable
                ),
                "simulator_args": (
                    {}
                    if isinstance(primary_sim, UnrealEditor)
                    else primary_sim.get_executable_args()
                ),
            },
        )
        .framework("torch")
        .env_runners(
            env_runner_cls=ScholaEnvRunner,
            num_env_runners=num_env_runners,
            # Use ConnectorV2 to flatten Dict observations to Box for neural network processing
            env_to_module_connector=lambda env, spaces=None, device=None: FlattenObservations(
                input_observation_space=env.single_observation_space,
                input_action_space=env.single_action_space,
                multi_agent=True,
            ),
        )
        .multi_agent(
            policies={agent_id: PolicySpec() for agent_id in agent_names.keys()},
            policy_mapping_fn=lambda agent_id, *args, **kwargs: agent_id,  # type: ignore
        )
        .resources(
            num_gpus=args.resource_settings.num_gpus,
        )
        .learners(
            # When num_cpus=1, use 0 learners (local learning on main process)
            # This avoids resource conflicts with env_runner
            num_learners=(
                args.resource_settings.num_learners
                if args.resource_settings.num_learners > 0
                or args.resource_settings.num_cpus > 1
                else 0
            ),
            num_gpus_per_learner=args.resource_settings.num_gpus_per_learner,
            num_cpus_per_learner=args.resource_settings.num_cpus_per_learner,
        )
        .rl_module(
            model_config={
                "fcnet_hiddens": args.network_architecture_settings.fcnet_hiddens,
                "fcnet_activation": activation_fn,
                "use_lstm": args.network_architecture_settings.use_lstm,
                "lstm_cell_size": args.network_architecture_settings.lstm_cell_size,
                "max_seq_len": args.network_architecture_settings.max_seq_len,
            },
        )
        .training(
            lr=args.training_settings.learning_rate,
            gamma=args.training_settings.gamma,
            num_epochs=args.training_settings.num_epochs,
            train_batch_size=args.training_settings.train_batch_size_per_learner,
            minibatch_size=args.training_settings.minibatch_size,
            **args.algorithm_settings.get_settings_dict()
        )
        .debugging(
            log_level=args.logging_settings.rllib_log_level,
        )
    )  # type: ignore

    # Use the new API stack metric name for stopping criterion
    # Old stack used "timesteps_total", new stack uses "num_env_steps_sampled_lifetime"
    stop = {
        "num_env_steps_sampled_lifetime": args.training_settings.timesteps,
    }

    callbacks = []

    logger.info("Starting training")
    try:
        results = tune.run(
            args.algorithm_settings.name,
            config=config,
            stop=stop,
            checkpoint_config=air.CheckpointConfig(
                checkpoint_frequency=(
                    args.checkpoint_settings.save_freq
                    if args.checkpoint_settings.enable_checkpoints
                    else 0
                ),
                checkpoint_at_end=args.checkpoint_settings.save_final_policy,
            ),
            restore=(
                str(args.resume_settings.resume_from)
                if args.resume_settings.resume_from
                else None
            ),
            verbose=args.logging_settings.rllib_verbosity,
            storage_path=str(args.checkpoint_settings.checkpoint_dir.resolve()),
            callbacks=callbacks,
        )
        last_checkpoint = results.get_last_checkpoint()
        logger.info("Training complete")
    finally:
        # Always shutdown ray and release the environment from training even if there is an error
        # will reraise the error unless a control flow statement is added
        if not args.resource_settings.using_cluster:
            ray.shutdown()

    if args.checkpoint_settings.export_onnx:
        export_onnx_from_policy(
            Algorithm.from_checkpoint(last_checkpoint), Path(results.trials[-1].path)
        )
        logger.info("Models exported to ONNX at %s", results.trials[-1].path)
    return results


class RLlibTrainCommand(MetaTrainCommand[RLlibScriptArgs]):

    @property
    def algorithm_table(self) -> Dict[str, Type[Any]]:
        return {
            "sac": SACSettings,
            "ppo": PPOSettings,
            "impala": IMPALASettings,
            "appo": APPOSettings,
        }

    @property
    def algorithm_help(self) -> Dict[str, str]:
        return {
            "sac": "Train a model using Soft Actor-Critic(SAC) with rllib.",
            "ppo": "Train a model using Proximal Policy Optimization(PPO) with rllib.",
            "impala": "Train a model using IMPALA with rllib.",
            "appo": "Train a model using Asynchronous Proximal Policy Optimization(APPO) with rllib.",
        }


app = RLlibTrainCommand(app, RLlibScriptArgs, main, logger).make()

if __name__ == "__main__":
    app.meta()
