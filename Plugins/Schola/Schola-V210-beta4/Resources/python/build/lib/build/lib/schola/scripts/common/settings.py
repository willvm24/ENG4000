# Copyright (c) 2024 Advanced Micro Devices, Inc. All Rights Reserved.

"""
Common utility functions and classes for use in Schola scripts.
"""
from enum import Enum
from typing import Annotated, Literal, Optional, Tuple, List, Type, Union

from dataclasses import dataclass, field
from cyclopts import App, Parameter, validators, group_extractors, Group, types
from pathlib import Path

from rich.console import Console

console = Console()


class ActivationFunctionEnum(str, Enum):
    """
    Activation functions for neural networks.
    """

    ReLU = "relu"  #: Rectified Linear Unit activation function.
    Sigmoid = "sigmoid"  #: Sigmoid activation function.
    TanH = "tanh"  #: Hyperbolic Tangent activation function.


def get_activation_function(activation: ActivationFunctionEnum) -> Type["torch.nn.Module"]:  # type: ignore
    """
    Get the PyTorch activation function class for the specified activation type.

    Parameters
    ----------
    activation : ActivationFunctionEnum
        The activation function type to retrieve.

    Returns
    -------
    Type[torch.nn.Module]
        The PyTorch activation function class (not an instance).

    Raises
    ------
    ValueError
        If the activation function type is not supported.

    Notes
    -----
    PyTorch is imported lazily within this function to avoid import overhead
    when not needed in cli scripts.
    """
    # we don't use a value on the ActivationFunctionEnum in order to lazily import torch only when needed
    from torch import nn as nn

    if activation == ActivationFunctionEnum.ReLU:
        return nn.ReLU6
    elif activation == ActivationFunctionEnum.Sigmoid:
        return nn.Sigmoid
    elif activation == ActivationFunctionEnum.TanH:
        return nn.Tanh
    else:
        raise ValueError(f"Unsupported activation function: {activation}")


@dataclass
class UnrealEditorSimulatorArgs:
    """
    Arguments for connecting to an existing Unreal Editor instance.

    This dataclass is used when you want to connect to a manually-launched
    Unreal Editor session rather than starting a standalone executable.
    """

    @property
    def num_simulators(self) -> int:
        """Number of parallel simulators; always 1 for editor."""
        return 1

    def make(self):
        """
        Create an UnrealEditor simulator instance.

        Returns
        -------
        UnrealEditor
            A configured UnrealEditor simulator instance.
        """
        from schola.core.simulators.unreal.editor import UnrealEditor

        return UnrealEditor()


@dataclass
class UnrealExecutableSimulatorArgs:
    """
    Arguments for the Unreal Engine executable simulator in Schola.

    This dataclass is used when you want to create a standalone Unreal Engine environment controlled
    by the Schola Python API.
    """

    executable_path: Annotated[
        Path,
        Parameter(
            validator=validators.Path(exists=True, file_okay=True, dir_okay=False)
        ),
    ]
    "Path to the standalone executable, when launching a standalone Environment must exist and be a file"

    disable_script: bool = True
    "Flag indicating if the autolaunch script setting in the Unreal Engine Schola Plugin should be disabled. Useful for testing."

    headless: bool = False
    "Flag indicating if the standalone Unreal Engine process should run in headless mode"

    map: Optional[str] = None
    "Map to load when launching a standalone Unreal Engine process"

    fps: Optional[int] = None
    "Fixed FPS to use when running standalone, if None no fixed timestep is used"

    display_logs: bool = True
    "Whether to render logs in a standalone window."

    num_simulators: Annotated[int, Parameter(validator=validators.Number(gte=1))] = 1
    "Number of parallel simulator processes. Headless mode is recommended when N > 1. With a fixed port P, instances use P, P+1, ... P+N-1."

    def make(self):
        """
        Create an UnrealExecutable simulator instance with the specified settings.

        Returns
        -------
        UnrealExecutable
            A configured UnrealExecutable simulator instance.
        """
        from schola.core.simulators.unreal.executable import UnrealExecutable

        return UnrealExecutable(
            self.executable_path,
            self.headless,
            self.map,
            self.display_logs,
            self.fps,
            self.disable_script,
        )


@dataclass
class UnrealProjectSimulatorArgs:
    """
    Arguments for the Unreal Engine project simulator in Schola.
    """

    uproject_path: Annotated[
        Path,
        Parameter(
            validator=validators.Path(exists=True, file_okay=True, dir_okay=False)
        ),
    ]
    "Path to the .uproject file"

    build_dir: Optional[Path] = None
    "Directory to build the Unreal Engine project to, if None builds to Build/Staging in the project directory."

    ubt_path: Optional[Path] = None
    "Path to the Unreal Build Tool, if None will be automatically detected from the project directory."

    disable_script: bool = True
    "Flag indicating if the autolaunch script setting in the Unreal Engine Schola Plugin should be disabled. Useful for testing."

    headless: bool = False
    "Flag indicating if the standalone Unreal Engine process should run in headless mode"

    map: Optional[str] = None
    "Map to load when launching a standalone Unreal Engine process"

    fps: Optional[int] = None
    "Fixed FPS to use when running standalone, if None no fixed timestep is used"

    display_logs: bool = True
    "Whether to render logs in a standalone window."

    num_simulators: Annotated[int, Parameter(validator=validators.Number(gte=1))] = 1
    "Number of parallel simulator processes. One project build is used; headless recommended when N > 1. With a fixed port P, instances use P, P+1, ... P+N-1."

    def make(self):
        """
        Create a UnrealProject simulator instance with the specified settings.

        Returns
        -------
        UnrealProject
            A configured UnrealProject simulator instance.
        """
        from schola.core.simulators.unreal.project import UnrealProject

        return UnrealProject(
            self.uproject_path,
            self.build_dir,
            self.ubt_path,
            use_cached_build=False,
            headless_mode=self.headless,
            map=self.map,
            display_logs=self.display_logs,
            set_fps=self.fps,
            disable_script=self.disable_script,
        )


def protocol_port_for_index(base_port: Optional[int], index: int) -> Optional[int]:
    """
    Return the port for simulator index i when using a base port.

    If base_port is set, returns base_port + index; otherwise None (auto port).
    """
    if base_port is None:
        return None
    return base_port + index


@dataclass
class gRPCProtocolArgs:
    """
    Settings for the gRPC protocol in Schola.
    """

    port: Optional[int] = None
    "Port to connect to the Unreal Engine process, if None an open port will be automatically selected when running standalone. Port is required if connecting to an existing Unreal Engine process."

    url: str = "localhost"
    "URL to connect to the Unreal Engine process."

    environment_start_timeout: Optional[int] = 45
    "Timeout for waiting to see if the environment is ready before assuming it crashed, in seconds."

    def make(self):
        """
        Create a gRPCProtocol instance with the specified settings.

        Returns
        -------
        gRPCProtocol
            A configured gRPCProtocol instance for communication with Unreal Engine.
        """
        from schola.core.protocols.protobuf.gRPC import gRPCProtocol

        return gRPCProtocol(self.url, self.port, self.environment_start_timeout)

    def make_n_async(self, n: int):
        """
        Create an AsyncgRPCProtocol instance with the specified settings.

        Parameters
        ----------
        port : Optional[int], default=None
            Port for this instance. If None, uses this args' port (base port).
            Used for multi-simulator: pass base_port + index for each instance.

        Returns
        -------
        AsyncgRPCProtocol
            A configured async gRPC protocol instance.
        """
        from schola.core.protocols.protobuf.async_gRPC import AsyncgRPCProtocol

        return [
            AsyncgRPCProtocol(
                self.url,
                protocol_port_for_index(self.port, i),
                self.environment_start_timeout,
            )
            for i in range(n)
        ]


IgnoreParameter = Parameter(show=False, parse=False)


@dataclass
class CheckpointArgs:
    """
    Settings for checkpoints in Schola.
    """

    enable_checkpoints: bool = False
    "Enable saving checkpoints"

    checkpoint_dir: types.Directory = Path("./ckpt")
    "Directory to save checkpoints to."

    save_freq: Annotated[int, Parameter(validator=validators.Number(gte=0))] = 100000
    "Frequency with which to save checkpoints."

    name_prefix_override: Optional[str] = None
    "Override the name prefix for the checkpoint files (e.g. SAC, PPO, etc.)"

    export_onnx: bool = False
    "Whether to export the model to ONNX format instead of just saving a checkpoint."

    save_final_policy: bool = False
    "Whether to save the final policy after training is complete."

    def __post_init__(self):
        if (
            self.enable_checkpoints or self.save_final_policy
        ) and not self.checkpoint_dir.exists():
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class EnvironmentArgs:
    """
    Settings for the environment in Schola.
    """

    simulator: Annotated[
        Union[
            UnrealEditorSimulatorArgs,
            UnrealExecutableSimulatorArgs,
            UnrealProjectSimulatorArgs,
        ],
        IgnoreParameter,
    ] = field(default_factory=UnrealEditorSimulatorArgs)
    "Settings for the simulator to use during training"

    protocol: Annotated[gRPCProtocolArgs, Parameter(group="Protocol Arguments")] = (
        field(default_factory=gRPCProtocolArgs)
    )
    "Settings for the protocol to use for communicating with the external simulator"


@dataclass
class Sb3LauncherExtension:
    def get_extra_KVWriters(self) -> List["stable_baselines3.common.logger.KVWriter"]:  # type: ignore
        """
        Returns a list of additional KVWriter to add to the training loop.

        Returns
        -------
        List[KVWriter]
            A list of additional KVWriters to add to the training loop.
        """
        return []

    def get_extra_callbacks(self) -> List["stable_baselines3.common.callbacks.BaseCallback"]:  # type: ignore
        """
        Returns a list of additional callbacks to add to the training loop.

        Returns
        -------
        List[BaseCallback]
            A list of additional callbacks to add to the training loop.
        """
        return []


@dataclass
class RLLibLauncherExtension:
    def get_extra_callbacks(self) -> List["ray.tune.callback"]:
        """
        Returns a list of additional callbacks to add to the training loop.

        Returns
        -------
        List[Callback]
            A list of additional callbacks to add to the training loop.
        """
        return []
