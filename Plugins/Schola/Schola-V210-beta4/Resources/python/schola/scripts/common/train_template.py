from collections import defaultdict
import logging
from typing import (
    Annotated,
    Any,
    Callable,
    Dict,
    Generic,
    NewType,
    Type,
    TypeVar,
    Union,
)

from cyclopts import App, Parameter

from schola.scripts.common.settings import (
    UnrealEditorSimulatorArgs,
    UnrealExecutableSimulatorArgs,
    UnrealProjectSimulatorArgs,
)

ScriptArgsType = TypeVar("ScriptArgsType")
SimulatorArgsType = Union[
    UnrealEditorSimulatorArgs, UnrealExecutableSimulatorArgs, UnrealProjectSimulatorArgs
]


class MetaCommand(Generic[ScriptArgsType]):
    def __init__(
        self,
        app: App,
        args_type: Type[ScriptArgsType],
        main_func: Callable[[ScriptArgsType], Any],
        logger: logging.Logger,
    ):
        self.app = app
        self.args_type = args_type
        self._main_func = main_func
        self._logger = logger

    def make_simulator_command(self, simulator_type: Type[SimulatorArgsType]):
        SimulatorType = NewType("SimulatorType", simulator_type)  # type: ignore
        ArgsType = NewType("ArgsType", self.args_type)  # type: ignore
        _main_func = self._main_func
        if not issubclass(simulator_type, UnrealEditorSimulatorArgs):

            def completed_simulator_command(
                simulator_args: Annotated[SimulatorType, Parameter(name="*")],
                *,
                hidden_script_args: Annotated[ArgsType, Parameter(parse=False)]
            ):
                hidden_script_args.environment_settings.simulator = simulator_args  # type: ignore
                self._logger.info("Arguments: %s", hidden_script_args)
                _main_func(hidden_script_args)

            return completed_simulator_command
        else:

            def completed_editor_command(
                *, hidden_script_args: Annotated[ArgsType, Parameter(parse=False)]
            ):
                hidden_script_args.environment_settings.simulator = UnrealEditorSimulatorArgs()  # type: ignore
                self._logger.info("Arguments: %s", hidden_script_args)
                _main_func(hidden_script_args)

            return completed_editor_command

    @property
    def simulator_table(self) -> Dict[str, Type[SimulatorArgsType]]:
        return {
            "editor": UnrealEditorSimulatorArgs,
            "executable": UnrealExecutableSimulatorArgs,
            "project": UnrealProjectSimulatorArgs,
        }

    @property
    def simulator_help(self) -> Dict[str, str]:
        return {
            "editor": "Connected to a running Unreal Editor instance. Default if no simulator is provided.",
            "executable": "Run Unreal from an pre-built executable.",
            "project": "Build and Run Unreal from a UProject File.",
        }


class MetaTrainCommand(MetaCommand[ScriptArgsType]):

    def make(self):
        # setup the default meta func on the base app to parse the Script Args
        ResolvedArgsType = NewType("ArgsType", self.args_type)  # type: ignore

        @self.app.meta.default
        def train_command_default(
            *tokens: Annotated[str, Parameter(show=False, allow_leading_hyphen=True)],
            script_args: Annotated[
                ResolvedArgsType, Parameter(name="*")
            ] = self.args_type()
        ):  # type: ignore
            additional_kwargs = {
                "hidden_script_args": script_args,
            }
            command, bound, ignored = self.app.parse_args(tokens)
            return command(*bound.args, **bound.kwargs, **additional_kwargs)

        self.app.group_commands = "Algorithm (Choose One)"
        # setup the algorithm commands (e.g. PPO, SAC, etc.)
        for algorithm in self.algorithm_table:
            algorithm_app = App(name=algorithm, group_commands="Simulator (Choose One)")
            algorithm_type = self.algorithm_table[algorithm]
            AlgorithmType = NewType("AlgorithmType", algorithm_type)  # type: ignore

            # setup the default meta func on the algorithm app to parse the Algorithm Args
            @algorithm_app.meta.default
            def algorithm_meta_app(
                *tokens: Annotated[
                    str, Parameter(show=False, allow_leading_hyphen=True)
                ],
                algorithm_args: Annotated[AlgorithmType, Parameter(name="*")] = algorithm_type(),  # type: ignore
                hidden_script_args: Annotated[ResolvedArgsType, Parameter(parse=False)]
            ):  # type: ignore
                additional_kwargs = {
                    "hidden_script_args": hidden_script_args,
                }
                hidden_script_args.algorithm_settings = algorithm_args  # type: ignore
                command, bound, ignored = algorithm_app.parse_args(tokens)
                return command(*bound.args, **bound.kwargs, **additional_kwargs)

            # setup the simulator commands (e.g. editor, executable, project)
            for simulator_type in self.simulator_table:
                sim_command = self.make_simulator_command(
                    self.simulator_table[simulator_type]
                )
                if simulator_type == "editor":
                    algorithm_app.default(sim_command)
                algorithm_app.command(sim_command, name=simulator_type)
                algorithm_app[simulator_type].help = self.simulator_help[simulator_type]

            self.app.command(algorithm_app.meta, name=algorithm)
            self.app[algorithm].help = self.algorithm_help[algorithm]

        return self.app

    @property
    def algorithm_table(self) -> Dict[str, Type[Any]]:
        raise NotImplementedError("algorithm_table must be implemented in the subclass")

    @property
    def algorithm_help(self) -> Dict[str, str]:
        return defaultdict(str)


class MetaNoAlgCommand(MetaCommand[ScriptArgsType]):

    def make(self):
        ResolvedArgsType = NewType("ArgsType", self.args_type)  # type: ignore

        # setup the default meta func on the base app to parse the Script Args
        @self.app.meta.default
        def train_command_default(
            *tokens: Annotated[str, Parameter(show=False, allow_leading_hyphen=True)],
            script_args: Annotated[
                ResolvedArgsType, Parameter(name="*")
            ] = self.args_type()
        ):  # type: ignore
            additional_kwargs = {
                "hidden_script_args": script_args,
            }
            command, bound, ignored = self.app.parse_args(tokens)
            return command(*bound.args, **bound.kwargs, **additional_kwargs)

        self.app.group_commands = "Simulator (Choose One)"

        # setup the simulator commands (e.g. editor, executable, project)
        for simulator_type in self.simulator_table:
            sim_command = self.make_simulator_command(
                self.simulator_table[simulator_type]
            )
            if simulator_type == "editor":
                self.app.default(sim_command)
            self.app.command(sim_command, name=simulator_type)
            self.app[simulator_type].help = self.simulator_help[simulator_type]

        return self.app
