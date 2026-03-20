# Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.

from typing import Any, Dict, Tuple, Type

from schola.core.protocols.async_base import AsyncBaseRLProtocol
from schola.core.protocols.base import BaseProtocol


class UnsupportedProtocolException(Exception):
    """
    Exception raised when a protocol is not supported by a simulator.

    Raised when the protocol is not listed in ``supported_protocols`` (sync) or
    ``supported_async_protocols`` (async).
    """

    pass


class BaseSimulator:
    """
    Base class for all simulators.

    This abstract class defines the interface for simulator implementations
    that manage simulation instances (e.g. Unreal Editor, standalone executable, etc.).
    """

    def start(self, protocol_properties: Dict[str, Any]) -> None:
        """
        Start the Simulator.

        Parameters
        ----------
        protocol_properties : Dict[str, Any]
            Protocol-specific properties to pass to the simulator at startup. Simulator is responsible for passing these. (e.g. Port)
        """
        ...

    def stop(self) -> None:
        """
        Stop the simulator.

        This method should safely shut down the simulator and clean up resources.
        """
        ...

    @property
    def supported_protocols(self) -> Tuple[Type[BaseProtocol], ...]:
        """
        Get the protocols supported by this simulator.

        Returns
        -------
        Tuple[Type[BaseProtocol], ...]
            A tuple of protocol classes that this simulator supports.
        """
        return tuple()

    @property
    def supported_async_protocols(self) -> Tuple[Type[AsyncBaseRLProtocol], ...]:
        """
        Async RL protocol classes this simulator supports (see ``AsyncBaseRLProtocol``).

        Returns
        -------
        Tuple[type, ...]
            Tuple of concrete async protocol types; empty if the simulator has no async support.
        """
        return tuple()

    def __bool__(self) -> bool:
        """
        Check if the simulator is currently running.

        Returns
        -------
        bool
            True if the simulator is running, False otherwise.
        """
        ...
