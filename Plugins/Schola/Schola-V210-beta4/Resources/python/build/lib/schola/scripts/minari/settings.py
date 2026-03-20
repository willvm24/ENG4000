# Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.

from __future__ import annotations

import logging
from typing import Annotated, Optional
from dataclasses import dataclass, field
from pathlib import Path
from cyclopts import Parameter, validators
from cyclopts.types import URL, Email
from schola.scripts.common.settings import EnvironmentArgs


@dataclass
class MinariCollectionArgs:
    """
    Dataclass for configuring Minari dataset collection parameters.
    """

    dataset_id: Optional[str] = None
    "Unique identifier for the Minari dataset. This will be used to name the dataset when it is created."

    num_steps: Annotated[int, Parameter(validator=validators.Number(gte=1))] = 1000
    "Total number of steps to collect for the dataset. This is the total number of environment steps that will be recorded."

    seed: Optional[int] = None
    "Random seed for reproducibility. If None, the environment will use a random seed."

    author: Optional[str] = None
    "Author name for the dataset metadata."

    author_email: Optional[Email] = None
    "Author email for the dataset metadata."

    code_permalink: Optional[URL] = None
    "URL to the code or repository used to generate the dataset."

    algorithm_name: Optional[str] = None
    "Name of the algorithm or policy used to collect the data."

    description: Optional[str] = None
    "Description of the dataset."

    record_infos: bool = False
    "Whether to record the info dictionaries in the dataset. If False, only observations, actions, rewards, terminations, and truncations are recorded."

    data_path: Annotated[
        Optional[Path],
        Parameter(validator=validators.Path(file_okay=False, dir_okay=True)),
    ] = None
    "Directory path where Minari datasets will be stored. If None, uses the default Minari datasets directory (MINARI_DATASETS_PATH environment variable or ~/.minari/datasets/)."

    def __post_init__(self):
        if self.data_path and not self.data_path.exists():
            self.data_path.mkdir(parents=True, exist_ok=True)


@dataclass
class MinariLoggingArgs:
    """
    Dataclass for configuring logging settings for Minari data collection.
    """

    schola_verbosity: Annotated[
        int, Parameter(validator=validators.Number(gte=0, lte=2))
    ] = 0
    "Verbosity level for Schola-specific logging. This controls the level of detail in the output from Schola-related components during data collection."


@dataclass
class MinariScriptArgs:
    """
    Top level dataclass for configuring the script arguments used in the Minari data collection launcher.
    """

    collection_settings: Annotated[
        MinariCollectionArgs, Parameter(group="Collection Arguments", name="*")
    ] = field(default_factory=MinariCollectionArgs)
    "Settings for configuring the dataset collection process."

    logging_settings: Annotated[
        MinariLoggingArgs, Parameter(group="Logging Arguments", name="*")
    ] = field(default_factory=MinariLoggingArgs)
    "Settings for configuring logging during data collection."

    environment_settings: Annotated[
        EnvironmentArgs, Parameter(group="Environment Arguments", name="*")
    ] = field(default_factory=EnvironmentArgs)
    "Settings for configuring the environment."
