# Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.

from datetime import datetime
from itertools import islice
import json
from queue import Queue
import subprocess
from typing import Callable, List, Literal, TypedDict
from dataclasses import dataclass
import sys
import os
import time

import pytest
import logging
logger = logging.getLogger(__name__)
import pathlib
import re
from functools import cache
from typing import Dict, List, Optional, Any, Tuple

from schola.core.utils.ubt import (
    UBTCommand,
    get_project_file,
    get_editor_executable_path,
    get_ue_version,
    get_ubt_path,
    build_executable,
)

def batched(iterable, size):
    it = iter(iterable)
    batch = list(islice(it, size))
    while batch:
        yield batch
        batch = list(islice(it, size))

# Constants
MAX_SEARCH_DEPTH = 10
CPP_EXTENSIONS = {".cpp", ".cc"}
AUTOMATION_TEST_PATTERN = r'\w+_AUTOMATION_TEST\s*\([^,]+,\s*"([^"]+)"'


@dataclass
class UnrealTestEvent:
    message: str
    filename: Optional[pathlib.Path]
    line_number: Optional[int]
    timestamp: str
    context: Optional[str] = None # Only known usage is for log entries which use "log"

    @classmethod
    def make(cls,event_entry):
        # Unreal sometimes uses placeholder values in the test reports for filename and line number
        # simplify it to use None instead of placeholder str/int
        filename = None if event_entry.get("filename", "") == "" else pathlib.Path(event_entry["filename"])
        line_number = None if event_entry.get("lineNumber", -1) == -1 else event_entry.get("lineNumber", None)
        context = None if event_entry["event"].get("context", "") == "" else event_entry["event"].get("context", "")
        return cls(
            message=event_entry["event"].get("message", ""),
            filename=filename,
            line_number=line_number,
            timestamp=event_entry.get("timestamp", datetime.now().isoformat()),
            context=context,
        )
    
    @classmethod
    def make_from_item(cls, item: "UnrealTestItem", msg: str):
        return cls(
            message=msg,
            filename=item.cpp_file,
            line_number=item.lineno,
            timestamp=datetime.now().isoformat(),
        )

@dataclass
class UnrealTestResult:
    name: str
    test_path: str
    outcome: Literal["passed", "failed", "skipped"]
    duration: float
    errors: List[UnrealTestEvent]
    warnings: List[UnrealTestEvent]

    @property
    def num_warnings(self) -> int:
        return len(self.warnings)
    
    @property
    def num_errors(self) -> int:
        return len(self.errors)

    @property
    def passed(self) -> bool:
        return self.outcome == "passed"
    
    @property
    def failed(self) -> bool:
        return self.outcome == "failed"
    
    @property
    def skipped(self) -> bool:
        return self.outcome == "skipped"

    @property
    def sanitized_test_path(self) -> str:
        return self.test_path.strip()

    @classmethod
    def make(cls,test_results):
        
        entries = [x for x in test_results.get("entries", []) if "event" in x]
        errors = filter(lambda x: x["event"].get("type", "").lower() == "error", entries)
        warnings = filter(lambda x: x["event"].get("type", "").lower() == "warning", entries)
        
        duration = float(test_results.get("duration", 0))
        
        outcome = test_results.get("state", "").strip().lower()
        if outcome == "success":
            outcome = "passed"
        elif outcome == "fail":
            outcome = "failed"
        else:
            outcome = "skipped"
        # hard fail if we don't have a valid display name or full test path
        if not test_results["testDisplayName"] or not test_results["fullTestPath"]:
            raise Exception(f"Invalid test results: {test_results}. Test display name or full test path is missing.")
        
        return cls(
            name=test_results["testDisplayName"],
            test_path=test_results["fullTestPath"],
            outcome=outcome,
            duration=duration,
            errors=[UnrealTestEvent.make(entry) for entry in errors],
            warnings=[UnrealTestEvent.make(entry) for entry in warnings],
        )

class UnrealTestItem(pytest.Item):

    def __init__(self, name:str, test_path:str, cpp_file:pathlib.Path, line_number:int=0, **kwargs):
        name = name.strip()
        super().__init__(name=name,**kwargs)
        self.unreal_result : Optional[UnrealTestResult] = None  # Will be populated by pytest_runtestloop
        self.cpp_file = cpp_file
        self.lineno = line_number 
        self.test_path = test_path
        try:
            self.add_marker(pytest.mark.xdist_group("unreal_automation_tests"))
        except:
            # wasn't able to mark, probably because pytest-xdist isn't installed which is OK
            pass

    @property
    def sanitized_test_path(self) -> str:
        return self.test_path.strip()

    def runtest(self):
        if self.unreal_result is None:
            pytest.skip(
                "Unreal Engine did not execute this test. "
                "Verify --engine-path, --test-hierarchy-path, and any test filters."
            )

        else:
            self._handle_test_outcome(self.unreal_result)

    # Handles unreal test results depending on success or failure of test execution
    def _handle_test_outcome(self, result: UnrealTestResult):
        if result.skipped:
            pytest.skip("Test not executed in this batch")
        elif result.passed:
            pass
        elif result.failed:
            # Format error messages from Unreal entries
            message_parts = [f"Test failed with {result.num_errors} errors and {result.num_warnings} warnings"]
            
            # Add error details
            for entry in result.errors:
                message_parts.append(f"\n  Error")
                if entry.filename is not None and entry.line_number is not None:
                    message_parts.append(f" at {entry.filename}:{entry.line_number}")
                message_parts.append(f":\n")
                message_parts.append(f"    {entry.message}")
            
            # Add warning details
            for entry in result.warnings:
                message_parts.append(f"\n  Warning")
                if entry.filename is not None and entry.line_number is not None:
                    message_parts.append(f" at {entry.filename}:{entry.line_number}")
                message_parts.append(f":\n")
                message_parts.append(f"    {entry.message}")
            
            message = "".join(message_parts)
            raise UnrealTestException(message)

    def repr_failure(self, excinfo, style=None):
        if isinstance(excinfo.value, UnrealTestException):
            return excinfo.value.args[0]
        return super().repr_failure(excinfo, style=None)

    def reportinfo(self):
        if self.cpp_file:
            return self.cpp_file, self.lineno, f"usecase: {self.name}"
        return self.fspath, 0, f"usecase: {self.name}"


@dataclass
class UnrealTestBatch:
    unreal_test_items: List[UnrealTestItem]
    report_dir: pathlib.Path
    batch_index: int
    command_line_args_folder: pathlib.Path
    _process: Optional[subprocess.Popen] = None
    retry: bool = False

    @property
    def process(self) -> subprocess.Popen:
        if self._process is None:
            raise Exception("Process is not initialized")
        return self._process

    @property
    def test_string(self) -> str:
        return "+".join((item.test_path for item in self.unreal_test_items))
    
    @property
    def report_path(self) -> pathlib.Path:
        return self.report_dir / f"batch_{self.batch_index}"

    @property
    def report_file(self) -> pathlib.Path:
        return self.report_path / "index.json"

    @property
    def command_line_args_file(self) -> pathlib.Path:
        return self.command_line_args_folder / f"batch_{self.batch_index}.txt"

    @property
    def command_line_arg_string(self) -> str:
        return " ".join(
            [
                "-unattended",
                "-nullrhi",
                "-NoTrace",
                f"-ReportExportPath={self.report_path}",
                f'-ExecCmds="Automation RunTest {self.test_string};Quit"',
            ]
        )

    def write_command_line_args_file(self) -> None:
        with open(self.command_line_args_file, "w") as f:
            f.write(self.command_line_arg_string)

    def prepare(self) -> None:
        self.report_path.mkdir(parents=True, exist_ok=True)
        self.command_line_args_folder.mkdir(parents=True, exist_ok=True)
        self.write_command_line_args_file()
    
    def run(self, editor_path: pathlib.Path, uproject_path: pathlib.Path) -> subprocess.Popen:
        args = [str(editor_path), f"-project={uproject_path}", "-ResumeRunTest", f"-CmdLineFile={self.command_line_args_file.resolve()}"]
        
        print(f"Running Unreal Test Command: " + " ".join(args))
        self._process = subprocess.Popen(
            args,
            stdout=subprocess.PIPE, # type: ignore
            stderr=subprocess.PIPE, # type: ignore
            text=True,
        )
        return self.process

    def load_test_results(self) -> None:
        test_results = {}
        if self.report_file.resolve().exists():
            with open(self.report_file, "r", encoding="utf-8-sig") as f:
                data = json.load(f)
                for test in data.get("tests", []):
                    test_result = UnrealTestResult.make(test)
                    test_results[test_result.sanitized_test_path] = test_result
        else:
            logger.warning(
                "Expected Report file for batch %s but no file found. This can sometimes mean Unreal failed to build correctly.",
                self.batch_index,
            )
        # If any tests are not found in the results, mark them as failed
        for item in self.unreal_test_items:
            if item.sanitized_test_path in test_results:
                item.unreal_result = test_results[item.sanitized_test_path]
            else:
                item.unreal_result = UnrealTestResult(
                    name=item.name,
                    test_path=item.test_path,
                    outcome="failed",
                    duration=0.0,
                    errors=[UnrealTestEvent(message="Test not found in Unreal Engine results", filename=item.cpp_file, line_number=item.lineno, timestamp="")],
                    warnings=[]
                )

    def mark_all_unreal_tests_failed(self, msg:str) -> None:
        for item in self.unreal_test_items:
            item.unreal_result = UnrealTestResult(
                name=item.name,
                test_path=item.test_path,
                outcome="failed",
                duration=0.0,
                errors=[UnrealTestEvent.make_from_item(item, msg)],
                warnings=[]
            )
    
    def clean_up(self) -> None:
        if self._process:
            self.process.kill()
            self.process.communicate()


def run_unreal_tests(
    unreal_test_batches: List[UnrealTestBatch],
    editor_path: pathlib.Path,
    uproject_path: pathlib.Path,
):
    """Run Unreal automation tests using a simple blocking subprocess call."""

    global_timeout = int(os.getenv("SCHOLA_UNREAL_TIMEOUT", "240"))  # seconds
    try:
        test_queue = Queue()
        for test_batch in unreal_test_batches:
            test_batch.prepare()
            
        remaining_test_batches = [test_batch for test_batch in unreal_test_batches]
        start_time = time.time()

        # There is a weird edge case, where batch_0 times-out, and batch_n failes to retry a critical failure because we were waiting for batch_0.
        # We handle this by never waiting more than half the remaining timeout for a single batch 
        # This can still fail but would require several threads to timeout while also having critical failures.
        while remaining_test_batches:
            failed_test_batches = []
            
            # Check if we've already exceeded the global timeout, and there are no active processes, since we are between rounds of retries
            elapsed_time = time.time() - start_time
            if elapsed_time >= global_timeout:
                logger.warning(
                    "Global timeout of %s seconds reached. Killing all remaining processes.",
                    global_timeout,
                )
                for test_batch in remaining_test_batches:
                    test_batch.clean_up()
                break
            
            for test_batch in remaining_test_batches:
                test_batch.run(editor_path, uproject_path)

            for test_batch in remaining_test_batches:
                # Calculate remaining time for this batch
                elapsed_time = time.time() - start_time
                remaining_time = max(0, global_timeout - elapsed_time)
                
                if remaining_time <= 0:
                    logger.warning(
                        "Global timeout of %s seconds reached. Killing batch %s.",
                        global_timeout,
                        test_batch.batch_index,
                    )
                    test_batch.clean_up()
                    continue
                
                try:
                    stdout_str, stderr_str = test_batch.process.communicate(timeout=remaining_time/2)
                    # 0 is success, 255 is some test came back as failed but the overall test run was successful
                    if test_batch.process.returncode != 0 and test_batch.process.returncode != 255:
                        logger.warning(
                            "Unreal Automation Test batch %s failed with error %s. Retrying...",
                            test_batch.batch_index,
                            test_batch.process.returncode,
                        )
                        test_batch.retry = True
                        failed_test_batches.append(test_batch)
                    else:
                        logger.info(
                            "Unreal Automation Test batch %s completed with no errors",
                            test_batch.batch_index,
                        )
                    
                except subprocess.TimeoutExpired:
                    logger.warning(
                        "Unreal Automation Test batch %s timed out (local timeout of %s seconds reached), killing process.",
                        test_batch.batch_index,
                        remaining_time / 2,
                    )
                    test_batch.clean_up()

            # retry all the failed test batches
            remaining_test_batches = failed_test_batches
    
    except:
        # cleanup all the test batches if something unexpected occurs
        for test_batch in unreal_test_batches:
            test_batch.clean_up()
        raise Exception("Unreal Automation Test failed with unknown error, killing all batches..")


def get_uproject_file(project_dir: pathlib.Path) -> Optional[pathlib.Path]:
    uproject_file = get_project_file(project_dir)
    return uproject_file.resolve() if uproject_file else None

def get_engine_path_from_config(config) -> pathlib.Path:
    engine_path = config.getoption("--engine-path") or config.getini("engine_path")
    return pathlib.Path(engine_path)

def get_build_unreal_from_config(config) -> bool:
    # --no-build-unreal takes precedence
    if config.getoption("--no-build-unreal"):
        return False
    build_option = config.getoption("--build-unreal")
    if build_option is not None:
        return build_option
    return config.getini("build_unreal")

def pytest_addoption(parser):
    try:
        parser.addoption(
            "--engine-path",
            action="store",
            default=None,
            help="The absolute path to the Unreal Engine installation",
        )
    except ValueError:
        pass  # Option already exists

    try:
        parser.addini(
            "engine_path",
            help="The path to the Unreal Engine installation",
            default="C:/Program Files/Epic Games/UE_5.6",
        )
    except ValueError:
        pass  # Exists in .ini

    try:
        parser.addoption(
            "--build-unreal",
            action="store_true",
            default=None,
            help="Build Unreal project before running tests",
        )
    except ValueError:
        pass  # Option already exists

    try:
        parser.addoption(
            "--no-build-unreal",
            action="store_true",
            default=False,
            help="Skip building Unreal project before running tests",
        )
    except ValueError:
        pass  # Option already exists

    try:
        parser.addini(
            "build_unreal",
            type="bool",
            help="Build Unreal project before running tests",
            default=False,
        )
    except ValueError:
        pass  # Exists in .ini

@pytest.fixture(scope="session")
def unreal_path(request) -> pathlib.Path:
    return get_engine_path_from_config(request.config)


@pytest.fixture(scope="session")
def unreal_report_dir(tmp_path_factory) -> pathlib.Path:
    """Create a session-scoped temporary directory for Unreal test reports."""
    report_dir = tmp_path_factory.mktemp("unreal_test_reports")
    return report_dir

@pytest.fixture(scope="session")
def unreal_command_line_file_dir(tmp_path_factory) -> pathlib.Path:
    """Create a session-scoped temporary directory for the Unreal Command Line Args written to a file."""
    command_line_file_dir = tmp_path_factory.mktemp("unreal_command_line_args")
    return command_line_file_dir


@pytest.hookimpl(tryfirst=True)
def pytest_sessionstart(session):
    """Set up session-level test report directory."""
    # Use pytest's temp path factory to create a session temp directory
    tmp_path_factory = session.config._tmp_path_factory
    report_dir = tmp_path_factory.mktemp("unreal_test_reports")
    session.config._unreal_report_dir = report_dir


class UnrealTestFile(pytest.File):

    def collect(self):
        # Skip C++ test collection on Linux
        if sys.platform != "win32":
            return
            
        test_details = []
        
        with open(self.path, "r", encoding="utf-8", errors="ignore") as f:
            for i,line in enumerate(f.readlines()):
                found_tests = re.findall(AUTOMATION_TEST_PATTERN, line)
                # If we found any tests, add them to the list
                if len(found_tests) > 0:
                    test_details += zip(found_tests, [i] * len(found_tests))

        for test_path, line_number in test_details:
            name = test_path.split(".")[-1] # Use the last part of the full path
            yield UnrealTestItem.from_parent(self, name=name, test_path=test_path, cpp_file=self.path, line_number=line_number)

# Use this to have custom exceptions
class UnrealTestException(Exception):
    pass

def is_cpp_test_file(file_path: pathlib.Path) -> bool:
    if file_path.suffix.lower() not in CPP_EXTENSIONS:
        return False
    return file_path.name.endswith("Test" + file_path.suffix)

def pytest_collect_file(parent, file_path : pathlib.Path):
    # Skip C++ test collection on Linux
    if sys.platform != "win32":
        return None
        
    # Only custom-collect C++ automation tests; let pytest's normal Python collectors
    # handle .py tests in `Test/` and doctests in `Resources/python/schola`.
    if file_path.suffix.lower() in CPP_EXTENSIONS:
        return UnrealTestFile.from_parent(parent=parent, path=file_path)
    return None

def pytest_ignore_collect(collection_path: pathlib.Path, config):
    # Skip C++ test files on Linux
    if sys.platform != "win32" and is_cpp_test_file(collection_path):
        return True  # Ignore C++ test files on Linux
        
    if is_cpp_test_file(collection_path):
        return False # Forcibly do not ignore C++ test files
    return None # Punt on everything else

class UnrealTestRunner:

    def __init__(self, session, unreal_items: List[UnrealTestItem], report_dir: pathlib.Path, command_line_file_dir: pathlib.Path):
        self.session = session
        self.unreal_items = unreal_items
        self.unreal_path = get_engine_path_from_config(session.config)
        self.should_build = get_build_unreal_from_config(session.config)
        self.report_dir = report_dir
        self.command_line_file_dir = command_line_file_dir

    def run_tests(self) -> bool:
        # Setup the report directory
        
        editor_path = get_editor_executable_path(self.unreal_path)
        max_tests_per_batch = 100
        unreal_test_batches = [UnrealTestBatch(batch, self.report_dir, i, self.command_line_file_dir) for i,batch in enumerate(batched(self.unreal_items, max_tests_per_batch))]
        # handle if we are in a few common subdirectories of the project/using common pytest.ini file locations
        uproject_path = (get_uproject_file(self.session.config.rootpath)
        or get_uproject_file(self.session.config.rootpath.parent.parent)
        or get_uproject_file(self.session.config.inipath.parent)
        or get_uproject_file(self.session.config.inipath.parent.parent.parent))
        
        if not uproject_path:
            logger.error(
                "Error Building Unreal Project: No .uproject file found in directory"
            )
            for test_batch in unreal_test_batches: 
                test_batch.mark_all_unreal_tests_failed("Unreal Engine failed to build and run tests")
            return False

        if self.should_build:
            try:
                self._build_unreal_project(uproject_path)
            except Exception as e:
                logger.error("Error building Unreal project: %s", e)
                for test_batch in unreal_test_batches: 
                    test_batch.mark_all_unreal_tests_failed("Unreal Engine failed to build and run tests")
                return False
        
        try:
            run_unreal_tests(
                editor_path=editor_path,
                uproject_path=uproject_path,
                unreal_test_batches=unreal_test_batches,
            )
        except Exception as e:
            logger.error("Error running Unreal tests: %s", e)
            for test_batch in unreal_test_batches: 
                test_batch.mark_all_unreal_tests_failed("Error Running Tests")
            return False

        for test_batch in unreal_test_batches:
            test_batch.load_test_results()

        return True

    def _build_unreal_project(self, uproject_path: pathlib.Path):
        """Build the Unreal project before running tests."""
        project_folder = uproject_path.parent
        
        # Get UE version and UBT path
        ue_version = get_ue_version(uproject_path)
        if not ue_version:
            raise Exception("Could not determine Unreal Engine version from .uproject file")
        
        ubt_path = get_ubt_path(project_folder, ue_version)

        # Try to use the engine path if we couldn't find UBT via the project files
        if not ubt_path and self.unreal_path:
            import platform
            script_name = "RunUAT.bat" if platform.system() == "Windows" else "RunUAT.sh"
            possible_path = self.unreal_path / "Engine" / "Build" / "BatchFiles" / script_name
            if possible_path.exists():
                ubt_path = possible_path

        if not ubt_path:
            raise Exception("Could not find Unreal Build Tool (UBT) path")
        
        # Setup build directory
        #build_dir = project_folder / "Build" / "Staged"
        #build_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Building Unreal project: %s", uproject_path)
        logger.info("Using UBT at: %s", ubt_path)
        
        command = UBTCommand(
            ubt_path=ubt_path,
            project_file=uproject_path,
            should_package=False,
            should_cook=False, # No need to cook content since tests should not have visibility on content
            force_monolithic=False, # Tests run from editor build so no need for monolithic build
            all_maps=False, # No need to cook any maps since tests should not have visibility on content
            should_clean=False, # No need to clean since we are not building a new project
        )
        
        result = command.run()

        logger.info("=" * 10 + " Unreal Build Output " + "=" * 10)
        logger.info(result.stdout.decode("utf-8", errors="ignore"))
        logger.info("=" * 10 + " End of Unreal Build Output " + "=" * 10)
        if result.returncode != 0:
            if result.stderr:
                logger.error("=" * 10 + " Unreal Build Error " + "=" * 10)
                logger.error(result.stderr.decode("utf-8", errors="ignore"))
                logger.error("=" * 10 + " Unreal Build Error " + "=" * 10)
            raise Exception(f"Unreal build failed with return code {result.returncode}")
             
@pytest.hookimpl(tryfirst=True)
def pytest_runtestloop(session):
    if session.testsfailed and not session.config.option.continue_on_collection_errors:
        raise session.Interrupted(
            f"{session.testsfailed} error{'s' if session.testsfailed != 1 else ''} during collection"
        )

    if session.config.option.collectonly:
        return True

    unreal_items = [item for item in session.items if isinstance(item, UnrealTestItem)]

    # Run Unreal tests in batch if we have any (only on Windows, since linux can hang)
    if unreal_items and sys.platform == "win32":
        # Ensure report directory exists (create if pytest_sessionstart wasn't called)
        if not hasattr(session.config, '_unreal_report_dir'):
            tmp_path_factory = session.config._tmp_path_factory
            session.config._unreal_report_dir = tmp_path_factory.mktemp("unreal_test_reports")
        if not hasattr(session.config, '_unreal_command_line_file_dir'):
            tmp_path_factory = session.config._tmp_path_factory
            session.config._unreal_command_line_file_dir = tmp_path_factory.mktemp("unreal_command_line_args")
        report_dir = session.config._unreal_report_dir
        command_line_file_dir = session.config._unreal_command_line_file_dir

        runner = UnrealTestRunner(session, unreal_items, report_dir, command_line_file_dir)
        runner.run_tests()
    
    # Run all tests (including the now-populated Unreal tests)
    _run_all_tests(session)

    return True


def _run_all_tests(session):
    for i, item in enumerate(session.items):
        nextitem = session.items[i + 1] if i + 1 < len(session.items) else None
        item.config.hook.pytest_runtest_protocol(item=item, nextitem=nextitem)

        if session.shouldfail:
            raise session.Failed(session.shouldfail)
        if session.shouldstop:
            raise session.Interrupted(session.shouldstop)
