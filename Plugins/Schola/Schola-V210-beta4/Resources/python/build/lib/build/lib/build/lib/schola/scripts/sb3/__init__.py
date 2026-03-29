# Copyright (c) 2024 Advanced Micro Devices, Inc. All Rights Reserved.
"""
Scripts for training with Stable Baselines 3.
"""

from cyclopts import App, Parameter, validators, group_extractors, Group

sb3_app = App(name="schola-sb3", help="Use StableBaselines3 with Schola!")

from .train import app as train_app

from .sb3_to_onnx import export_onnx_app

sb3_app.command(train_app.meta, name="train")

sb3_app.command(export_onnx_app, name="export")

if __name__ == "__main__":
    sb3_app()
