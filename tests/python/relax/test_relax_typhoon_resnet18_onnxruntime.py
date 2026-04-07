# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import importlib

import pytest

from tests.python.relax.typhoon_resnet18_test_utils import (
    build_canonical_resnet18_feed_dict,
    load_canonical_resnet18_model,
    run_canonical_resnet18_reference_onnxruntime,
)


def test_relax_resnet18_onnxruntime_reference_runs():
    model = load_canonical_resnet18_model()
    feed_dict = build_canonical_resnet18_feed_dict(model)
    output = run_canonical_resnet18_reference_onnxruntime(feed_dict, model)
    assert output.shape == (1, 1000)


def test_relax_resnet18_onnxruntime_missing_dependency_fails_clearly(monkeypatch):
    real_import_module = importlib.import_module

    def _patched_import_module(name, package=None):
        if name == "onnxruntime":
            raise ImportError("missing onnxruntime")
        return real_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", _patched_import_module)

    with pytest.raises(RuntimeError, match="onnxruntime"):
        run_canonical_resnet18_reference_onnxruntime(
            {"input": build_canonical_resnet18_feed_dict(load_canonical_resnet18_model())["input"]},
            load_canonical_resnet18_model(),
        )
