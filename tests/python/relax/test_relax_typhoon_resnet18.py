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

import json
import re
from types import SimpleNamespace

import numpy as np
import pytest
import tvm
import tvm.testing
from tvm.contrib import cc as tvm_cc
from tvm.relax import vm_build
from tests.python.relax import typhoon_resnet18_test_utils as resnet18_test_utils

from tests.python.relax.typhoon_resnet18_test_utils import (
    assert_elementwise_close_with_context,
    build_canonical_resnet18_feed_dict,
    build_canonical_resnet18_vm_executable,
    format_first_mismatch,
    import_canonical_resnet18_relax_module,
    load_canonical_resnet18_model,
    run_canonical_resnet18_on_typhoon,
    run_canonical_resnet18_reference_onnxruntime,
)


_FFI_DEF_RE = re.compile(
    r"__tvm_ffi_([A-Za-z0-9_]+)\(void\* self_handle, void\* args, int32_t num_args, void\* result\) \{"
)


def _extract_ffi_function_bodies(source):
    matches = list(_FFI_DEF_RE.finditer(source))
    return {
        match.group(1): source[match.start() : matches[index + 1].start() if index + 1 < len(matches) else len(source)]
        for index, match in enumerate(matches)
    }


def _assert_graphized_function_has_no_host_fallback(body, *required_markers):
    assert "TVMTyphoonGraphBegin" in body
    assert "TVMTyphoonSubmitGraph" in body
    assert "TVMTyphoonWaitGraph" in body
    for marker in required_markers:
        assert marker in body
    assert "for (" not in body


def _assert_resnet18_typhoon_source_contract(source):
    ffi_bodies = _extract_ffi_function_bodies(source)
    graphized = {
        name: body for name, body in ffi_bodies.items() if "TVMTyphoonGraphBegin" in body
    }
    assert graphized
    for body in graphized.values():
        assert "for (" not in body

    assert "matmul" in graphized
    assert "reshape" in graphized
    _assert_graphized_function_has_no_host_fallback(
        graphized["matmul"],
        "TVMTyphoonAddMatmulTask",
    )
    _assert_graphized_function_has_no_host_fallback(
        graphized["reshape"],
        "TVMTyphoonAddReshapeTask",
    )

    for name, body in ffi_bodies.items():
        if re.fullmatch(r"conv2d\d*", name) and "TVMTyphoonGraphBegin" in body:
            _assert_graphized_function_has_no_host_fallback(
                body,
                "TVMTyphoonAddReshapeTask",
                "TVMTyphoonAddMatmulTask",
            )


def test_relax_resnet18_compiles_to_typhoon_graph():
    model = load_canonical_resnet18_model()
    mod = import_canonical_resnet18_relax_module(model)
    ex, source = build_canonical_resnet18_vm_executable(mod)
    assert isinstance(ex, vm_build.VMExecutable)
    _assert_resnet18_typhoon_source_contract(source)


def test_relax_resnet18_runs_in_typhoon_simulator():
    model = load_canonical_resnet18_model()
    mod = import_canonical_resnet18_relax_module(model)
    feed_dict = build_canonical_resnet18_feed_dict(model)
    reset = tvm.get_global_func("runtime.typhoon.testing_reset", allow_missing=True)
    get_trace = tvm.get_global_func("runtime.typhoon_get_last_trace_json", allow_missing=True)
    assert reset is not None
    assert get_trace is not None
    reset()
    executable, source = build_canonical_resnet18_vm_executable(mod)
    output = run_canonical_resnet18_on_typhoon(feed_dict, mod, executable=executable)
    ref = run_canonical_resnet18_reference_onnxruntime(feed_dict, model)
    trace = json.loads(get_trace())
    _assert_resnet18_typhoon_source_contract(source)
    assert trace
    tvm.testing.assert_allclose(output, ref, rtol=1e-4, atol=1e-4)


def test_run_canonical_resnet18_on_typhoon_reuses_prebuilt_executable(monkeypatch):
    model = load_canonical_resnet18_model()
    mod = import_canonical_resnet18_relax_module(model)
    feed_dict = build_canonical_resnet18_feed_dict(model)
    exe, _ = build_canonical_resnet18_vm_executable(mod)

    def _unexpected_rebuild(*args, **kwargs):
        raise AssertionError("run_canonical_resnet18_on_typhoon should reuse the provided executable")

    monkeypatch.setattr(
        "tests.python.relax.typhoon_resnet18_test_utils.build_canonical_resnet18_vm_executable",
        _unexpected_rebuild,
    )

    output = run_canonical_resnet18_on_typhoon(feed_dict, mod, executable=exe)
    assert output.shape == (1, 1000)


def test_run_canonical_resnet18_on_typhoon_jits_with_c_compiler(monkeypatch):
    calls = {}

    class FakeExecutable:
        def jit(self, **kwargs):
            calls.update(kwargs)
            return "jitted-runtime-module"

    class FakeOutput:
        def numpy(self):
            return np.zeros((1, 1000), dtype="float32")

    class FakeVM:
        def __init__(self, mod, dev):
            calls["vm_mod"] = mod

        def set_input(self, name, *args):
            calls["set_input_name"] = name
            calls["num_inputs"] = len(args)

        def invoke_stateful(self, name):
            calls["invoke_name"] = name

        def get_outputs(self, name):
            calls["get_outputs_name"] = name
            return FakeOutput()

    fake_mod = {"main": SimpleNamespace(params=[SimpleNamespace(name_hint="input")])}
    fake_feed = {"input": np.zeros((1, 3, 224, 224), dtype="float32")}

    monkeypatch.setattr(
        resnet18_test_utils.shutil,
        "which",
        lambda name: f"/usr/bin/{name}" if name in ("clang", "gcc") else None,
    )
    monkeypatch.setattr("tvm.relax.VirtualMachine", FakeVM)

    output = run_canonical_resnet18_on_typhoon(fake_feed, fake_mod, executable=FakeExecutable())

    assert output.shape == (1, 1000)
    assert calls["fcompile"] is tvm_cc.create_shared
    assert calls["cc"] == "clang"
    assert calls["vm_mod"] == "jitted-runtime-module"


def test_typhoon_resnet18_test_utils_loads_canonical_model():
    from tests.python.relax.typhoon_resnet18_test_utils import load_canonical_resnet18_model

    model = load_canonical_resnet18_model()
    assert model is not None


def test_relax_typhoon_reports_first_mismatch_context():
    actual = np.array([0.0, 1.0], dtype="float32")
    expected = np.array([0.0, 2.0], dtype="float32")
    message = format_first_mismatch(
        actual,
        expected,
        layer_hint="stage3_block1",
        task_hint=42,
    )
    assert "first mismatch" in message.lower()
    assert "stage3_block1" in message
    assert "42" in message


def test_relax_typhoon_onnxruntime_failure_surfaces_context():
    with pytest.raises(AssertionError, match="first mismatch|layer|task"):
        assert_elementwise_close_with_context(
            np.array([0.0, 1.0], dtype="float32"),
            np.array([0.0, 2.0], dtype="float32"),
            rtol=1e-4,
            atol=1e-4,
            layer_hint="final_output",
            task_hint=999,
        )
