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

import os
import shutil
import importlib

import numpy as np
import onnx
from onnx.numpy_helper import to_array

TYPHOON_RESNET18_ENV = "TYPHOON_RESNET18_ONNX_PATH"
DEFAULT_RESNET18_PATH = os.path.expanduser("~/model/resnet18.onnx")


def get_canonical_resnet18_model_path():
    model_path = os.environ.get(TYPHOON_RESNET18_ENV, DEFAULT_RESNET18_PATH)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"ResNet18 model not found at {model_path}")
    return model_path


def load_canonical_resnet18_model(model_path=None):
    return onnx.load(model_path or get_canonical_resnet18_model_path())


def build_canonical_resnet18_feed_dict(
    model=None,
    *,
    seed=0,
    input_shape=(1, 3, 224, 224),
    strip_initializer_prefix="onnx::",
):
    model = model or load_canonical_resnet18_model()
    input_name = model.graph.input[0].name
    rng = np.random.default_rng(seed)
    feed = {input_name: rng.standard_normal(input_shape, dtype=np.float32)}
    for initializer in model.graph.initializer:
        name = initializer.name
        if strip_initializer_prefix:
            name = name.replace(strip_initializer_prefix, "")
        feed[name] = to_array(initializer).astype("float32")
    return feed


def import_canonical_resnet18_relax_module(model=None, *, keep_params_in_input=True):
    from tvm import relax
    from tvm.relax.frontend.onnx import from_onnx

    model = model or load_canonical_resnet18_model()
    mod = from_onnx(model, keep_params_in_input=keep_params_in_input)
    mod = relax.transform.DecomposeOpsForInference()(mod)
    mod = relax.transform.LegalizeOps()(mod)
    return mod


def _make_vm_inputs(mod, feed_dict):
    return [feed_dict[param.name_hint] for param in mod["main"].params]


def _pick_host_c_compiler():
    for candidate in ("clang", "gcc", "cc"):
        if shutil.which(candidate):
            return candidate
    return None


def _jit_executable_for_host_runtime(executable):
    from tvm.contrib import cc as tvm_cc

    if not hasattr(executable, "jit"):
        return executable

    cc_name = _pick_host_c_compiler()
    if cc_name is None:
        return executable.jit()
    return executable.jit(fcompile=tvm_cc.create_shared, cc=cc_name)


def build_canonical_resnet18_vm_tir_module(mod=None):
    from tvm import relax
    from tvm.relax import vm_build
    import tvm

    mod = mod or import_canonical_resnet18_relax_module()
    with tvm.target.Target({"kind": "typhoon"}):
        lowered = relax.get_pipeline("default")(mod)
    builder = relax.ExecBuilder()
    return vm_build._vmcodegen(builder, lowered, exec_mode="bytecode")


def build_canonical_resnet18_vm_executable(mod=None):
    from tvm import relax
    from tvm.relax import vm_build
    import tvm

    mod = mod or import_canonical_resnet18_relax_module()
    with tvm.target.Target({"kind": "typhoon"}):
        lowered = relax.get_pipeline("default")(mod)
    builder = relax.ExecBuilder()
    tir_mod = vm_build._vmcodegen(builder, lowered, exec_mode="bytecode")
    lib = tvm.tirx.build(tir_mod, target=tvm.target.Target({"kind": "typhoon"}))
    source_text = lib.inspect_source()
    exe = vm_build.VMExecutable(
        vm_build._ffi_api.VMLink(builder, tvm.target.Target({"kind": "typhoon"}), lib, [], {})
    )
    return exe, source_text


def run_canonical_resnet18_on_typhoon(feed_dict=None, mod=None, executable=None):
    from tvm import relax
    import tvm

    mod = mod or import_canonical_resnet18_relax_module()
    feed_dict = feed_dict or build_canonical_resnet18_feed_dict(load_canonical_resnet18_model())
    executable = executable or build_canonical_resnet18_vm_executable(mod)[0]
    vm = relax.VirtualMachine(_jit_executable_for_host_runtime(executable), tvm.cpu())
    vm.set_input("main", *_make_vm_inputs(mod, feed_dict))
    vm.invoke_stateful("main")
    return vm.get_outputs("main").numpy()


def run_canonical_resnet18_reference(feed_dict=None, mod=None):
    from tvm import relax
    import tvm

    mod = mod or import_canonical_resnet18_relax_module()
    feed_dict = feed_dict or build_canonical_resnet18_feed_dict(load_canonical_resnet18_model())
    ex = relax.build(mod, target="c")
    vm = relax.VirtualMachine(ex, tvm.cpu())
    vm.set_input("main", *_make_vm_inputs(mod, feed_dict))
    vm.invoke_stateful("main")
    return vm.get_outputs("main").numpy()


def run_canonical_resnet18_reference_onnxruntime(feed_dict=None, model=None):
    try:
        onnxruntime = importlib.import_module("onnxruntime")
    except ImportError as err:
        raise RuntimeError(
            "onnxruntime is required for Typhoon ResNet18 reference validation"
        ) from err

    model = model or load_canonical_resnet18_model()
    feed_dict = feed_dict or build_canonical_resnet18_feed_dict(model)
    initializer_names = {initializer.name for initializer in model.graph.initializer}
    ort_feed = {
        value_info.name: feed_dict[value_info.name]
        for value_info in model.graph.input
        if value_info.name not in initializer_names
    }
    session = onnxruntime.InferenceSession(
        model.SerializeToString(),
        providers=["CPUExecutionProvider"],
    )
    output_names = [value.name for value in model.graph.output]
    return session.run(output_names, ort_feed)[0]


def assert_canonical_resnet18_outputs_close(actual, expected, rtol=1e-4, atol=1e-4):
    assert_elementwise_close_with_context(actual, expected, rtol=rtol, atol=atol)


def format_first_mismatch(
    actual, expected, rtol=1e-5, atol=1e-5, layer_hint=None, task_hint=None
):
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    if actual.shape != expected.shape:
        parts = [f"shape mismatch: actual={actual.shape}, expected={expected.shape}"]
        if layer_hint is not None:
            parts.append(f"layer={layer_hint}")
        if task_hint is not None:
            parts.append(f"task={task_hint}")
        return ", ".join(parts)

    mismatch = ~np.isclose(actual, expected, rtol=rtol, atol=atol, equal_nan=True)
    if not np.any(mismatch):
        return "no mismatch"

    index = tuple(np.argwhere(mismatch)[0])
    actual_value = actual[index]
    expected_value = expected[index]
    diff = actual_value - expected_value
    message = (
        f"first mismatch at index {index}: actual={actual_value}, expected={expected_value}, "
        f"diff={diff}, rtol={rtol}, atol={atol}"
    )
    if layer_hint is not None:
        message += f", layer={layer_hint}"
    if task_hint is not None:
        message += f", task={task_hint}"
    return message


def assert_elementwise_close_with_context(
    actual, expected, rtol=1e-5, atol=1e-5, layer_hint=None, task_hint=None
):
    if not np.allclose(actual, expected, rtol=rtol, atol=atol, equal_nan=True):
        raise AssertionError(
            format_first_mismatch(
                actual,
                expected,
                rtol=rtol,
                atol=atol,
                layer_hint=layer_hint,
                task_hint=task_hint,
            )
        )
