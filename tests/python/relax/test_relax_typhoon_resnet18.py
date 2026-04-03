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

import numpy as np
import onnx
import tvm
import tvm.testing
from onnx.numpy_helper import to_array
from tvm import relax
from tvm.relax import vm_build
from tvm.relax.frontend.onnx import from_onnx

TYPHOON_RESNET18_ENV = "TYPHOON_RESNET18_ONNX_PATH"
DEFAULT_RESNET18_PATH = os.path.expanduser("~/model/resnet18.onnx")


def get_resnet18_model_path():
    model_path = os.environ.get(TYPHOON_RESNET18_ENV, DEFAULT_RESNET18_PATH)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"ResNet18 model not found at {model_path}")
    return model_path


def _load_onnx_model(model_path):
    return onnx.load(model_path)


def _build_feed_dict(model):
    input_name = model.graph.input[0].name
    feed = {input_name: np.random.default_rng(0).standard_normal((1, 3, 224, 224), dtype=np.float32)}
    for initializer in model.graph.initializer:
        feed[initializer.name.replace("onnx::", "")] = to_array(initializer).astype("float32")
    return feed


def _import_relax(model):
    mod = from_onnx(model, keep_params_in_input=True)
    mod = relax.transform.DecomposeOpsForInference()(mod)
    mod = relax.transform.LegalizeOps()(mod)
    return mod


def _make_vm_inputs(mod, feed_dict):
    return [feed_dict[param.name_hint] for param in mod["main"].params]


def _build_typhoon_vmexec(mod):
    with tvm.target.Target({"kind": "typhoon"}):
        lowered = relax.get_pipeline("default")(mod)
    builder = relax.ExecBuilder()
    tir_mod = vm_build._vmcodegen(builder, lowered, exec_mode="bytecode")
    tir_mod = tvm.tirx.transform.BindTarget(tvm.target.Target({"kind": "typhoon"}))(tir_mod)
    finalized = tvm.tirx.pipeline.finalize_host_passes()(tir_mod)
    source_text = finalized.script()

    builder = relax.ExecBuilder()
    tir_mod = vm_build._vmcodegen(builder, lowered, exec_mode="bytecode")
    lib = tvm.tirx.build(tir_mod, target=tvm.target.Target({"kind": "typhoon"}))
    exe = vm_build.VMExecutable(
        vm_build._ffi_api.VMLink(builder, tvm.target.Target({"kind": "typhoon"}), lib, [], {})
    )
    return exe, source_text


def _run_vm(exe, mod, feed_dict):
    vm = relax.VirtualMachine(exe, tvm.cpu())
    vm.set_input("main", *_make_vm_inputs(mod, feed_dict))
    vm.invoke_stateful("main")
    return vm.get_outputs("main").numpy()


def _run_reference(mod, feed_dict):
    ex = relax.build(mod, target="c")
    vm = relax.VirtualMachine(ex, tvm.cpu())
    vm.set_input("main", *_make_vm_inputs(mod, feed_dict))
    vm.invoke_stateful("main")
    return vm.get_outputs("main").numpy()


def test_relax_resnet18_compiles_to_typhoon_graph():
    model = _load_onnx_model(get_resnet18_model_path())
    mod = _import_relax(model)
    ex, source = _build_typhoon_vmexec(mod)
    assert isinstance(ex, vm_build.VMExecutable)
    assert "TVMTyphoonAddMatmulTask" in source
    assert "TVMTyphoonSubmitGraph" in source


def test_relax_resnet18_runs_in_typhoon_simulator():
    model = _load_onnx_model(get_resnet18_model_path())
    mod = _import_relax(model)
    feed_dict = _build_feed_dict(model)
    ex, source = _build_typhoon_vmexec(mod)
    output = _run_vm(ex, mod, feed_dict)
    ref = _run_reference(mod, feed_dict)
    assert "TVMTyphoonAddMatmulTask" in source
    assert "TVMTyphoonSubmitGraph" in source
    tvm.testing.assert_allclose(output, ref, rtol=1e-4, atol=1e-4)
