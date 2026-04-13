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
import os
import tempfile

import numpy as np

import tvm
from tvm.contrib import cc
from tests.python.tirx_transform.typhoon_resnet18_test_utils import (
    build_targeted_canonical_resnet18_tir_module,
)


def test_identify_typhoon_graph_builds_generic_graph_plan_from_fused_resnet18():
    mod = build_targeted_canonical_resnet18_tir_module()
    out = tvm.tirx.transform.IdentifyTyphoonGraph()(mod)
    assert "typhoon_graph_plan" in out.attrs

    plan = json.loads(out.attrs["typhoon_graph_plan"])
    assert plan["version"] == 1
    assert plan["target"] == "typhoon"

    families = []
    symbols = []
    for graph in plan["graphs"]:
        for node in graph["nodes"]:
            families.append(node["op_family"])
            symbols.append(node["symbol"])

    assert "conv_bias_relu" in families
    assert "conv_bias_residual_relu" in families
    assert "matmul_bias" in families
    assert "fused_conv2d_add_relu" in symbols
    assert "fused_matmul_add9" in symbols


def test_build_typhoon_graph_lowers_fused_semantic_functions_from_generic_plan():
    mod = build_targeted_canonical_resnet18_tir_module()
    mod = tvm.tirx.transform.IdentifyTyphoonGraph()(mod)
    out = tvm.tirx.transform.BuildTyphoonGraph()(mod)

    conv_text = out["fused_conv2d_add_relu"].script()
    assert "T.typhoon.task_matmul" in conv_text
    assert "T.typhoon.task_vector" in conv_text
    assert "T.typhoon.submit_graph" in conv_text

    dense_text = out["fused_matmul_add9"].script()
    assert "T.typhoon.task_matmul" in dense_text
    assert "T.typhoon.task_vector" in dense_text
    assert "T.typhoon.submit_graph" in dense_text


def _run_lowered_typhoon_primfunc(func, symbol):
    lib = tvm.tirx.build(tvm.IRModule.from_expr(func), target=tvm.target.Target({"kind": "typhoon"}))
    path = tempfile.mktemp(suffix=".so")
    lib.export_library(path, fcompile=cc.create_shared)
    try:
        return tvm.runtime.load_module(path)[symbol]
    finally:
        if os.path.exists(path):
            os.remove(path)


def test_build_typhoon_graph_preserves_numeric_behavior_for_fused_residual_conv():
    mod = build_targeted_canonical_resnet18_tir_module()
    lowered = tvm.tirx.transform.BuildTyphoonGraph()(tvm.tirx.transform.IdentifyTyphoonGraph()(mod))

    symbol = "fused_conv2d1_add1_add2_relu1"
    ref_func = tvm.compile(
        mod[symbol]
        .with_attr("target", tvm.target.Target("c"))
        .with_attr("global_symbol", "main"),
        "c",
    )["main"]
    typhoon_func = _run_lowered_typhoon_primfunc(lowered[symbol], symbol)

    rng = np.random.default_rng(0)
    args = []
    for buf in mod[symbol].buffer_map.values():
        shape = tuple(int(x) for x in buf.shape)
        args.append(rng.standard_normal(shape, dtype=np.float32))

    ref_args = [tvm.runtime.tensor(arg.copy()) for arg in args]
    typhoon_args = [tvm.runtime.tensor(arg.copy()) for arg in args]
    ref_func(*ref_args)
    typhoon_func(*typhoon_args)

    np.testing.assert_allclose(typhoon_args[-1].numpy(), ref_args[-1].numpy(), rtol=1e-5, atol=1e-5)
