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

import tvm

from tests.python.tirx_transform.typhoon_resnet18_test_utils import (
    build_canonical_resnet18_tir_module,
)


def _bind_typhoon_target(mod):
    target = tvm.target.Target({"kind": "typhoon"})
    functions = {}
    for gvar, func in mod.functions.items():
        if isinstance(func, tvm.tirx.PrimFunc):
            functions[gvar] = func.with_attr("target", target)
        else:
            functions[gvar] = func
    return tvm.IRModule(functions, attrs=mod.attrs)


def test_typhoon_resnet18_recognizes_full_graph_scope():
    mod = _bind_typhoon_target(build_canonical_resnet18_tir_module())
    out = tvm.tirx.transform.IdentifyTyphoonResNet18()(mod)
    plan = json.loads(out.attrs["typhoon_resnet18_plan"])
    assert {"model", "recognized_scope", "input_shape", "dtype", "stages", "blocks", "layers", "edges"} <= set(
        plan.keys()
    )
    assert plan["recognized_scope"] == "full_graph"
    assert plan["model"] == "resnet18"
    assert plan["input_shape"] == [1, 3, 224, 224]
    assert plan["dtype"] == "float32"
    assert len(plan["layers"]) >= 29
    assert [stage["stage_id"] for stage in plan["stages"]] == [0, 1, 2, 3]
    assert [block["block_id"] for block in plan["blocks"][:2]] == [0, 1]
    layer_ids = [layer["layer_id"] for layer in plan["layers"]]
    assert len(layer_ids) == len(set(layer_ids))
    first = plan["layers"][0]
    assert {"layer_id", "kind", "logical_input_shape", "logical_output_shape"} <= first.keys()
    assert all(
        {"layer_id", "stage_id", "block_id", "kind", "logical_input_shape", "logical_output_shape"}
        <= set(layer.keys())
        for layer in plan["layers"]
    )
    assert any("layout_transform" in layer for layer in plan["layers"])
    assert any(layer.get("requires_im2col", False) for layer in plan["layers"])
    assert any(edge.get("edge_kind") == "residual" for edge in plan["edges"])
    assert all(edge["src_layer_id"] in layer_ids and edge["dst_layer_id"] in layer_ids for edge in plan["edges"])
    assert [layer["kind"] for layer in plan["layers"][-3:]] == [
        "global_avg_pool",
        "flatten",
        "dense",
    ]
