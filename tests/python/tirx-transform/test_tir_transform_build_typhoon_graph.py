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


def _make_stem_body(input_buffer, weight_buffer, output_buffer):
    oc = tvm.tirx.Var("oc", "int32")
    oh = tvm.tirx.Var("oh", "int32")
    ow = tvm.tirx.Var("ow", "int32")
    store = tvm.tirx.BufferStore(
        output_buffer,
        tvm.tirx.BufferLoad(input_buffer, [0, 0, oh * 2, ow * 2])
        + tvm.tirx.BufferLoad(weight_buffer, [oc, 0, 0, 0]),
        [0, oc, oh, ow],
    )
    return tvm.tirx.For(
        oc,
        tvm.tirx.const(0, "int32"),
        tvm.tirx.const(64, "int32"),
        tvm.tirx.ForKind.SERIAL,
        tvm.tirx.For(
            oh,
            tvm.tirx.const(0, "int32"),
            tvm.tirx.const(112, "int32"),
            tvm.tirx.ForKind.SERIAL,
            tvm.tirx.For(
                ow,
                tvm.tirx.const(0, "int32"),
                tvm.tirx.const(112, "int32"),
                tvm.tirx.ForKind.SERIAL,
                store,
            ),
        ),
    )


def build_resnet18_stem_tir_module():
    input_buffer = tvm.tirx.decl_buffer((1, 3, 224, 224), "float32", name="input")
    weight_buffer = tvm.tirx.decl_buffer((64, 3, 7, 7), "float32", name="weight")
    output_buffer = tvm.tirx.decl_buffer((1, 64, 112, 112), "float32", name="output")
    func = tvm.tirx.PrimFunc(
        [input_buffer, weight_buffer, output_buffer],
        _make_stem_body(input_buffer, weight_buffer, output_buffer),
    ).with_attr("target", tvm.target.Target({"kind": "typhoon"}))
    return tvm.IRModule.from_expr(func)


def build_planned_resnet18_conv_block():
    input_buffer = tvm.tirx.decl_buffer((1, 3, 224, 224), "float32", name="input")
    weight_buffer = tvm.tirx.decl_buffer((64, 3, 7, 7), "float32", name="weight")
    output_buffer = tvm.tirx.decl_buffer((1, 64, 112, 112), "float32", name="output")
    func = tvm.tirx.PrimFunc(
        [input_buffer, weight_buffer, output_buffer],
        tvm.tirx.Evaluate(0),
    ).with_attr("target", tvm.target.Target({"kind": "typhoon"}))
    mod = tvm.IRModule.from_expr(func)
    mod = mod.with_attr(
        "typhoon_resnet18_plan",
        json.dumps(
            {
                "model": "resnet18",
                "recognized_scope": "stem",
                "input_shape": [1, 3, 224, 224],
                "dtype": "float32",
                "layers": [
                    {
                        "layer_id": 0,
                        "stage_id": 0,
                        "block_id": 0,
                        "kind": "conv2d",
                        "op_name": "stem_conv",
                        "logical_input_shape": [1, 3, 224, 224],
                        "logical_output_shape": [1, 64, 112, 112],
                        "weight_shape": [64, 3, 7, 7],
                        "requires_im2col": True,
                        "preferred_output_layout": "zZ",
                    }
                ],
            }
        ),
    )
    return mod.with_attr(
        "typhoon_sram_plan",
        json.dumps(
            {
                "matmul_tile": [64, 64, 64],
                "tile": {"Mt": 64, "Nt": 64, "Kt": 64, "m0": 8, "n0": 8, "k0": 8},
                "regions": [
                    {"region_id": 0, "name": "act0", "offset": 0, "size": 262144, "alignment": 64},
                    {
                        "region_id": 1,
                        "name": "act1",
                        "offset": 262144,
                        "size": 262144,
                        "alignment": 64,
                    },
                    {
                        "region_id": 2,
                        "name": "wgt0",
                        "offset": 524288,
                        "size": 65536,
                        "alignment": 64,
                    },
                    {
                        "region_id": 3,
                        "name": "wgt1",
                        "offset": 589824,
                        "size": 65536,
                        "alignment": 64,
                    },
                    {
                        "region_id": 4,
                        "name": "col0",
                        "offset": 655360,
                        "size": 131072,
                        "alignment": 64,
                    },
                    {
                        "region_id": 5,
                        "name": "aux0",
                        "offset": 786432,
                        "size": 65536,
                        "alignment": 64,
                    },
                ],
                "layer_tiles": [
                    {
                        "layer_id": 0,
                        "tile_id": 0,
                        "reads": ["act0", "wgt0"],
                        "writes": ["col0", "aux0"],
                    }
                ],
            }
        ),
    )


def test_build_typhoon_graph_emits_region_decl_and_tasks():
    mod = build_planned_resnet18_conv_block()
    out = tvm.tirx.transform.BuildTyphoonGraph()(mod)
    text = out.script()
    assert "T.typhoon.region_decl" in text
    assert "T.typhoon.task_matmul" in text
    assert "T.typhoon.submit_graph" in text


def test_finalize_host_passes_auto_builds_typhoon_graph_for_resnet18_stem():
    mod = build_resnet18_stem_tir_module()
    out = tvm.tirx.pipeline.finalize_host_passes()(mod)
    text = out.script()
    assert "typhoon_resnet18_plan" in out.attrs
    assert "typhoon_sram_plan" in out.attrs
    assert "TVMTyphoonDeclareRegion" in text
    assert "TVMTyphoonAddMatmulTask" in text
    assert "TVMTyphoonGraphBegin" in text
    assert "TVMTyphoonSubmitGraph" in text
