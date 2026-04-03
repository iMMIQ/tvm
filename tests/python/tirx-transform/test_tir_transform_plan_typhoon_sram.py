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

import pytest
import tvm


def _make_mod(plan):
    func = tvm.tirx.PrimFunc([], tvm.tirx.Evaluate(0)).with_attr(
        "target", tvm.target.Target({"kind": "typhoon"})
    )
    mod = tvm.IRModule({"main": func})
    return mod.with_attr("typhoon_resnet18_plan", json.dumps(plan))


def build_resnet18_plan_module():
    return _make_mod(
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
    )


def build_artificial_large_layer_plan():
    return _make_mod(
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
                    "op_name": "oversized_conv",
                    "logical_input_shape": [1, 256, 512, 512],
                    "logical_output_shape": [1, 512, 256, 256],
                    "weight_shape": [512, 256, 3, 3],
                    "requires_im2col": True,
                    "preferred_output_layout": "zZ",
                }
            ],
        }
    )


def test_typhoon_sram_plan_uses_fixed_f32_tiles():
    mod = build_resnet18_plan_module()
    out = tvm.tirx.transform.PlanTyphoonSRAM()(mod)
    plan = json.loads(out.attrs["typhoon_sram_plan"])
    assert plan["matmul_tile"] == [64, 64, 64]
    assert plan["tile"]["m0"] == 8
    assert plan["regions"][0]["name"] == "act0"


def test_typhoon_sram_plan_rejects_tile_overflow():
    mod = build_artificial_large_layer_plan()
    with pytest.raises(ValueError, match="1MB|SRAM"):
        tvm.tirx.transform.PlanTyphoonSRAM()(mod)
