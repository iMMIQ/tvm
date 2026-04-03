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


def _make_mod(input_shape, output_shape):
    input_buffer = tvm.tirx.decl_buffer(input_shape, "float32", name="input")
    output_buffer = tvm.tirx.decl_buffer(output_shape, "float32", name="output")
    func = tvm.tirx.PrimFunc([input_buffer, output_buffer], tvm.tirx.Evaluate(0)).with_attr(
        "target", tvm.target.Target({"kind": "typhoon"})
    )
    return tvm.IRModule.from_expr(func)


def build_non_resnet18_tir_module():
    return _make_mod((1, 3, 128, 128), (1, 32, 64, 64))


def build_resnet18_stem_tir_module():
    return _make_mod((1, 3, 224, 224), (1, 64, 112, 112))


def test_typhoon_resnet18_rejects_non_resnet18_graph():
    mod = build_non_resnet18_tir_module()
    with pytest.raises(ValueError, match="ResNet18"):
        tvm.tirx.transform.IdentifyTyphoonResNet18()(mod)


def test_typhoon_resnet18_accepts_fixed_shape_resnet18_conv_stem():
    mod = build_resnet18_stem_tir_module()
    out = tvm.tirx.transform.IdentifyTyphoonResNet18()(mod)
    assert "typhoon_resnet18_plan" in out.attrs

    plan = json.loads(out.attrs["typhoon_resnet18_plan"])
    assert plan["model"] == "resnet18"
    assert plan["input_shape"] == [1, 3, 224, 224]
    assert plan["dtype"] == "float32"
    assert plan["layers"][0]["op_name"] == "stem_conv"
