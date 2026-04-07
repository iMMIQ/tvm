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
from tests.python.relax.typhoon_resnet18_test_utils import (
    build_canonical_resnet18_vm_tir_module,
)


def build_canonical_resnet18_tir_module():
    return build_canonical_resnet18_vm_tir_module()


def _bind_typhoon_target(mod):
    target = tvm.target.Target({"kind": "typhoon"})
    functions = {}
    for gvar, func in mod.functions.items():
        if isinstance(func, tvm.tirx.PrimFunc):
            functions[gvar] = func.with_attr("target", target)
        else:
            functions[gvar] = func
    return tvm.IRModule(functions, attrs=mod.attrs)


def build_targeted_canonical_resnet18_tir_module():
    return _bind_typhoon_target(build_canonical_resnet18_tir_module())


def build_resnet18_plan_module():
    return tvm.tirx.transform.IdentifyTyphoonResNet18()(build_targeted_canonical_resnet18_tir_module())


def build_full_graph_plan_module():
    return build_resnet18_plan_module()


def build_invalid_residual_plan_module():
    mod = build_full_graph_plan_module()
    plan = json.loads(mod.attrs["typhoon_resnet18_plan"])
    residual_edge = next(edge for edge in plan["edges"] if edge["edge_kind"] == "residual")
    residual_edge["src_layer_id"] = residual_edge["dst_layer_id"] + 3
    return mod.with_attr("typhoon_resnet18_plan", json.dumps(plan))
