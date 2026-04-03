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

import tvm


def test_lower_typhoon_region_decl_to_runtime_call():
    stmt = tvm.tirx.typhoon.region_decl(7, 0, 0, 4096, 64, 0, "A")
    func = tvm.tirx.PrimFunc([], stmt).with_attr("target", tvm.target.Target({"kind": "typhoon"}))
    mod = tvm.IRModule.from_expr(func)
    lowered = tvm.tirx.transform.LowerIntrin()(mod)["main"].body
    assert "TVMTyphoonDeclareRegion" in str(lowered)


def test_lower_typhoon_submit_graph_expands_graph_begin_then_submit():
    stmt = tvm.tirx.typhoon.submit_graph(7)
    func = tvm.tirx.PrimFunc([], stmt).with_attr("target", tvm.target.Target({"kind": "typhoon"}))
    mod = tvm.IRModule.from_expr(func)
    lowered = tvm.transform.Sequential(
        [tvm.tirx.transform.LowerTyphoonSubmitGraph(), tvm.tirx.transform.LowerIntrin()]
    )(mod)["main"].body
    lowered_text = str(lowered)
    assert "TVMTyphoonGraphBegin" in lowered_text
    assert "TVMTyphoonSubmitGraph" in lowered_text
    assert lowered_text.index("TVMTyphoonGraphBegin") < lowered_text.index("TVMTyphoonSubmitGraph")


def test_lower_typhoon_submit_graph_is_idempotent():
    stmt = tvm.tirx.typhoon.submit_graph(7)
    func = tvm.tirx.PrimFunc([], stmt).with_attr("target", tvm.target.Target({"kind": "typhoon"}))
    mod = tvm.IRModule.from_expr(func)
    lowered = tvm.transform.Sequential(
        [
            tvm.tirx.transform.LowerTyphoonSubmitGraph(),
            tvm.tirx.transform.LowerTyphoonSubmitGraph(),
            tvm.tirx.transform.LowerIntrin(),
        ]
    )(mod)["main"].body
    lowered_text = str(lowered)
    assert lowered_text.count("TVMTyphoonGraphBegin") == 1
    assert lowered_text.count("TVMTyphoonSubmitGraph") == 1


def test_lower_typhoon_task_matmul_materializes_dep_array():
    stmt = tvm.tirx.typhoon.task_matmul(7, 1, 0, 1, 2, 64, 64, 64, 1, 0, [4, 9])
    func = tvm.tirx.PrimFunc([], stmt).with_attr("target", tvm.target.Target({"kind": "typhoon"}))
    mod = tvm.IRModule.from_expr(func)
    lowered = tvm.transform.Sequential(
        [tvm.tirx.transform.LowerTyphoonTaskDeps(), tvm.tirx.transform.LowerIntrin()]
    )(mod)["main"].body
    lowered_text = str(lowered)
    assert "TVMTyphoonAddMatmulTask" in lowered_text
    assert "tirx.tvm_stack_alloca" in lowered_text


def test_finalize_host_passes_lower_typhoon_runtime_abi():
    stmt = tvm.tirx.stmt_seq(
        tvm.tirx.typhoon.region_decl(7, 0, 0, 4096, 64, 1, "A"),
        tvm.tirx.typhoon.region_decl(7, 1, 4096, 4096, 64, 1, "B"),
        tvm.tirx.typhoon.region_decl(7, 2, 8192, 4096, 64, 0, "C"),
        tvm.tirx.typhoon.task_matmul(7, 1, 0, 1, 2, 64, 64, 64, 1, 0, []),
        tvm.tirx.typhoon.submit_graph(7),
    )
    func = tvm.tirx.PrimFunc([], stmt).with_attr("target", tvm.target.Target({"kind": "typhoon"}))
    mod = tvm.IRModule.from_expr(func)
    lowered = tvm.tirx.pipeline.finalize_host_passes()(mod)["main"].body
    lowered_text = str(lowered)
    assert "TVMTyphoonAddMatmulTask" in lowered_text
    assert "TVMTyphoonGraphBegin" in lowered_text
    assert "TVMTyphoonSubmitGraph" in lowered_text
