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

import pytest
import tvm


GRAPH_ID = 7


def _global_handle():
    return tvm.tirx.Var("global_handle", "handle")


def _make_mod(*stmts):
    body = tvm.tirx.stmt_seq(*stmts)
    func = tvm.tirx.PrimFunc([], body).with_attr("target", tvm.target.Target({"kind": "typhoon"}))
    return tvm.IRModule.from_expr(func)


def _verify(mod):
    return tvm.tirx.transform.VerifyTyphoonGraph()(mod)


def test_verify_typhoon_graph_is_exported_to_python():
    assert callable(tvm.tirx.transform.VerifyTyphoonGraph)


def test_verify_typhoon_graph_rejects_overlapping_regions():
    mod = _make_mod(
        tvm.tirx.typhoon.region_decl(GRAPH_ID, 0, 0, 256, 64, 1, "input"),
        tvm.tirx.typhoon.region_decl(GRAPH_ID, 1, 128, 256, 64, 0, "output"),
    )
    with pytest.raises(ValueError, match="overlap"):
        _verify(mod)


def test_verify_typhoon_graph_rejects_duplicate_region_ids():
    mod = _make_mod(
        tvm.tirx.typhoon.region_decl(GRAPH_ID, 0, 0, 256, 64, 1, "input0"),
        tvm.tirx.typhoon.region_decl(GRAPH_ID, 0, 256, 256, 64, 0, "input1"),
    )
    with pytest.raises(ValueError, match="region_id"):
        _verify(mod)


def test_verify_typhoon_graph_rejects_misaligned_regions():
    mod = _make_mod(tvm.tirx.typhoon.region_decl(GRAPH_ID, 0, 32, 256, 64, 1, "input"))
    with pytest.raises(ValueError, match="alignment"):
        _verify(mod)


def test_verify_typhoon_graph_rejects_regions_out_of_sram_bounds():
    mod = _make_mod(tvm.tirx.typhoon.region_decl(GRAPH_ID, 0, 1048448, 512, 64, 1, "input"))
    with pytest.raises(ValueError, match="bounds"):
        _verify(mod)


def test_verify_typhoon_graph_rejects_duplicate_task_ids():
    mod = _make_mod(
        tvm.tirx.typhoon.region_decl(GRAPH_ID, 0, 0, 256, 64, 1, "input"),
        tvm.tirx.typhoon.region_decl(GRAPH_ID, 1, 256, 256, 64, 0, "output0"),
        tvm.tirx.typhoon.region_decl(GRAPH_ID, 2, 512, 256, 64, 0, "output1"),
        tvm.tirx.typhoon.task_reshape(GRAPH_ID, 1, 0, 1, 64, 0, []),
        tvm.tirx.typhoon.task_reshape(GRAPH_ID, 1, 0, 2, 64, 0, []),
    )
    with pytest.raises(ValueError, match="task_id"):
        _verify(mod)


def test_verify_typhoon_graph_rejects_mixed_graph_ids():
    mod = _make_mod(
        tvm.tirx.typhoon.region_decl(GRAPH_ID, 0, 0, 256, 64, 1, "input"),
        tvm.tirx.typhoon.region_decl(GRAPH_ID, 1, 256, 256, 64, 0, "output"),
        tvm.tirx.typhoon.task_reshape(GRAPH_ID + 1, 1, 0, 1, 64, 0, []),
    )
    with pytest.raises(ValueError, match="graph_id"):
        _verify(mod)


def test_verify_typhoon_graph_rejects_unknown_dependency():
    mod = _make_mod(
        tvm.tirx.typhoon.region_decl(GRAPH_ID, 0, 0, 256, 64, 1, "input"),
        tvm.tirx.typhoon.region_decl(GRAPH_ID, 1, 256, 256, 64, 0, "output"),
        tvm.tirx.typhoon.task_reshape(GRAPH_ID, 1, 0, 1, 64, 0, [99]),
    )
    with pytest.raises(ValueError, match="dependency"):
        _verify(mod)


def test_verify_typhoon_graph_rejects_unknown_region_reference():
    mod = _make_mod(
        tvm.tirx.typhoon.region_decl(GRAPH_ID, 1, 256, 256, 64, 0, "output"),
        tvm.tirx.typhoon.task_reshape(GRAPH_ID, 1, 0, 1, 64, 0, []),
    )
    with pytest.raises(ValueError, match="unknown input region_id"):
        _verify(mod)


def test_verify_typhoon_graph_rejects_dependency_cycles():
    mod = _make_mod(
        tvm.tirx.typhoon.region_decl(GRAPH_ID, 0, 0, 256, 64, 1, "input"),
        tvm.tirx.typhoon.region_decl(GRAPH_ID, 1, 256, 256, 64, 0, "output0"),
        tvm.tirx.typhoon.region_decl(GRAPH_ID, 2, 512, 256, 64, 0, "output1"),
        tvm.tirx.typhoon.task_reshape(GRAPH_ID, 1, 0, 1, 64, 0, [2]),
        tvm.tirx.typhoon.task_reshape(GRAPH_ID, 2, 0, 2, 64, 0, [1]),
    )
    with pytest.raises(ValueError, match="cycle"):
        _verify(mod)


def test_verify_typhoon_graph_rejects_compute_operands_that_are_not_region_ids():
    mod = _make_mod(
        tvm.tirx.typhoon.region_decl(GRAPH_ID, 1, 256, 256, 64, 0, "output"),
        tvm.tirx.typhoon.task_reshape(GRAPH_ID, 1, _global_handle(), 1, 64, 0, []),
    )
    with pytest.raises(ValueError, match="region_id"):
        _verify(mod)


def test_verify_typhoon_graph_rejects_malformed_task_arity():
    mod = _make_mod(
        tvm.tirx.Evaluate(
            tvm.tirx.call_intrin("int32", "tirx.typhoon.task_reshape", GRAPH_ID, 1, 0, 1)
        )
    )
    with pytest.raises(ValueError, match="missing required operands"):
        _verify(mod)


def test_verify_typhoon_graph_rejects_read_before_init():
    mod = _make_mod(
        tvm.tirx.typhoon.region_decl(GRAPH_ID, 0, 0, 256, 64, 0, "input"),
        tvm.tirx.typhoon.region_decl(GRAPH_ID, 1, 256, 256, 64, 0, "output"),
        tvm.tirx.typhoon.task_reshape(GRAPH_ID, 1, 0, 1, 64, 0, []),
    )
    with pytest.raises(ValueError, match="initialized"):
        _verify(mod)


def test_verify_typhoon_graph_accepts_global_to_sram_dma_before_compute_read():
    mod = _make_mod(
        tvm.tirx.typhoon.region_decl(GRAPH_ID, 0, 0, 256, 64, 0, "input"),
        tvm.tirx.typhoon.region_decl(GRAPH_ID, 1, 256, 256, 64, 0, "output"),
        tvm.tirx.typhoon.task_dma(GRAPH_ID, 1, 0, _global_handle(), 0, 0, 256, []),
        tvm.tirx.typhoon.task_reshape(GRAPH_ID, 2, 0, 1, 64, 0, [1]),
        tvm.tirx.typhoon.submit_graph(GRAPH_ID),
    )
    _verify(mod)


def test_verify_typhoon_graph_accepts_preinitialized_region_reads():
    mod = _make_mod(
        tvm.tirx.typhoon.region_decl(GRAPH_ID, 0, 0, 256, 64, 1, "input"),
        tvm.tirx.typhoon.region_decl(GRAPH_ID, 1, 256, 256, 64, 0, "output"),
        tvm.tirx.typhoon.task_reshape(GRAPH_ID, 1, 0, 1, 64, 0, []),
        tvm.tirx.typhoon.submit_graph(GRAPH_ID),
    )
    _verify(mod)


def test_verify_typhoon_graph_rejects_unordered_writers_to_same_region():
    mod = _make_mod(
        tvm.tirx.typhoon.region_decl(GRAPH_ID, 0, 0, 256, 64, 1, "input"),
        tvm.tirx.typhoon.region_decl(GRAPH_ID, 1, 256, 256, 64, 0, "output"),
        tvm.tirx.typhoon.task_reshape(GRAPH_ID, 1, 0, 1, 64, 0, []),
        tvm.tirx.typhoon.task_reshape(GRAPH_ID, 2, 0, 1, 64, 0, []),
    )
    with pytest.raises(ValueError, match="write hazard"):
        _verify(mod)


def test_verify_typhoon_graph_accepts_im2col_reshape_payload():
    mod = _make_mod(
        tvm.tirx.typhoon.region_decl(GRAPH_ID, 0, 0, 64, 64, 1, "input"),
        tvm.tirx.typhoon.region_decl(GRAPH_ID, 1, 64, 64, 64, 0, "output"),
        tvm.tirx.typhoon.task_reshape(
            GRAPH_ID,
            1,
            0,
            1,
            64,
            1,
            extra_shape_metadata=[1, 1, 3, 3, 2, 2, 1, 1, 0, 0, 2, 2],
            deps=[],
        ),
        tvm.tirx.typhoon.submit_graph(GRAPH_ID),
    )
    _verify(mod)


def test_verify_typhoon_graph_rejects_im2col_reshape_payload_shape_mismatch():
    mod = _make_mod(
        tvm.tirx.typhoon.region_decl(GRAPH_ID, 0, 0, 64, 64, 1, "input"),
        tvm.tirx.typhoon.region_decl(GRAPH_ID, 1, 64, 64, 64, 0, "output"),
        tvm.tirx.typhoon.task_reshape(
            GRAPH_ID,
            1,
            0,
            1,
            64,
            1,
            extra_shape_metadata=[1, 1, 3, 3, 2, 2, 1, 1, 0, 0, 3, 2],
            deps=[],
        ),
        tvm.tirx.typhoon.submit_graph(GRAPH_ID),
    )
    with pytest.raises(ValueError, match="im2col|shape|metadata"):
        _verify(mod)


def test_verify_typhoon_graph_rejects_maxpool_payload_shape_mismatch():
    mod = _make_mod(
        tvm.tirx.typhoon.region_decl(GRAPH_ID, 0, 0, 256, 64, 1, "input"),
        tvm.tirx.typhoon.region_decl(GRAPH_ID, 1, 256, 256, 64, 0, "output"),
        tvm.tirx.typhoon.task_vector(
            GRAPH_ID,
            1,
            2,
            0,
            0,
            1,
            5,
            2,
            extra_window_metadata=[1, 1, 4, 4, 3, 3, 2, 2, 1, 1, 2, 2],
            deps=[],
        ),
        tvm.tirx.typhoon.submit_graph(GRAPH_ID),
    )
    with pytest.raises(ValueError, match="maxpool|elem_count|metadata"):
        _verify(mod)


def test_verify_typhoon_graph_accepts_global_average_pool_payload():
    mod = _make_mod(
        tvm.tirx.typhoon.region_decl(GRAPH_ID, 0, 0, 64, 64, 1, "input"),
        tvm.tirx.typhoon.region_decl(GRAPH_ID, 1, 64, 64, 64, 0, "output"),
        tvm.tirx.typhoon.task_vector(
            GRAPH_ID,
            1,
            3,
            0,
            -1,
            1,
            2,
            2,
            extra_window_metadata=[1, 2, 2, 2],
            deps=[],
        ),
        tvm.tirx.typhoon.submit_graph(GRAPH_ID),
    )
    _verify(mod)


def test_finalize_host_passes_runs_verify_typhoon_graph_before_lowering():
    mod = _make_mod(
        tvm.tirx.typhoon.region_decl(GRAPH_ID, 0, 0, 256, 64, 0, "input"),
        tvm.tirx.typhoon.region_decl(GRAPH_ID, 1, 256, 256, 64, 0, "output"),
        tvm.tirx.typhoon.task_reshape(GRAPH_ID, 1, 0, 1, 64, 0, []),
    )
    with pytest.raises(ValueError, match="initialized"):
        tvm.tirx.pipeline.finalize_host_passes()(mod)
