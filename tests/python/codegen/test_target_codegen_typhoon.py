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

import numpy as np
import pytest

import tvm
import tvm.testing
from tvm.runtime import Executable


def _reset_typhoon_runtime():
    reset = tvm.get_global_func("runtime.typhoon.testing_reset", allow_missing=True)
    assert reset is not None
    reset()


def _get_typhoon_trace():
    get_trace = tvm.get_global_func("runtime.typhoon_get_last_trace_json", allow_missing=True)
    assert get_trace is not None
    return json.loads(get_trace())


def _jit_module(mod):
    _reset_typhoon_runtime()
    lib = tvm.tirx.build(mod, target={"kind": "typhoon"})
    return Executable(lib).jit()


def _tensor(array):
    return tvm.runtime.tensor(np.array(array))


def _graph_module(params, *stmts):
    func = tvm.tirx.PrimFunc(params, tvm.tirx.stmt_seq(*stmts)).with_attr("global_symbol", "main")
    return tvm.IRModule.from_expr(func)


def _build_dma_matmul_dma_module():
    a = tvm.tirx.decl_buffer((2, 2), "float32", name="A")
    b = tvm.tirx.decl_buffer((2, 2), "float32", name="B")
    c = tvm.tirx.decl_buffer((2, 2), "float32", name="C")
    return _graph_module(
        [a, b, c],
        tvm.tirx.typhoon.region_decl(7, 0, 0, 16, 16, 0, "lhs"),
        tvm.tirx.typhoon.region_decl(7, 1, 16, 16, 16, 0, "rhs"),
        tvm.tirx.typhoon.region_decl(7, 2, 32, 16, 16, 0, "out"),
        tvm.tirx.typhoon.task_dma(7, 1, 0, a.data, 0, 0, 16, []),
        tvm.tirx.typhoon.task_dma(7, 2, 0, b.data, 0, 1, 16, []),
        tvm.tirx.typhoon.task_matmul(7, 3, 0, 1, 2, 2, 2, 2, 2, 0, [1, 2]),
        tvm.tirx.typhoon.task_dma(7, 4, 1, c.data, 0, 2, 16, [3]),
        tvm.tirx.typhoon.submit_graph(7),
        tvm.tirx.typhoon.wait_graph(7),
    )


def _build_overlap_trace_module():
    d = tvm.tirx.decl_buffer((4,), "float32", name="D")
    return _graph_module(
        [d],
        tvm.tirx.typhoon.region_decl(9, 0, 0, 16, 16, 1, "vec_in0"),
        tvm.tirx.typhoon.region_decl(9, 1, 16, 16, 16, 1, "vec_in1"),
        tvm.tirx.typhoon.region_decl(9, 2, 32, 16, 16, 0, "vec_out"),
        tvm.tirx.typhoon.region_decl(9, 3, 48, 16, 16, 0, "dma_out"),
        tvm.tirx.typhoon.task_vector(9, 1, 0, 0, 1, 2, 4, 2, []),
        tvm.tirx.typhoon.task_dma(9, 2, 0, d.data, 0, 3, 16, []),
        tvm.tirx.typhoon.submit_graph(9),
        tvm.tirx.typhoon.wait_graph(9),
    )


def _build_mixed_module():
    a = tvm.tirx.decl_buffer((4,), "float32", name="A")
    b = tvm.tirx.decl_buffer((4,), "float32", name="B")
    c = tvm.tirx.decl_buffer((2, 2), "float32", name="C")
    d = tvm.tirx.decl_buffer((2, 2), "float32", name="D")
    return _graph_module(
        [a, b, c, d],
        tvm.tirx.typhoon.region_decl(11, 0, 0, 16, 16, 0, "vec_lhs"),
        tvm.tirx.typhoon.region_decl(11, 1, 16, 16, 16, 0, "vec_rhs"),
        tvm.tirx.typhoon.region_decl(11, 2, 32, 16, 16, 0, "vec_out"),
        tvm.tirx.typhoon.region_decl(11, 3, 48, 16, 16, 0, "reshape_out"),
        tvm.tirx.typhoon.region_decl(11, 4, 64, 16, 16, 0, "mat_rhs"),
        tvm.tirx.typhoon.region_decl(11, 5, 80, 16, 16, 0, "mat_out"),
        tvm.tirx.typhoon.task_dma(11, 1, 0, a.data, 0, 0, 16, []),
        tvm.tirx.typhoon.task_dma(11, 2, 0, b.data, 0, 1, 16, []),
        tvm.tirx.typhoon.task_vector(11, 3, 0, 0, 1, 2, 4, 2, [1, 2]),
        tvm.tirx.typhoon.task_reshape(11, 4, 2, 3, 16, 0, [3]),
        tvm.tirx.typhoon.task_dma(11, 5, 0, c.data, 0, 4, 16, []),
        tvm.tirx.typhoon.task_matmul(11, 6, 3, 4, 5, 2, 2, 2, 2, 0, [4, 5]),
        tvm.tirx.typhoon.task_dma(11, 7, 1, d.data, 0, 5, 16, [6]),
        tvm.tirx.typhoon.submit_graph(11),
        tvm.tirx.typhoon.wait_graph(11),
    )


def _build_invalid_sram_module():
    a = tvm.tirx.decl_buffer((4,), "float32", name="A")
    return _graph_module(
        [a],
        tvm.tirx.typhoon.region_decl(13, 0, 0, 16, 16, 0, "lhs"),
        tvm.tirx.typhoon.region_decl(13, 1, 8, 16, 8, 0, "overlap"),
        tvm.tirx.typhoon.task_dma(13, 1, 0, a.data, 0, 0, 16, []),
        tvm.tirx.typhoon.submit_graph(13),
        tvm.tirx.typhoon.wait_graph(13),
    )


def test_typhoon_dma_matmul_dma_roundtrip():
    mod = _build_dma_matmul_dma_module()
    fn = _jit_module(mod)["main"]

    lhs = np.array([[1.0, 2.0], [3.0, 4.0]], dtype="float32")
    rhs = np.array([[5.0, 6.0], [7.0, 8.0]], dtype="float32")
    out = _tensor(np.zeros((2, 2), dtype="float32"))

    fn(_tensor(lhs), _tensor(rhs), out)

    tvm.testing.assert_allclose(out.numpy(), lhs @ rhs)


def test_typhoon_dma_vector_overlap_trace():
    mod = _build_overlap_trace_module()
    fn = _jit_module(mod)["main"]

    fn(_tensor(np.array([1.0, 2.0, 3.0, 4.0], dtype="float32")))

    trace = _get_typhoon_trace()
    dma = next(record for record in trace if record["kind"] == "dma")
    vector = next(record for record in trace if record["kind"] == "vector")
    assert dma["start_time"] < vector["end_time"]
    assert vector["start_time"] < dma["end_time"]


def test_typhoon_mixed_reshape_vector_matmul_roundtrip():
    mod = _build_mixed_module()
    fn = _jit_module(mod)["main"]

    lhs = np.array([1.0, 2.0, 3.0, 4.0], dtype="float32")
    rhs = np.array([10.0, 20.0, 30.0, 40.0], dtype="float32")
    mat_rhs = np.array([[2.0, 0.0], [0.0, 2.0]], dtype="float32")
    out = _tensor(np.zeros((2, 2), dtype="float32"))

    fn(_tensor(lhs), _tensor(rhs), _tensor(mat_rhs), out)

    expected = (lhs + rhs).reshape(2, 2) @ mat_rhs
    tvm.testing.assert_allclose(out.numpy(), expected)


def test_typhoon_invalid_sram_usage_raises_clear_error():
    with pytest.raises(ValueError, match="overlap|bounds"):
        _jit_module(_build_invalid_sram_module())
