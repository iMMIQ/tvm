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
import re

import numpy as np
import pytest

import tvm
import tvm.testing
from tvm.runtime import Executable
from tests.python.tirx_transform.typhoon_resnet18_test_utils import (
    build_targeted_canonical_resnet18_tir_module,
)


_FFI_DEF_RE = re.compile(
    r"__tvm_ffi_([A-Za-z0-9_]+)\(void\* self_handle, void\* args, int32_t num_args, void\* result\) \{"
)


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


def _inspect_source(mod):
    return tvm.tirx.build(mod, target={"kind": "typhoon"}).inspect_source()


def _extract_ffi_function_bodies(source):
    matches = list(_FFI_DEF_RE.finditer(source))
    return {
        match.group(1): source[match.start() : matches[index + 1].start() if index + 1 < len(matches) else len(source)]
        for index, match in enumerate(matches)
    }


def _count_marker_in_ffi_bodies(source, marker):
    return sum(body.count(marker) for body in _extract_ffi_function_bodies(source).values())


def _assert_graphized_function_has_no_host_fallback(body, *required_markers, require_submission=True):
    if require_submission:
        assert "TVMTyphoonGraphBegin" in body
        assert "TVMTyphoonSubmitGraph" in body
        assert "TVMTyphoonWaitGraph" in body
    for marker in required_markers:
        assert marker in body
    assert "for (" not in body


def _tensor(array):
    return tvm.runtime.tensor(np.array(array))


def _graph_module(params, *stmts):
    func = tvm.tirx.PrimFunc(params, tvm.tirx.stmt_seq(*stmts)).with_attr("global_symbol", "main")
    return tvm.IRModule.from_expr(func)


def _build_dma_matmul_dma_module(m=2, n=2, k=2):
    a = tvm.tirx.decl_buffer((m, k), "float32", name="A")
    b = tvm.tirx.decl_buffer((k, n), "float32", name="B")
    c = tvm.tirx.decl_buffer((m, n), "float32", name="C")
    a_bytes = m * k * 4
    b_bytes = k * n * 4
    c_bytes = m * n * 4
    return _graph_module(
        [a, b, c],
        tvm.tirx.typhoon.region_decl(7, 0, 0, a_bytes, 16, 0, "lhs"),
        tvm.tirx.typhoon.region_decl(7, 1, a_bytes, b_bytes, 16, 0, "rhs"),
        tvm.tirx.typhoon.region_decl(7, 2, a_bytes + b_bytes, c_bytes, 16, 0, "out"),
        tvm.tirx.typhoon.task_dma(7, 1, 0, a.data, 0, 0, a_bytes, []),
        tvm.tirx.typhoon.task_dma(7, 2, 0, b.data, 0, 1, b_bytes, []),
        tvm.tirx.typhoon.task_matmul(7, 3, 0, 1, 2, m, n, k, 2, 0, [1, 2]),
        tvm.tirx.typhoon.task_dma(7, 4, 1, c.data, 0, 2, c_bytes, [3]),
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


def test_typhoon_graph_entry_codegen_has_no_host_fallback_loops():
    source = _inspect_source(_build_mixed_module())
    main_body = _extract_ffi_function_bodies(source)["main"]
    _assert_graphized_function_has_no_host_fallback(
        main_body,
        "TVMTyphoonAddReshapeTask",
        "TVMTyphoonAddMatmulTask",
    )


def test_typhoon_codegen_emits_dma_decl_with_bytes_parameter():
    source = _inspect_source(_build_mixed_module())
    assert (
        "int32_t TVMTyphoonAddDMATask(int32_t, int32_t, int32_t, void*, int64_t, "
        "int32_t, int64_t, int64_t, int32_t, void*)"
    ) in source


def test_typhoon_resnet18_graphized_conv_codegen_stays_graph_only():
    mod = build_targeted_canonical_resnet18_tir_module()
    mod = tvm.tirx.transform.IdentifyTyphoonResNet18()(mod)
    mod = tvm.tirx.transform.PlanTyphoonSRAM()(mod)
    mod = tvm.tirx.transform.BuildTyphoonGraph()(mod)
    text = mod.script()
    assert text.count("T.typhoon.submit_graph") == 1

    for name in ["conv2d", "conv2d1", "conv2d4"]:
        text = mod[name].script()
        assert "TVMTyphoonCaptureCallPlanned" in text
        assert "T.typhoon.task_reshape" not in text

    source = _inspect_source(mod)
    assert _count_marker_in_ffi_bodies(source, "TVMTyphoonGraphBegin") == 1
    assert _count_marker_in_ffi_bodies(source, "TVMTyphoonSubmitGraph") == 1
    assert _count_marker_in_ffi_bodies(source, "TVMTyphoonWaitGraph") == 1
    ffi_bodies = _extract_ffi_function_bodies(source)
    graph_entry_bodies = [body for body in ffi_bodies.values() if "TVMTyphoonGraphBegin" in body]
    assert len(graph_entry_bodies) == 1
    _assert_graphized_function_has_no_host_fallback(graph_entry_bodies[0])
    for name in ["conv2d", "conv2d1", "conv2d4"]:
        body = ffi_bodies[name]
        assert "TVMTyphoonCaptureCallPlanned" in body
        assert "TVMTyphoonAddReshapeTask" not in body
    assert "TVMTyphoonReplayWholeGraphBegin" in ffi_bodies["add9"]
    assert "TVMTyphoonReplayCapturedLayer" in ffi_bodies["add9"]
    assert (
        "TVMTyphoonCapturePackedArgsPlanned" in ffi_bodies["add9"]
        or "TVMTyphoonCaptureCallPlanned" in ffi_bodies["add9"]
    )
    assert "TVMTyphoonAddReshapeTask" not in ffi_bodies["add9"]
    assert "TVMTyphoonAddMatmulTask" not in ffi_bodies["add9"]
    assert "for (" not in ffi_bodies["add9"]


def test_typhoon_resnet18_final_codegen_uses_compact_replay_helper():
    mod = build_targeted_canonical_resnet18_tir_module()
    source = _inspect_source(mod)
    ffi_bodies = _extract_ffi_function_bodies(source)
    final_body = ffi_bodies["add9"]

    assert "TVMTyphoonReplayWholeGraphBegin" in final_body
    assert "TVMTyphoonReplayCapturedLayer" in final_body
    assert (
        "TVMTyphoonCapturePackedArgsPlanned" in final_body
        or "TVMTyphoonCaptureCallPlanned" in final_body
    )
    assert "TVMTyphoonAddDMATask" not in final_body
    assert "TVMTyphoonAddReshapeTask" not in final_body
    assert "TVMTyphoonAddMatmulTask" not in final_body
    assert "TVMTyphoonAddVectorTask" not in final_body


def test_typhoon_invalid_sram_usage_raises_clear_error():
    with pytest.raises(ValueError, match="overlap|bounds"):
        _jit_module(_build_invalid_sram_module())


def estimate_matmul_latency(m, n, k):
    mod = _build_dma_matmul_dma_module(m=m, n=n, k=k)
    fn = _jit_module(mod)["main"]
    out = _tensor(np.zeros((m, n), dtype="float32"))

    fn(_tensor(np.ones((m, k), dtype="float32")), _tensor(np.ones((k, n), dtype="float32")), out)

    trace = _get_typhoon_trace()
    matmul = next(record for record in trace if record["kind"] == "matmul")
    return matmul["end_time"] - matmul["start_time"]


def test_typhoon_cost_model_penalizes_smaller_tiles():
    big = estimate_matmul_latency(m=64, n=64, k=64)
    small = estimate_matmul_latency(m=32, n=32, k=32) * 8
    assert small > big
