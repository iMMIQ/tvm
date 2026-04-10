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

import ctypes
import json

import numpy as np
import pytest
from tests.python.runtime.typhoon_test_utils import (
    _add_dma_task,
    _add_matmul_task,
    _add_reshape_task,
    _add_vector_task,
    _check_ok,
    _dep_buffer,
    _runtime,
    build_runtime_graph_with_broadcast_add,
    build_runtime_graph_with_global_average_pool,
    build_runtime_graph_with_matmul_rhs_row_major,
    build_runtime_graph_with_reshape_im2col,
    build_runtime_graph_with_transpose_2d,
    build_runtime_graph_with_vector_maxpool,
)


def _run_graph(**flags):
    lib, reset, last_error, _ = _runtime()
    reset()
    graph_id = 41

    _check_ok(
        last_error,
        lib.TVMTyphoonDeclareRegion(
            graph_id,
            0,
            0,
            256,
            64,
            int(flags.get("preinitialized_input", False) or flags.get("vector_output_too_large", False)),
            b"input",
        ),
    )
    _check_ok(
        last_error,
        lib.TVMTyphoonDeclareRegion(
            graph_id,
            1,
            128 if flags.get("overlap", False) else 256,
            256,
            64,
            0,
            b"output",
        ),
    )

    if flags.get("duplicate_task_id", False):
        _add_reshape_task(lib, last_error, graph_id, 1, 0, 1, 64, 0)
        _add_reshape_task(lib, last_error, graph_id, 1, 0, 1, 64, 0)
    elif flags.get("unknown_dep", False):
        _add_reshape_task(lib, last_error, graph_id, 1, 0, 1, 64, 0, deps=[99])
    elif flags.get("compute_reads_unknown_region", False):
        _add_vector_task(lib, last_error, graph_id, 1, 0, 99, 99, 1, 64, 2)
    elif flags.get("dma_bytes_too_large", False):
        _add_dma_task(lib, last_error, graph_id, 1, 0, ctypes.c_void_p(1), 0, 0, 512)
    elif flags.get("vector_output_too_large", False):
        _add_vector_task(lib, last_error, graph_id, 1, 0, 0, 0, 1, 128, 2)
    else:
        if flags.get("preinitialized_input", False):
            _add_reshape_task(lib, last_error, graph_id, 1, 0, 1, 64, 0)
        else:
            _add_dma_task(lib, last_error, graph_id, 1, 0, ctypes.c_void_p(1), 0, 0, 256)
            _add_reshape_task(lib, last_error, graph_id, 2, 0, 1, 64, 0, deps=[1])

    _check_ok(last_error, lib.TVMTyphoonGraphBegin(graph_id))
    _check_ok(last_error, lib.TVMTyphoonSubmitGraph(graph_id))
    _check_ok(last_error, lib.TVMTyphoonWaitGraph(graph_id))


def test_typhoon_rejects_overlapping_regions():
    with pytest.raises(RuntimeError, match="overlap"):
        _run_graph(overlap=True)


def test_typhoon_rejects_compute_from_unknown_sram_region():
    with pytest.raises(RuntimeError, match="SRAM|region"):
        _run_graph(compute_reads_unknown_region=True)


def test_typhoon_rejects_unknown_task_dependency():
    with pytest.raises(RuntimeError, match="unknown"):
        _run_graph(unknown_dep=True)


def test_typhoon_rejects_duplicate_task_ids():
    with pytest.raises(RuntimeError, match="duplicate task_id"):
        _run_graph(duplicate_task_id=True)


def test_typhoon_accepts_preinitialized_region_reads():
    _run_graph(preinitialized_input=True)


def test_typhoon_rejects_dma_bytes_larger_than_region():
    with pytest.raises(RuntimeError, match="out-of-bounds|size mismatch"):
        _run_graph(dma_bytes_too_large=True)


def test_typhoon_rejects_vector_output_larger_than_region():
    with pytest.raises(RuntimeError, match="out-of-bounds|size mismatch"):
        _run_graph(vector_output_too_large=True)


def test_runtime_typhoon_test_utils_exposes_trace_helper():
    from tests.python.runtime.typhoon_test_utils import get_trace_json

    assert callable(get_trace_json)


def _run_and_get_trace():
    lib, reset, last_error, get_trace = _runtime()
    reset()
    graph_id = 52

    _check_ok(last_error, lib.TVMTyphoonDeclareRegion(graph_id, 0, 0, 256, 64, 1, b"input"))
    _check_ok(last_error, lib.TVMTyphoonDeclareRegion(graph_id, 1, 256, 256, 64, 0, b"left"))
    _check_ok(last_error, lib.TVMTyphoonDeclareRegion(graph_id, 2, 512, 256, 64, 0, b"right"))

    _add_reshape_task(lib, last_error, graph_id, 2, 0, 2, 64, 0)
    _add_reshape_task(lib, last_error, graph_id, 1, 0, 1, 64, 0)
    _check_ok(last_error, lib.TVMTyphoonGraphBegin(graph_id))
    _check_ok(last_error, lib.TVMTyphoonSubmitGraph(graph_id))
    _check_ok(last_error, lib.TVMTyphoonWaitGraph(graph_id))
    return json.loads(get_trace())


def test_typhoon_scheduler_is_deterministic():
    trace_a = _run_and_get_trace()
    trace_b = _run_and_get_trace()
    assert trace_a == trace_b


def _trace_duration(trace, task_kind):
    record = next(record for record in trace if record["kind"] == task_kind)
    return record["end_time"] - record["start_time"]


def _single_task_trace(task_kind):
    lib, reset, last_error, get_trace = _runtime()
    reset()
    graph_id = 77
    dma_buffer = (ctypes.c_uint8 * 16)()

    _check_ok(last_error, lib.TVMTyphoonDeclareRegion(graph_id, 0, 0, 64, 16, 1, b"in0"))
    _check_ok(last_error, lib.TVMTyphoonDeclareRegion(graph_id, 1, 64, 64, 16, 1, b"in1"))
    _check_ok(last_error, lib.TVMTyphoonDeclareRegion(graph_id, 2, 128, 64, 16, 0, b"out"))

    if task_kind == "dma":
        _add_dma_task(
            lib,
            last_error,
            graph_id,
            1,
            0,
            ctypes.cast(dma_buffer, ctypes.c_void_p),
            0,
            2,
            1,
        )
    elif task_kind == "matmul":
        _add_matmul_task(lib, last_error, graph_id, 1, 0, 1, 2, 1, 1, 1, 2, 0)
    elif task_kind == "vector":
        _add_vector_task(lib, last_error, graph_id, 1, 1, 0, 0, 2, 1, 2)
    elif task_kind == "reshape":
        _add_reshape_task(lib, last_error, graph_id, 1, 0, 2, 1, 0)
    else:
        raise AssertionError(f"unsupported task kind: {task_kind}")

    _check_ok(last_error, lib.TVMTyphoonGraphBegin(graph_id))
    _check_ok(last_error, lib.TVMTyphoonSubmitGraph(graph_id))
    _check_ok(last_error, lib.TVMTyphoonWaitGraph(graph_id))
    return json.loads(get_trace())


def test_typhoon_cost_model_uses_common_fixed_noise_across_task_kinds():
    latencies = {
        task_kind: _trace_duration(_single_task_trace(task_kind), task_kind)
        for task_kind in ("dma", "matmul", "vector", "reshape")
    }
    assert len(set(latencies.values())) == 1, latencies


def test_typhoon_dma_honors_sram_byte_offset():
    lib, reset, last_error, _ = _runtime()
    reset()
    graph_id = 94
    input_array = np.array([1.0, 2.0, 3.0, 4.0], dtype="float32")
    output_array = np.zeros_like(input_array)

    _check_ok(last_error, lib.TVMTyphoonDeclareRegion(graph_id, 0, 0, 64, 16, 0, b"scratch"))
    _add_dma_task(
        lib,
        last_error,
        graph_id,
        1,
        0,
        ctypes.c_void_p(input_array.ctypes.data),
        0,
        0,
        input_array.nbytes,
        sram_byte_offset=16,
    )
    _add_dma_task(
        lib,
        last_error,
        graph_id,
        2,
        1,
        ctypes.c_void_p(output_array.ctypes.data),
        0,
        0,
        output_array.nbytes,
        deps=[1],
        sram_byte_offset=16,
    )

    _check_ok(last_error, lib.TVMTyphoonGraphBegin(graph_id))
    _check_ok(last_error, lib.TVMTyphoonSubmitGraph(graph_id))
    _check_ok(last_error, lib.TVMTyphoonWaitGraph(graph_id))
    np.testing.assert_allclose(output_array, input_array)


def test_typhoon_trace_has_required_fields():
    trace = _run_and_get_trace()
    assert trace
    assert {
        "task_id",
        "layer_id",
        "kind",
        "resource",
        "start_time",
        "end_time",
        "sram_bytes_read",
        "sram_bytes_written",
        "region_reads",
        "region_writes",
    } <= set(trace[0])


def test_runtime_typhoon_executes_reshape_im2col():
    output = build_runtime_graph_with_reshape_im2col(
        np.arange(1, 10, dtype="float32").reshape(1, 1, 3, 3)
    )
    np.testing.assert_allclose(
        output,
        np.array(
            [1.0, 2.0, 4.0, 5.0, 2.0, 3.0, 5.0, 6.0, 4.0, 5.0, 7.0, 8.0, 5.0, 6.0, 8.0, 9.0],
            dtype="float32",
        ),
    )


def test_runtime_typhoon_executes_vector_maxpool():
    output = build_runtime_graph_with_vector_maxpool(
        np.arange(1, 17, dtype="float32").reshape(1, 1, 4, 4)
    )
    np.testing.assert_allclose(output, np.array([[[[6.0, 8.0], [14.0, 16.0]]]], dtype="float32"))


def test_runtime_typhoon_executes_vector_global_average_pool():
    output = build_runtime_graph_with_global_average_pool(
        np.array(
            [[[[1.0, 2.0], [3.0, 4.0]], [[10.0, 14.0], [18.0, 22.0]]]], dtype="float32"
        )
    )
    np.testing.assert_allclose(output, np.array([[[[2.5]], [[16.0]]]], dtype="float32"))


def test_runtime_typhoon_executes_broadcast_add():
    output = build_runtime_graph_with_broadcast_add(
        np.arange(16, dtype="float32").reshape(2, 8),
        np.array([10.0, 20.0, 30.0, 40.0, 1.0, 2.0, 3.0, 4.0], dtype="float32"),
    )
    np.testing.assert_allclose(
        output,
        np.array(
            [
                [10.0, 21.0, 32.0, 43.0, 5.0, 7.0, 9.0, 11.0],
                [18.0, 29.0, 40.0, 51.0, 13.0, 15.0, 17.0, 19.0],
            ],
            dtype="float32",
        ),
    )


def test_runtime_typhoon_executes_matmul_with_rhs_row_major_layout():
    output = build_runtime_graph_with_matmul_rhs_row_major(
        np.array([[1.0, 2.0, 3.0]], dtype="float32"),
        np.array([[10.0, 20.0, 30.0], [1.0, -1.0, 2.0]], dtype="float32"),
    )
    np.testing.assert_allclose(output, np.array([[140.0, 5.0]], dtype="float32"))


def test_runtime_typhoon_executes_transpose_2d():
    output = build_runtime_graph_with_transpose_2d(
        np.array([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]], dtype="float32")
    )
    np.testing.assert_allclose(
        output,
        np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]], dtype="float32"),
    )
