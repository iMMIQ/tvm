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
import os

import pytest
import tvm


def _runtime():
    lib_dir = os.environ["TVM_LIBRARY_PATH"]
    lib = ctypes.CDLL(os.path.join(lib_dir, "libtvm.so"))

    lib.TVMTyphoonGraphBegin.argtypes = [ctypes.c_int32]
    lib.TVMTyphoonGraphBegin.restype = ctypes.c_int
    lib.TVMTyphoonDeclareRegion.argtypes = [
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int32,
        ctypes.c_char_p,
    ]
    lib.TVMTyphoonDeclareRegion.restype = ctypes.c_int
    lib.TVMTyphoonAddDMATask.argtypes = [
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_int32,
        ctypes.c_int64,
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_int32),
    ]
    lib.TVMTyphoonAddDMATask.restype = ctypes.c_int
    lib.TVMTyphoonAddVectorTask.argtypes = [
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int64,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_int32),
    ]
    lib.TVMTyphoonAddVectorTask.restype = ctypes.c_int
    lib.TVMTyphoonAddReshapeTask.argtypes = [
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int64,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_int32),
    ]
    lib.TVMTyphoonAddReshapeTask.restype = ctypes.c_int
    lib.TVMTyphoonSubmitGraph.argtypes = [ctypes.c_int32]
    lib.TVMTyphoonSubmitGraph.restype = ctypes.c_int
    lib.TVMTyphoonWaitGraph.argtypes = [ctypes.c_int32]
    lib.TVMTyphoonWaitGraph.restype = ctypes.c_int

    reset = tvm.get_global_func("runtime.typhoon.testing_reset", allow_missing=True)
    last_error = tvm.get_global_func("runtime.typhoon.testing_last_error", allow_missing=True)
    get_trace = tvm.get_global_func("runtime.typhoon_get_last_trace_json", allow_missing=True)
    assert reset is not None
    assert last_error is not None
    assert get_trace is not None
    return lib, reset, last_error, get_trace


def _dep_buffer(dep_ids):
    dep_ids = list(dep_ids)
    if not dep_ids:
        return 0, None
    arr = (ctypes.c_int32 * len(dep_ids))(*dep_ids)
    return len(dep_ids), arr


def _check_ok(last_error, code):
    if code != 0:
        raise RuntimeError(last_error())


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
        num_deps, dep_ids = _dep_buffer([])
        _check_ok(last_error, lib.TVMTyphoonAddReshapeTask(graph_id, 1, 0, 1, 64, 0, num_deps, dep_ids))
        _check_ok(last_error, lib.TVMTyphoonAddReshapeTask(graph_id, 1, 0, 1, 64, 0, num_deps, dep_ids))
    elif flags.get("unknown_dep", False):
        num_deps, dep_ids = _dep_buffer([99])
        _check_ok(last_error, lib.TVMTyphoonAddReshapeTask(graph_id, 1, 0, 1, 64, 0, num_deps, dep_ids))
    elif flags.get("compute_reads_unknown_region", False):
        num_deps, dep_ids = _dep_buffer([])
        _check_ok(last_error, lib.TVMTyphoonAddVectorTask(graph_id, 1, 0, 99, 99, 1, 64, 2, num_deps, dep_ids))
    elif flags.get("dma_bytes_too_large", False):
        num_deps, dep_ids = _dep_buffer([])
        _check_ok(last_error, lib.TVMTyphoonAddDMATask(graph_id, 1, 0, ctypes.c_void_p(1), 0, 0, 512, num_deps, dep_ids))
    elif flags.get("vector_output_too_large", False):
        num_deps, dep_ids = _dep_buffer([])
        _check_ok(last_error, lib.TVMTyphoonAddVectorTask(graph_id, 1, 0, 0, 0, 1, 128, 2, num_deps, dep_ids))
    else:
        if flags.get("preinitialized_input", False):
            num_deps, dep_ids = _dep_buffer([])
            _check_ok(last_error, lib.TVMTyphoonAddReshapeTask(graph_id, 1, 0, 1, 64, 0, num_deps, dep_ids))
        else:
            num_deps, dep_ids = _dep_buffer([])
            _check_ok(last_error, lib.TVMTyphoonAddDMATask(graph_id, 1, 0, ctypes.c_void_p(1), 0, 0, 256, num_deps, dep_ids))
            num_deps, dep_ids = _dep_buffer([1])
            _check_ok(last_error, lib.TVMTyphoonAddReshapeTask(graph_id, 2, 0, 1, 64, 0, num_deps, dep_ids))

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


def _run_and_get_trace():
    lib, reset, last_error, get_trace = _runtime()
    reset()
    graph_id = 52

    _check_ok(last_error, lib.TVMTyphoonDeclareRegion(graph_id, 0, 0, 256, 64, 1, b"input"))
    _check_ok(last_error, lib.TVMTyphoonDeclareRegion(graph_id, 1, 256, 256, 64, 0, b"left"))
    _check_ok(last_error, lib.TVMTyphoonDeclareRegion(graph_id, 2, 512, 256, 64, 0, b"right"))

    num_deps, dep_ids = _dep_buffer([])
    _check_ok(last_error, lib.TVMTyphoonAddReshapeTask(graph_id, 2, 0, 2, 64, 0, num_deps, dep_ids))
    _check_ok(last_error, lib.TVMTyphoonAddReshapeTask(graph_id, 1, 0, 1, 64, 0, num_deps, dep_ids))
    _check_ok(last_error, lib.TVMTyphoonGraphBegin(graph_id))
    _check_ok(last_error, lib.TVMTyphoonSubmitGraph(graph_id))
    _check_ok(last_error, lib.TVMTyphoonWaitGraph(graph_id))
    return json.loads(get_trace())


def test_typhoon_scheduler_is_deterministic():
    trace_a = _run_and_get_trace()
    trace_b = _run_and_get_trace()
    assert trace_a == trace_b


def test_typhoon_trace_has_required_fields():
    trace = _run_and_get_trace()
    assert trace
    assert {
        "task_id",
        "kind",
        "resource",
        "start_time",
        "end_time",
        "sram_bytes_read",
        "sram_bytes_written",
    } <= set(trace[0])
