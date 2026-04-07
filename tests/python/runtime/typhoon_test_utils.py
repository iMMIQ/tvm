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

import numpy as np


def _runtime():
    import tvm

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
        ctypes.c_int64,
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_int32),
    ]
    lib.TVMTyphoonAddDMATask.restype = ctypes.c_int
    lib.TVMTyphoonAddMatmulTask.argtypes = [
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_int32),
    ]
    lib.TVMTyphoonAddMatmulTask.restype = ctypes.c_int
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
        ctypes.POINTER(ctypes.c_int64),
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
        ctypes.POINTER(ctypes.c_int64),
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


def _metadata_buffer(metadata):
    metadata = list(metadata)
    if not metadata:
        return 0, None
    arr = (ctypes.c_int64 * len(metadata))(*metadata)
    return len(metadata), arr


def _check_ok(last_error, code):
    if code != 0:
        raise RuntimeError(last_error())


def _add_vector_task(lib, last_error, *base_args, metadata=(), deps=()):
    num_metadata, metadata_ptr = _metadata_buffer(metadata)
    num_deps, dep_ids = _dep_buffer(deps)
    _check_ok(
        last_error,
        lib.TVMTyphoonAddVectorTask(
            *base_args, num_metadata, metadata_ptr, num_deps, dep_ids
        ),
    )


def _add_matmul_task(lib, last_error, *base_args, deps=()):
    num_deps, dep_ids = _dep_buffer(deps)
    _check_ok(last_error, lib.TVMTyphoonAddMatmulTask(*base_args, num_deps, dep_ids))


def _add_dma_task(lib, last_error, *base_args, deps=(), sram_byte_offset=0):
    graph_id, task_id, direction, global_handle, global_byte_offset, sram_region_id, bytes_ = (
        base_args
    )
    num_deps, dep_ids = _dep_buffer(deps)
    _check_ok(
        last_error,
        lib.TVMTyphoonAddDMATask(
            graph_id,
            task_id,
            direction,
            global_handle,
            global_byte_offset,
            sram_region_id,
            sram_byte_offset,
            bytes_,
            num_deps,
            dep_ids,
        ),
    )


def _add_reshape_task(lib, last_error, *base_args, metadata=(), deps=()):
    num_metadata, metadata_ptr = _metadata_buffer(metadata)
    num_deps, dep_ids = _dep_buffer(deps)
    _check_ok(
        last_error,
        lib.TVMTyphoonAddReshapeTask(
            *base_args, num_metadata, metadata_ptr, num_deps, dep_ids
        ),
    )


def get_trace_json():
    _, _, _, get_trace = _runtime()
    return get_trace()


def _run_dataflow_graph(task_id, input_array, output_shape, emit_task):
    lib, reset, last_error, _ = _runtime()
    reset()
    graph_id = 80 + task_id
    input_array = np.array(input_array, dtype="float32", copy=True)
    output_array = np.zeros(output_shape, dtype="float32")

    _check_ok(last_error, lib.TVMTyphoonDeclareRegion(graph_id, 0, 0, input_array.nbytes, 64, 0, b"input"))
    _check_ok(
        last_error,
        lib.TVMTyphoonDeclareRegion(graph_id, 1, 256, output_array.nbytes, 64, 0, b"output"),
    )

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
    )
    emit_task(lib, last_error, graph_id)
    _add_dma_task(
        lib,
        last_error,
        graph_id,
        3,
        1,
        ctypes.c_void_p(output_array.ctypes.data),
        0,
        1,
        output_array.nbytes,
        deps=[2],
    )
    _check_ok(last_error, lib.TVMTyphoonGraphBegin(graph_id))
    _check_ok(last_error, lib.TVMTyphoonSubmitGraph(graph_id))
    _check_ok(last_error, lib.TVMTyphoonWaitGraph(graph_id))
    return output_array


def build_runtime_graph_with_reshape_im2col(input_array):
    return _run_dataflow_graph(
        1,
        input_array,
        (16,),
        lambda lib, last_error, graph_id: _add_reshape_task(
            lib,
            last_error,
            graph_id,
            2,
            0,
            1,
            64,
            1,
            metadata=[1, 1, 3, 3, 2, 2, 1, 1, 0, 0, 2, 2],
            deps=[1],
        ),
    )


def build_runtime_graph_with_vector_maxpool(input_array):
    return _run_dataflow_graph(
        2,
        input_array,
        (1, 1, 2, 2),
        lambda lib, last_error, graph_id: _add_vector_task(
            lib,
            last_error,
            graph_id,
            2,
            2,
            0,
            0,
            1,
            4,
            2,
            metadata=[1, 1, 4, 4, 3, 3, 2, 2, 1, 1, 2, 2],
            deps=[1],
        ),
    )


def build_runtime_graph_with_global_average_pool(input_array):
    return _run_dataflow_graph(
        3,
        input_array,
        (1, 2, 1, 1),
        lambda lib, last_error, graph_id: _add_vector_task(
            lib,
            last_error,
            graph_id,
            2,
            3,
            0,
            0,
            1,
            2,
            2,
            metadata=[1, 2, 2, 2],
            deps=[1],
        ),
    )


def build_runtime_graph_with_broadcast_add(input_array, bias_array):
    lib, reset, last_error, _ = _runtime()
    reset()
    graph_id = 90
    input_array = np.array(input_array, dtype="float32", copy=True)
    bias_array = np.array(bias_array, dtype="float32", copy=True)
    output_array = np.zeros_like(input_array)

    _check_ok(last_error, lib.TVMTyphoonDeclareRegion(graph_id, 0, 0, input_array.nbytes, 64, 0, b"input"))
    _check_ok(
        last_error,
        lib.TVMTyphoonDeclareRegion(graph_id, 1, 262144, bias_array.nbytes, 64, 0, b"bias"),
    )
    _check_ok(
        last_error,
        lib.TVMTyphoonDeclareRegion(graph_id, 2, 327680, output_array.nbytes, 64, 0, b"output"),
    )

    _add_dma_task(
        lib, last_error, graph_id, 1, 0, ctypes.c_void_p(input_array.ctypes.data), 0, 0, input_array.nbytes
    )
    _add_dma_task(
        lib, last_error, graph_id, 2, 0, ctypes.c_void_p(bias_array.ctypes.data), 0, 1, bias_array.nbytes
    )
    _add_vector_task(
        lib,
        last_error,
        graph_id,
        3,
        0,
        0,
        1,
        2,
        input_array.size,
        2,
        metadata=[input_array.size // bias_array.size, bias_array.size],
        deps=[1, 2],
    )
    _add_dma_task(
        lib,
        last_error,
        graph_id,
        4,
        1,
        ctypes.c_void_p(output_array.ctypes.data),
        0,
        2,
        output_array.nbytes,
        deps=[3],
    )
    _check_ok(last_error, lib.TVMTyphoonGraphBegin(graph_id))
    _check_ok(last_error, lib.TVMTyphoonSubmitGraph(graph_id))
    _check_ok(last_error, lib.TVMTyphoonWaitGraph(graph_id))
    return output_array


def build_runtime_graph_with_matmul_rhs_row_major(input_array, weight_array):
    lib, reset, last_error, _ = _runtime()
    reset()
    graph_id = 92
    input_array = np.array(input_array, dtype="float32", copy=True)
    weight_array = np.array(weight_array, dtype="float32", copy=True)
    output_array = np.zeros((input_array.shape[0], weight_array.shape[0]), dtype="float32")

    _check_ok(last_error, lib.TVMTyphoonDeclareRegion(graph_id, 0, 0, input_array.nbytes, 64, 0, b"input"))
    _check_ok(
        last_error,
        lib.TVMTyphoonDeclareRegion(graph_id, 1, 262144, weight_array.nbytes, 64, 0, b"weight"),
    )
    _check_ok(
        last_error,
        lib.TVMTyphoonDeclareRegion(graph_id, 2, 524288, output_array.nbytes, 64, 0, b"output"),
    )

    _add_dma_task(
        lib, last_error, graph_id, 1, 0, ctypes.c_void_p(input_array.ctypes.data), 0, 0, input_array.nbytes
    )
    _add_dma_task(
        lib, last_error, graph_id, 2, 0, ctypes.c_void_p(weight_array.ctypes.data), 0, 1, weight_array.nbytes
    )
    _add_matmul_task(
        lib,
        last_error,
        graph_id,
        3,
        0,
        1,
        2,
        input_array.shape[0],
        weight_array.shape[0],
        input_array.shape[1],
        2,
        1,
        deps=[1, 2],
    )
    _add_dma_task(
        lib,
        last_error,
        graph_id,
        4,
        1,
        ctypes.c_void_p(output_array.ctypes.data),
        0,
        2,
        output_array.nbytes,
        deps=[3],
    )
    _check_ok(last_error, lib.TVMTyphoonGraphBegin(graph_id))
    _check_ok(last_error, lib.TVMTyphoonSubmitGraph(graph_id))
    _check_ok(last_error, lib.TVMTyphoonWaitGraph(graph_id))
    return output_array


def build_runtime_graph_with_transpose_2d(input_array):
    lib, reset, last_error, _ = _runtime()
    reset()
    graph_id = 91
    input_array = np.array(input_array, dtype="float32", copy=True)
    output_array = np.zeros((input_array.shape[1], input_array.shape[0]), dtype="float32")

    _check_ok(last_error, lib.TVMTyphoonDeclareRegion(graph_id, 0, 0, input_array.nbytes, 64, 0, b"input"))
    _check_ok(
        last_error,
        lib.TVMTyphoonDeclareRegion(graph_id, 1, 262144, output_array.nbytes, 64, 0, b"output"),
    )

    _add_dma_task(
        lib, last_error, graph_id, 1, 0, ctypes.c_void_p(input_array.ctypes.data), 0, 0, input_array.nbytes
    )
    _add_reshape_task(
        lib,
        last_error,
        graph_id,
        2,
        0,
        1,
        output_array.nbytes,
        2,
        metadata=[input_array.shape[0], input_array.shape[1]],
        deps=[1],
    )
    _add_dma_task(
        lib,
        last_error,
        graph_id,
        3,
        1,
        ctypes.c_void_p(output_array.ctypes.data),
        0,
        1,
        output_array.nbytes,
        deps=[2],
    )
    _check_ok(last_error, lib.TVMTyphoonGraphBegin(graph_id))
    _check_ok(last_error, lib.TVMTyphoonSubmitGraph(graph_id))
    _check_ok(last_error, lib.TVMTyphoonWaitGraph(graph_id))
    return output_array


def get_trace_records():
    return json.loads(get_trace_json())
