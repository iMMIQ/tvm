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
"""Typhoon graph-building helpers."""

import tvm


def _task_args(args, deps):
    deps = [] if deps is None else list(deps)
    return [*args, len(deps), *deps]


def region_decl(graph_id, region_id, offset, size, alignment, preinitialized, tag, span=None):
    call = tvm.tirx.call_intrin(
        "int32",
        "tirx.typhoon.region_decl",
        graph_id,
        region_id,
        offset,
        size,
        alignment,
        preinitialized,
        tag,
        span=span,
    )
    return tvm.tirx.Evaluate(call)


def task_dma(
    graph_id,
    task_id,
    direction,
    global_handle,
    global_byte_offset,
    sram_region_id,
    bytes,
    deps,
    span=None,
):
    call = tvm.tirx.call_intrin(
        "int32",
        "tirx.typhoon.task_dma",
        *_task_args(
            [graph_id, task_id, direction, global_handle, global_byte_offset, sram_region_id, bytes],
            deps,
        ),
        span=span,
    )
    return tvm.tirx.Evaluate(call)


def task_matmul(
    graph_id,
    task_id,
    a_region_id,
    b_region_id,
    c_region_id,
    m,
    n,
    k,
    dtype_code,
    layout_code,
    deps,
    span=None,
):
    call = tvm.tirx.call_intrin(
        "int32",
        "tirx.typhoon.task_matmul",
        *_task_args(
            [
                graph_id,
                task_id,
                a_region_id,
                b_region_id,
                c_region_id,
                m,
                n,
                k,
                dtype_code,
                layout_code,
            ],
            deps,
        ),
        span=span,
    )
    return tvm.tirx.Evaluate(call)


def task_vector(
    graph_id,
    task_id,
    op_code,
    in0_region_id,
    in1_region_id,
    out_region_id,
    elem_count,
    dtype_code,
    deps,
    span=None,
):
    call = tvm.tirx.call_intrin(
        "int32",
        "tirx.typhoon.task_vector",
        *_task_args(
            [
                graph_id,
                task_id,
                op_code,
                in0_region_id,
                in1_region_id,
                out_region_id,
                elem_count,
                dtype_code,
            ],
            deps,
        ),
        span=span,
    )
    return tvm.tirx.Evaluate(call)


def task_reshape(
    graph_id,
    task_id,
    in_region_id,
    out_region_id,
    elem_count,
    transform_code,
    deps,
    span=None,
):
    call = tvm.tirx.call_intrin(
        "int32",
        "tirx.typhoon.task_reshape",
        *_task_args(
            [graph_id, task_id, in_region_id, out_region_id, elem_count, transform_code], deps
        ),
        span=span,
    )
    return tvm.tirx.Evaluate(call)


def submit_graph(graph_id, span=None):
    call = tvm.tirx.call_intrin("int32", "tirx.typhoon.submit_graph", graph_id, span=span)
    return tvm.tirx.Evaluate(call)


def wait_graph(graph_id, span=None):
    call = tvm.tirx.call_intrin("int32", "tirx.typhoon.wait_graph", graph_id, span=span)
    return tvm.tirx.Evaluate(call)
