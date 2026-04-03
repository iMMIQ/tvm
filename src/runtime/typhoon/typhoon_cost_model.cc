/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

#include "typhoon_cost_model.h"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace tvm {
namespace runtime {
namespace typhoon {

namespace {

int64_t DTypeBytes(int32_t dtype_code) {
  switch (dtype_code) {
    case 0:
      return 1;
    case 1:
      return 2;
    case 2:
      return 4;
    case 3:
      return 8;
    default:
      throw std::runtime_error("Typhoon dtype_code " + std::to_string(dtype_code) +
                               " is unsupported");
  }
}

int64_t DivRoundUp(int64_t value, int64_t divisor) {
  return (value + divisor - 1) / divisor;
}

int64_t EstimateLatencyFromWorkload(int64_t workload_term, const TyphoonHWConfig& hw) {
  return std::max<int64_t>(1, hw.common_fixed_noise_cycles + workload_term);
}

}  // namespace

int64_t EstimateLatency(const TyphoonTask& task, const TyphoonHWConfig& hw) {
  switch (task.kind) {
    case TaskKind::kDMA:
      return EstimateLatencyFromWorkload(DivRoundUp(task.bytes, hw.dma_bytes_per_cycle), hw);
    case TaskKind::kMatmul: {
      int64_t ops = task.m * task.n * task.k;
      return EstimateLatencyFromWorkload(DivRoundUp(ops, hw.matmul_ops_per_cycle), hw);
    }
    case TaskKind::kVector:
      return EstimateLatencyFromWorkload(DivRoundUp(task.elem_count, hw.vector_elems_per_cycle),
                                         hw);
    case TaskKind::kReshape:
      return EstimateLatencyFromWorkload(DivRoundUp(task.elem_count, hw.reshape_bytes_per_cycle),
                                         hw);
  }
  return 1;
}

int64_t EstimateSramBytesRead(const TyphoonTask& task) {
  switch (task.kind) {
    case TaskKind::kDMA:
      return task.direction == 1 ? task.bytes : 0;
    case TaskKind::kMatmul:
      return (task.m * task.k + task.k * task.n) * DTypeBytes(task.dtype_code);
    case TaskKind::kVector:
      return 2 * task.elem_count * DTypeBytes(task.dtype_code);
    case TaskKind::kReshape:
      return task.elem_count;
  }
  return 0;
}

int64_t EstimateSramBytesWritten(const TyphoonTask& task) {
  switch (task.kind) {
    case TaskKind::kDMA:
      return task.direction == 0 ? task.bytes : 0;
    case TaskKind::kMatmul:
      return task.m * task.n * DTypeBytes(task.dtype_code);
    case TaskKind::kVector:
      return task.elem_count * DTypeBytes(task.dtype_code);
    case TaskKind::kReshape:
      return task.elem_count;
  }
  return 0;
}

}  // namespace typhoon
}  // namespace runtime
}  // namespace tvm
