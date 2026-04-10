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

#ifndef TVM_RUNTIME_TYPHOON_TYPES_H_
#define TVM_RUNTIME_TYPHOON_TYPES_H_

#include <stdint.h>

#include <string>
#include <vector>

namespace tvm {
namespace runtime {
namespace typhoon {

constexpr int64_t kTyphoonDefaultSRAMSize = 1024 * 1024;

enum class TaskKind : int {
  kDMA = 0,
  kMatmul = 1,
  kVector = 2,
  kReshape = 3,
};

struct GlobalEndpoint {
  void* handle{nullptr};
  int64_t byte_offset{0};
};

struct TyphoonRegion {
  int32_t region_id{0};
  int64_t offset{0};
  int64_t size{0};
  int64_t alignment{0};
  bool preinitialized{false};
  std::string tag;
};

struct TyphoonTask {
  TaskKind kind{TaskKind::kDMA};
  int32_t task_id{0};
  int32_t layer_id{-1};
  std::vector<int32_t> deps;
  std::vector<int32_t> reads;
  std::vector<int32_t> writes;
  std::vector<int64_t> metadata;
  GlobalEndpoint global_endpoint;
  int32_t direction{0};
  int32_t op_code{0};
  int32_t transform_code{0};
  int32_t layout_code{0};
  int64_t sram_byte_offset{0};
  int64_t bytes{0};
  int64_t dma_batch_count{1};
  int64_t dma_global_stride{0};
  int64_t dma_sram_stride{0};
  int64_t elem_count{0};
  int32_t dtype_code{0};
  int64_t m{0};
  int64_t n{0};
  int64_t k{0};
};

struct TyphoonTraceRecord {
  int32_t task_id{0};
  int32_t layer_id{-1};
  std::string kind;
  std::string resource;
  int64_t start_time{0};
  int64_t end_time{0};
  int64_t sram_bytes_read{0};
  int64_t sram_bytes_written{0};
  std::vector<int32_t> region_reads;
  std::vector<int32_t> region_writes;
};

}  // namespace typhoon
}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_TYPHOON_TYPES_H_
