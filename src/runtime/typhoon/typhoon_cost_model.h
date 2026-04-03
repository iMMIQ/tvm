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

#ifndef TVM_RUNTIME_TYPHOON_COST_MODEL_H_
#define TVM_RUNTIME_TYPHOON_COST_MODEL_H_

#include "typhoon_types.h"

#include <stdint.h>

namespace tvm {
namespace runtime {
namespace typhoon {

struct TyphoonHWConfig {
  int64_t common_fixed_noise_cycles{16};
  int64_t dma_bytes_per_cycle{32};
  int64_t matmul_ops_per_cycle{256};
  int64_t vector_elems_per_cycle{32};
  int64_t reshape_bytes_per_cycle{64};
};

int64_t EstimateLatency(const TyphoonTask& task, const TyphoonHWConfig& hw);
int64_t EstimateSramBytesRead(const TyphoonTask& task);
int64_t EstimateSramBytesWritten(const TyphoonTask& task);

}  // namespace typhoon
}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_TYPHOON_COST_MODEL_H_
