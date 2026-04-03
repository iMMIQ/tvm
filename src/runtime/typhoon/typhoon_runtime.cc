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

#include "typhoon_graph.h"

#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>

#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>

namespace tvm {
namespace runtime {
namespace typhoon {

class TyphoonRuntimeState {
 public:
  static TyphoonRuntimeState& Global() {
    static TyphoonRuntimeState runtime;
    return runtime;
  }

  void ResetForTesting() {
    std::lock_guard<std::mutex> lock(mu_);
    graphs_.clear();
    last_error_.clear();
  }

  std::string LastError() const {
    std::lock_guard<std::mutex> lock(mu_);
    return last_error_;
  }

  int GraphBegin(int32_t graph_id) {
    return Call([&]() { GetOrCreateGraph(graph_id).GraphBegin(); });
  }

  int DeclareRegion(int32_t graph_id, int32_t region_id, int64_t offset, int64_t size,
                    int64_t alignment, int32_t preinitialized, const char* tag) {
    return Call([&]() {
      GetOrCreateGraph(graph_id).DeclareRegion(region_id, offset, size, alignment,
                                               preinitialized != 0, tag);
    });
  }

  int AddDMATask(int32_t graph_id, int32_t task_id, int32_t direction, void* global_handle,
                 int64_t global_byte_offset, int32_t sram_region_id, int64_t bytes,
                 int32_t num_deps, const int32_t* dep_ids) {
    return Call([&]() {
      GetOrCreateGraph(graph_id)
          .AddDMATask(task_id, direction, global_handle, global_byte_offset, sram_region_id, bytes,
                      num_deps, dep_ids);
    });
  }

  int AddMatmulTask(int32_t graph_id, int32_t task_id, int32_t a_region_id, int32_t b_region_id,
                    int32_t c_region_id, int64_t m, int64_t n, int64_t k, int32_t dtype_code,
                    int32_t layout_code, int32_t num_deps, const int32_t* dep_ids) {
    return Call([&]() {
      GetOrCreateGraph(graph_id)
          .AddMatmulTask(task_id, a_region_id, b_region_id, c_region_id, m, n, k, dtype_code,
                         layout_code, num_deps, dep_ids);
    });
  }

  int AddVectorTask(int32_t graph_id, int32_t task_id, int32_t op_code, int32_t in0_region_id,
                    int32_t in1_region_id, int32_t out_region_id, int64_t elem_count,
                    int32_t dtype_code, int32_t num_deps, const int32_t* dep_ids) {
    return Call([&]() {
      GetOrCreateGraph(graph_id)
          .AddVectorTask(task_id, op_code, in0_region_id, in1_region_id, out_region_id,
                         elem_count, dtype_code, num_deps, dep_ids);
    });
  }

  int AddReshapeTask(int32_t graph_id, int32_t task_id, int32_t in_region_id,
                     int32_t out_region_id, int64_t elem_count, int32_t transform_code,
                     int32_t num_deps, const int32_t* dep_ids) {
    return Call([&]() {
      GetOrCreateGraph(graph_id)
          .AddReshapeTask(task_id, in_region_id, out_region_id, elem_count, transform_code,
                          num_deps, dep_ids);
    });
  }

  int SubmitGraph(int32_t graph_id) {
    return Call([&]() { GetOrCreateGraph(graph_id).Submit(); });
  }

  int WaitGraph(int32_t graph_id) {
    return Call([&]() {
      auto it = graphs_.find(graph_id);
      if (it == graphs_.end()) {
        throw std::runtime_error("Typhoon graph " + std::to_string(graph_id) + " is unknown");
      }
      it->second.Wait();
      graphs_.erase(it);
    });
  }

 private:
  template <typename F>
  int Call(F&& f) {
    std::lock_guard<std::mutex> lock(mu_);
    try {
      std::forward<F>(f)();
      last_error_.clear();
      return 0;
    } catch (const std::exception& e) {
      last_error_ = e.what();
      return -1;
    }
  }

  TyphoonGraphBuilder& GetOrCreateGraph(int32_t graph_id) {
    auto it = graphs_.find(graph_id);
    if (it == graphs_.end()) {
      it = graphs_.emplace(graph_id, TyphoonGraphBuilder(graph_id)).first;
    }
    return it->second;
  }

  mutable std::mutex mu_;
  std::unordered_map<int32_t, TyphoonGraphBuilder> graphs_;
  std::string last_error_;
};

}  // namespace typhoon
}  // namespace runtime
}  // namespace tvm

extern "C" int TVMTyphoonGraphBegin(int32_t graph_id) {
  return tvm::runtime::typhoon::TyphoonRuntimeState::Global().GraphBegin(graph_id);
}

extern "C" int TVMTyphoonDeclareRegion(int32_t graph_id, int32_t region_id, int64_t offset,
                                       int64_t size, int64_t alignment, int32_t preinitialized,
                                       const char* tag) {
  return tvm::runtime::typhoon::TyphoonRuntimeState::Global().DeclareRegion(
      graph_id, region_id, offset, size, alignment, preinitialized, tag);
}

extern "C" int TVMTyphoonAddDMATask(int32_t graph_id, int32_t task_id, int32_t direction,
                                    void* global_handle, int64_t global_byte_offset,
                                    int32_t sram_region_id, int64_t bytes, int32_t num_deps,
                                    const int32_t* dep_ids) {
  return tvm::runtime::typhoon::TyphoonRuntimeState::Global().AddDMATask(
      graph_id, task_id, direction, global_handle, global_byte_offset, sram_region_id, bytes,
      num_deps, dep_ids);
}

extern "C" int TVMTyphoonAddMatmulTask(int32_t graph_id, int32_t task_id, int32_t a_region_id,
                                       int32_t b_region_id, int32_t c_region_id, int64_t m,
                                       int64_t n, int64_t k, int32_t dtype_code,
                                       int32_t layout_code, int32_t num_deps,
                                       const int32_t* dep_ids) {
  return tvm::runtime::typhoon::TyphoonRuntimeState::Global().AddMatmulTask(
      graph_id, task_id, a_region_id, b_region_id, c_region_id, m, n, k, dtype_code, layout_code,
      num_deps, dep_ids);
}

extern "C" int TVMTyphoonAddVectorTask(int32_t graph_id, int32_t task_id, int32_t op_code,
                                       int32_t in0_region_id, int32_t in1_region_id,
                                       int32_t out_region_id, int64_t elem_count,
                                       int32_t dtype_code, int32_t num_deps,
                                       const int32_t* dep_ids) {
  return tvm::runtime::typhoon::TyphoonRuntimeState::Global().AddVectorTask(
      graph_id, task_id, op_code, in0_region_id, in1_region_id, out_region_id, elem_count,
      dtype_code, num_deps, dep_ids);
}

extern "C" int TVMTyphoonAddReshapeTask(int32_t graph_id, int32_t task_id, int32_t in_region_id,
                                        int32_t out_region_id, int64_t elem_count,
                                        int32_t transform_code, int32_t num_deps,
                                        const int32_t* dep_ids) {
  return tvm::runtime::typhoon::TyphoonRuntimeState::Global().AddReshapeTask(
      graph_id, task_id, in_region_id, out_region_id, elem_count, transform_code, num_deps,
      dep_ids);
}

extern "C" int TVMTyphoonSubmitGraph(int32_t graph_id) {
  return tvm::runtime::typhoon::TyphoonRuntimeState::Global().SubmitGraph(graph_id);
}

extern "C" int TVMTyphoonWaitGraph(int32_t graph_id) {
  return tvm::runtime::typhoon::TyphoonRuntimeState::Global().WaitGraph(graph_id);
}

namespace tvm {
namespace runtime {
namespace typhoon {

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("runtime.typhoon.testing_reset",
                        []() { TyphoonRuntimeState::Global().ResetForTesting(); });
  refl::GlobalDef().def("runtime.typhoon.testing_last_error",
                        []() { return TyphoonRuntimeState::Global().LastError(); });
}

}  // namespace typhoon
}  // namespace runtime
}  // namespace tvm
