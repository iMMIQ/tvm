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
#include "typhoon_scheduler.h"

#include <algorithm>
#include <cstring>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>

#include <mutex>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace tvm {
namespace runtime {
namespace typhoon {

namespace {

const TyphoonRegion& FindRegion(const TyphoonGraphBuilder& graph, int32_t region_id) {
  for (const auto& region : graph.regions()) {
    if (region.region_id == region_id) {
      return region;
    }
  }
  throw std::runtime_error("Unknown Typhoon region " + std::to_string(region_id));
}

template <typename T = uint8_t>
T* SramPtr(std::vector<uint8_t>* sram, const TyphoonRegion& region) {
  return reinterpret_cast<T*>(sram->data() + region.offset);
}

int64_t MetadataAt(const TyphoonTask& task, size_t index, const char* task_kind) {
  if (index >= task.metadata.size()) {
    throw std::runtime_error(std::string("Typhoon ") + task_kind + " task " +
                             std::to_string(task.task_id) + " metadata is incomplete");
  }
  return task.metadata[index];
}

void ExecuteTask(const TyphoonGraphBuilder& graph, const TyphoonTask& task, std::vector<uint8_t>* sram) {
  switch (task.kind) {
    case TaskKind::kDMA: {
      const auto& region =
          task.direction == 0 ? FindRegion(graph, task.writes.at(0)) : FindRegion(graph, task.reads.at(0));
      uint8_t* sram_ptr = SramPtr<>(sram, region);
      uint8_t* global_ptr = reinterpret_cast<uint8_t*>(task.global_endpoint.handle) + task.global_endpoint.byte_offset;
      if (task.direction == 0) {
        std::memcpy(sram_ptr, global_ptr, static_cast<size_t>(task.bytes));
      } else {
        std::memcpy(global_ptr, sram_ptr, static_cast<size_t>(task.bytes));
      }
      return;
    }
    case TaskKind::kReshape: {
      const auto& input = FindRegion(graph, task.reads.at(0));
      const auto& output = FindRegion(graph, task.writes.at(0));
      if (task.transform_code == 0) {
        std::memcpy(SramPtr<>(sram, output), SramPtr<>(sram, input), static_cast<size_t>(task.elem_count));
        return;
      }
      if (task.transform_code != 1) {
        throw std::runtime_error("Typhoon reshape execution uses unsupported transform_code " +
                                 std::to_string(task.transform_code));
      }

      auto* input_ptr = SramPtr<float>(sram, input);
      auto* output_ptr = SramPtr<float>(sram, output);
      int64_t n = MetadataAt(task, 0, "reshape");
      int64_t c = MetadataAt(task, 1, "reshape");
      int64_t in_h = MetadataAt(task, 2, "reshape");
      int64_t in_w = MetadataAt(task, 3, "reshape");
      int64_t kernel_h = MetadataAt(task, 4, "reshape");
      int64_t kernel_w = MetadataAt(task, 5, "reshape");
      int64_t stride_h = MetadataAt(task, 6, "reshape");
      int64_t stride_w = MetadataAt(task, 7, "reshape");
      int64_t pad_h = MetadataAt(task, 8, "reshape");
      int64_t pad_w = MetadataAt(task, 9, "reshape");
      int64_t out_h = MetadataAt(task, 10, "reshape");
      int64_t out_w = MetadataAt(task, 11, "reshape");
      int64_t patch_size = c * kernel_h * kernel_w;

      for (int64_t ni = 0; ni < n; ++ni) {
        for (int64_t oh = 0; oh < out_h; ++oh) {
          for (int64_t ow = 0; ow < out_w; ++ow) {
            int64_t row = ((ni * out_h) + oh) * out_w + ow;
            for (int64_t ci = 0; ci < c; ++ci) {
              for (int64_t kh = 0; kh < kernel_h; ++kh) {
                for (int64_t kw = 0; kw < kernel_w; ++kw) {
                  int64_t ih = oh * stride_h + kh - pad_h;
                  int64_t iw = ow * stride_w + kw - pad_w;
                  int64_t col = ((ci * kernel_h) + kh) * kernel_w + kw;
                  float value = 0.0f;
                  if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                    int64_t input_index = ((ni * c + ci) * in_h + ih) * in_w + iw;
                    value = input_ptr[input_index];
                  }
                  output_ptr[row * patch_size + col] = value;
                }
              }
            }
          }
        }
      }
      return;
    }
    case TaskKind::kVector: {
      if (task.dtype_code != 2) {
        throw std::runtime_error("Typhoon vector execution currently supports dtype_code=2 only");
      }
      const auto& in0 = FindRegion(graph, task.reads.at(0));
      const auto& out = FindRegion(graph, task.writes.at(0));
      auto* in0_ptr = SramPtr<float>(sram, in0);
      auto* out_ptr = SramPtr<float>(sram, out);

      if (task.op_code == 0) {
        const auto& in1 = FindRegion(graph, task.reads.at(1));
        auto* in1_ptr = SramPtr<float>(sram, in1);
        for (int64_t i = 0; i < task.elem_count; ++i) {
          out_ptr[i] = in0_ptr[i] + in1_ptr[i];
        }
        return;
      }

      if (task.op_code == 1) {
        for (int64_t i = 0; i < task.elem_count; ++i) {
          out_ptr[i] = std::max(in0_ptr[i], 0.0f);
        }
        return;
      }

      if (task.op_code == 2) {
        int64_t n = MetadataAt(task, 0, "vector");
        int64_t c = MetadataAt(task, 1, "vector");
        int64_t in_h = MetadataAt(task, 2, "vector");
        int64_t in_w = MetadataAt(task, 3, "vector");
        int64_t kernel_h = MetadataAt(task, 4, "vector");
        int64_t kernel_w = MetadataAt(task, 5, "vector");
        int64_t stride_h = MetadataAt(task, 6, "vector");
        int64_t stride_w = MetadataAt(task, 7, "vector");
        int64_t pad_h = MetadataAt(task, 8, "vector");
        int64_t pad_w = MetadataAt(task, 9, "vector");
        int64_t out_h = MetadataAt(task, 10, "vector");
        int64_t out_w = MetadataAt(task, 11, "vector");
        for (int64_t ni = 0; ni < n; ++ni) {
          for (int64_t ci = 0; ci < c; ++ci) {
            for (int64_t oh = 0; oh < out_h; ++oh) {
              for (int64_t ow = 0; ow < out_w; ++ow) {
                float best = -std::numeric_limits<float>::infinity();
                for (int64_t kh = 0; kh < kernel_h; ++kh) {
                  for (int64_t kw = 0; kw < kernel_w; ++kw) {
                    int64_t ih = oh * stride_h + kh - pad_h;
                    int64_t iw = ow * stride_w + kw - pad_w;
                    if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                      int64_t input_index = ((ni * c + ci) * in_h + ih) * in_w + iw;
                      best = std::max(best, in0_ptr[input_index]);
                    }
                  }
                }
                int64_t output_index = ((ni * c + ci) * out_h + oh) * out_w + ow;
                out_ptr[output_index] = best;
              }
            }
          }
        }
        return;
      }

      if (task.op_code == 3) {
        int64_t n = MetadataAt(task, 0, "vector");
        int64_t c = MetadataAt(task, 1, "vector");
        int64_t in_h = MetadataAt(task, 2, "vector");
        int64_t in_w = MetadataAt(task, 3, "vector");
        float denom = static_cast<float>(in_h * in_w);
        for (int64_t ni = 0; ni < n; ++ni) {
          for (int64_t ci = 0; ci < c; ++ci) {
            float sum = 0.0f;
            for (int64_t ih = 0; ih < in_h; ++ih) {
              for (int64_t iw = 0; iw < in_w; ++iw) {
                int64_t input_index = ((ni * c + ci) * in_h + ih) * in_w + iw;
                sum += in0_ptr[input_index];
              }
            }
            out_ptr[ni * c + ci] = sum / denom;
          }
        }
        return;
      }

      throw std::runtime_error("Typhoon vector execution uses unsupported op_code " +
                               std::to_string(task.op_code));
    }
    case TaskKind::kMatmul: {
      if (task.dtype_code != 2) {
        throw std::runtime_error("Typhoon matmul execution currently supports dtype_code=2 only");
      }
      const auto& a = FindRegion(graph, task.reads.at(0));
      const auto& b = FindRegion(graph, task.reads.at(1));
      const auto& c = FindRegion(graph, task.writes.at(0));
      auto* a_ptr = SramPtr<float>(sram, a);
      auto* b_ptr = SramPtr<float>(sram, b);
      auto* c_ptr = SramPtr<float>(sram, c);
      for (int64_t i = 0; i < task.m; ++i) {
        for (int64_t j = 0; j < task.n; ++j) {
          float acc = 0.0f;
          for (int64_t kk = 0; kk < task.k; ++kk) {
            acc += a_ptr[i * task.k + kk] * b_ptr[kk * task.n + j];
          }
          c_ptr[i * task.n + j] = acc;
        }
      }
      return;
    }
  }
}

void ExecuteGraph(const TyphoonGraphBuilder& graph) {
  const auto& tasks = graph.tasks();
  std::unordered_map<int32_t, size_t> task_index = graph.task_index();
  std::vector<int> remaining_deps(tasks.size(), 0);
  std::vector<std::vector<size_t>> users(tasks.size());
  for (size_t i = 0; i < tasks.size(); ++i) {
    remaining_deps[i] = static_cast<int>(tasks[i].deps.size());
    for (int32_t dep_id : tasks[i].deps) {
      users.at(task_index.at(dep_id)).push_back(i);
    }
  }

  std::vector<uint8_t> sram(kTyphoonDefaultSRAMSize, 0);
  std::vector<size_t> ready;
  for (size_t i = 0; i < tasks.size(); ++i) {
    if (remaining_deps[i] == 0) {
      ready.push_back(i);
    }
  }

  auto ready_cmp = [&tasks](size_t lhs, size_t rhs) {
    if (tasks[lhs].task_id != tasks[rhs].task_id) {
      return tasks[lhs].task_id < tasks[rhs].task_id;
    }
    return lhs < rhs;
  };

  while (!ready.empty()) {
    std::sort(ready.begin(), ready.end(), ready_cmp);
    size_t current = ready.front();
    ready.erase(ready.begin());
    ExecuteTask(graph, tasks[current], &sram);
    for (size_t user : users[current]) {
      --remaining_deps[user];
      if (remaining_deps[user] == 0) {
        ready.push_back(user);
      }
    }
  }
}

}  // namespace

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
    last_trace_json_ = "[]";
  }

  std::string LastError() const {
    std::lock_guard<std::mutex> lock(mu_);
    return last_error_;
  }

  std::string LastTraceJSON() const {
    std::lock_guard<std::mutex> lock(mu_);
    return last_trace_json_;
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
                    int32_t dtype_code, int32_t num_metadata, const int64_t* metadata,
                    int32_t num_deps, const int32_t* dep_ids) {
    return Call([&]() {
      GetOrCreateGraph(graph_id)
          .AddVectorTask(task_id, op_code, in0_region_id, in1_region_id, out_region_id,
                         elem_count, dtype_code, num_metadata, metadata, num_deps, dep_ids);
    });
  }

  int AddReshapeTask(int32_t graph_id, int32_t task_id, int32_t in_region_id,
                     int32_t out_region_id, int64_t elem_count, int32_t transform_code,
                     int32_t num_metadata, const int64_t* metadata, int32_t num_deps,
                     const int32_t* dep_ids) {
    return Call([&]() {
      GetOrCreateGraph(graph_id)
          .AddReshapeTask(task_id, in_region_id, out_region_id, elem_count, transform_code,
                          num_metadata, metadata, num_deps, dep_ids);
    });
  }

  int SubmitGraph(int32_t graph_id) {
    return Call([&]() {
      auto& graph = GetOrCreateGraph(graph_id);
      graph.Submit();
      ExecuteGraph(graph);
      last_trace_json_ = SerializeTraceToJSON(TyphoonScheduler().Run(graph));
    });
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
  std::string last_trace_json_{"[]"};
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
                                       int32_t dtype_code, int32_t num_metadata,
                                       const int64_t* metadata, int32_t num_deps,
                                       const int32_t* dep_ids) {
  return tvm::runtime::typhoon::TyphoonRuntimeState::Global().AddVectorTask(
      graph_id, task_id, op_code, in0_region_id, in1_region_id, out_region_id, elem_count,
      dtype_code, num_metadata, metadata, num_deps, dep_ids);
}

extern "C" int TVMTyphoonAddReshapeTask(int32_t graph_id, int32_t task_id, int32_t in_region_id,
                                        int32_t out_region_id, int64_t elem_count,
                                        int32_t transform_code, int32_t num_metadata,
                                        const int64_t* metadata, int32_t num_deps,
                                        const int32_t* dep_ids) {
  return tvm::runtime::typhoon::TyphoonRuntimeState::Global().AddReshapeTask(
      graph_id, task_id, in_region_id, out_region_id, elem_count, transform_code, num_metadata,
      metadata, num_deps, dep_ids);
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
  refl::GlobalDef().def("runtime.typhoon_get_last_trace_json",
                        []() { return TyphoonRuntimeState::Global().LastTraceJSON(); });
}

}  // namespace typhoon
}  // namespace runtime
}  // namespace tvm
