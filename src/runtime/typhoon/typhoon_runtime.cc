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
#include <tvm/runtime/threading_backend.h>

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

constexpr int64_t kTyphoonParallelOuterThreshold = 4;
constexpr int64_t kTyphoonParallelWorkThreshold = 1 << 15;

bool ShouldParallelize(int64_t outer_extent, int64_t total_work) {
  return outer_extent >= kTyphoonParallelOuterThreshold &&
         total_work >= kTyphoonParallelWorkThreshold;
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
          task.direction == 0 ? graph.GetRegion(task.writes.at(0)) : graph.GetRegion(task.reads.at(0));
      uint8_t* sram_ptr = SramPtr<>(sram, region) + task.sram_byte_offset;
      uint8_t* global_ptr = reinterpret_cast<uint8_t*>(task.global_endpoint.handle) + task.global_endpoint.byte_offset;
      if (task.direction == 0) {
        std::memcpy(sram_ptr, global_ptr, static_cast<size_t>(task.bytes));
      } else {
        std::memcpy(global_ptr, sram_ptr, static_cast<size_t>(task.bytes));
      }
      return;
    }
    case TaskKind::kReshape: {
      const auto& input = graph.GetRegion(task.reads.at(0));
      const auto& output = graph.GetRegion(task.writes.at(0));
      if (task.transform_code == 0) {
        std::memcpy(SramPtr<>(sram, output), SramPtr<>(sram, input), static_cast<size_t>(task.elem_count));
        return;
      }
      if (task.transform_code == 2) {
        auto* input_ptr = SramPtr<float>(sram, input);
        auto* output_ptr = SramPtr<float>(sram, output);
        int64_t rows = MetadataAt(task, 0, "reshape");
        int64_t cols = MetadataAt(task, 1, "reshape");
        auto transpose_row = [&](int64_t row) {
          for (int64_t col = 0; col < cols; ++col) {
            output_ptr[col * rows + row] = input_ptr[row * cols + col];
          }
        };
        if (ShouldParallelize(rows, rows * cols)) {
          tvm::runtime::parallel_for_with_threading_backend(transpose_row, 0, rows);
        } else {
          for (int64_t row = 0; row < rows; ++row) {
            transpose_row(row);
          }
        }
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
      int64_t input_origin_h = task.metadata.size() >= 16 ? MetadataAt(task, 12, "reshape") : 0;
      int64_t input_origin_w = task.metadata.size() >= 16 ? MetadataAt(task, 13, "reshape") : 0;
      int64_t output_origin_h = task.metadata.size() >= 16 ? MetadataAt(task, 14, "reshape") : 0;
      int64_t output_origin_w = task.metadata.size() >= 16 ? MetadataAt(task, 15, "reshape") : 0;
      int64_t patch_size = c * kernel_h * kernel_w;

      int64_t row_count = n * out_h * out_w;
      auto write_im2col_row = [&](int64_t row) {
        int64_t ni = row / (out_h * out_w);
        int64_t rem = row % (out_h * out_w);
        int64_t oh = rem / out_w;
        int64_t ow = rem % out_w;
        for (int64_t ci = 0; ci < c; ++ci) {
          for (int64_t kh = 0; kh < kernel_h; ++kh) {
            for (int64_t kw = 0; kw < kernel_w; ++kw) {
              int64_t ih = (output_origin_h + oh) * stride_h + kh - pad_h - input_origin_h;
              int64_t iw = (output_origin_w + ow) * stride_w + kw - pad_w - input_origin_w;
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
      };
      if (ShouldParallelize(row_count, row_count * patch_size)) {
        tvm::runtime::parallel_for_with_threading_backend(write_im2col_row, 0, row_count);
      } else {
        for (int64_t row = 0; row < row_count; ++row) {
          write_im2col_row(row);
        }
      }
      return;
    }
    case TaskKind::kVector: {
      if (task.dtype_code != 2) {
        throw std::runtime_error("Typhoon vector execution currently supports dtype_code=2 only");
      }
      const auto& in0 = graph.GetRegion(task.reads.at(0));
      const auto& out = graph.GetRegion(task.writes.at(0));
      auto* in0_ptr = SramPtr<float>(sram, in0);
      auto* out_ptr = SramPtr<float>(sram, out);

      if (task.op_code == 0) {
        const auto& in1 = graph.GetRegion(task.reads.at(1));
        auto* in1_ptr = SramPtr<float>(sram, in1);
        if (task.metadata.empty()) {
          for (int64_t i = 0; i < task.elem_count; ++i) {
            out_ptr[i] = in0_ptr[i] + in1_ptr[i];
          }
        } else {
          int64_t outer = MetadataAt(task, 0, "vector");
          int64_t inner = MetadataAt(task, 1, "vector");
          for (int64_t i = 0; i < outer; ++i) {
            for (int64_t j = 0; j < inner; ++j) {
              out_ptr[i * inner + j] = in0_ptr[i * inner + j] + in1_ptr[j];
            }
          }
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
        int64_t output_count = n * c * out_h * out_w;
        auto compute_maxpool_output = [&](int64_t output_index) {
          int64_t tmp = output_index;
          int64_t ow = tmp % out_w;
          tmp /= out_w;
          int64_t oh = tmp % out_h;
          tmp /= out_h;
          int64_t ci = tmp % c;
          int64_t ni = tmp / c;
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
          out_ptr[output_index] = best;
        };
        if (ShouldParallelize(output_count, output_count * kernel_h * kernel_w)) {
          tvm::runtime::parallel_for_with_threading_backend(compute_maxpool_output, 0,
                                                            output_count);
        } else {
          for (int64_t output_index = 0; output_index < output_count; ++output_index) {
            compute_maxpool_output(output_index);
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
        int64_t output_count = n * c;
        auto compute_gap_output = [&](int64_t output_index) {
          int64_t ni = output_index / c;
          int64_t ci = output_index % c;
          float sum = 0.0f;
          for (int64_t ih = 0; ih < in_h; ++ih) {
            for (int64_t iw = 0; iw < in_w; ++iw) {
              int64_t input_index = ((ni * c + ci) * in_h + ih) * in_w + iw;
              sum += in0_ptr[input_index];
            }
          }
          out_ptr[output_index] = sum / denom;
        };
        if (ShouldParallelize(output_count, output_count * in_h * in_w)) {
          tvm::runtime::parallel_for_with_threading_backend(compute_gap_output, 0, output_count);
        } else {
          for (int64_t output_index = 0; output_index < output_count; ++output_index) {
            compute_gap_output(output_index);
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
      const auto& a = graph.GetRegion(task.reads.at(0));
      const auto& b = graph.GetRegion(task.reads.at(1));
      const auto& c = graph.GetRegion(task.writes.at(0));
      auto* a_ptr = SramPtr<float>(sram, a);
      auto* b_ptr = SramPtr<float>(sram, b);
      auto* c_ptr = SramPtr<float>(sram, c);
      auto compute_output_row = [&](int64_t i) {
        for (int64_t j = 0; j < task.n; ++j) {
          float acc = 0.0f;
          for (int64_t kk = 0; kk < task.k; ++kk) {
            float rhs = 0.0f;
            if (task.layout_code == 0) {
              rhs = b_ptr[kk * task.n + j];
            } else if (task.layout_code == 1) {
              rhs = b_ptr[j * task.k + kk];
            } else {
              throw std::runtime_error("Typhoon matmul execution uses unsupported layout_code " +
                                       std::to_string(task.layout_code));
            }
            acc += a_ptr[i * task.k + kk] * rhs;
          }
          c_ptr[i * task.n + j] = acc;
        }
      };
      if (ShouldParallelize(task.m, task.m * task.n * task.k)) {
        tvm::runtime::parallel_for_with_threading_backend(compute_output_row, 0, task.m);
      } else {
        for (int64_t i = 0; i < task.m; ++i) {
          compute_output_row(i);
        }
      }
      return;
    }
  }
}

void ExecuteGraph(const TyphoonGraphBuilder& graph) {
  std::vector<uint8_t> sram(kTyphoonDefaultSRAMSize, 0);
  const auto& tasks = graph.tasks();
  for (size_t task_index : graph.topo_order()) {
    ExecuteTask(graph, tasks[task_index], &sram);
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
                 int64_t global_byte_offset, int32_t sram_region_id,
                 int64_t sram_byte_offset, int64_t bytes, int32_t num_deps,
                 const int32_t* dep_ids) {
    return Call([&]() {
      GetOrCreateGraph(graph_id)
          .AddDMATask(task_id, direction, global_handle, global_byte_offset, sram_region_id,
                      sram_byte_offset, bytes, num_deps, dep_ids);
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
                                    int32_t sram_region_id, int64_t sram_byte_offset,
                                    int64_t bytes, int32_t num_deps, const int32_t* dep_ids) {
  return tvm::runtime::typhoon::TyphoonRuntimeState::Global().AddDMATask(
      graph_id, task_id, direction, global_handle, global_byte_offset, sram_region_id,
      sram_byte_offset, bytes, num_deps, dep_ids);
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
