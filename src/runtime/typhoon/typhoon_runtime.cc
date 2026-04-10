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
#include <array>
#include <cstring>
#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/threading_backend.h>

#include <mutex>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace tvm {
namespace runtime {
namespace typhoon {

namespace {

enum CanonicalReplayKind : int32_t {
  kReplayConv2D = 1,
  kReplayRelu = 2,
  kReplayAdd = 3,
  kReplayMaxPool = 4,
  kReplayGlobalAveragePool = 5,
  kReplayDenseMatmul = 6,
  kReplayCopy = 7,
};

struct CanonicalRegionSpec {
  int32_t region_id;
  int64_t offset;
  int64_t size;
  int64_t alignment;
  bool preinitialized;
  const char* tag;
};

constexpr CanonicalRegionSpec kCanonicalRegions[] = {
    {0, 0, 262144, 64, false, "act0"},       {1, 262144, 262144, 64, false, "act1"},
    {2, 524288, 262144, 64, false, "wgt0"},  {3, 786432, 131072, 64, false, "col0"},
    {4, 917504, 65536, 64, false, "aux0"},   {5, 983040, 65536, 64, false, "residual"},
};

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
      uint8_t* global_ptr =
          reinterpret_cast<uint8_t*>(task.global_endpoint.handle) + task.global_endpoint.byte_offset;
      for (int64_t batch = 0; batch < task.dma_batch_count; ++batch) {
        uint8_t* batch_sram_ptr = sram_ptr + batch * task.dma_sram_stride;
        uint8_t* batch_global_ptr = global_ptr + batch * task.dma_global_stride;
        if (task.direction == 0) {
          std::memcpy(batch_sram_ptr, batch_global_ptr, static_cast<size_t>(task.bytes));
        } else {
          std::memcpy(batch_global_ptr, batch_sram_ptr, static_cast<size_t>(task.bytes));
        }
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

      if (task.op_code == 4) {
        const auto& in1 = graph.GetRegion(task.reads.at(1));
        auto* in1_ptr = SramPtr<float>(sram, in1);
        int64_t channels = MetadataAt(task, 0, "vector");
        int64_t plane_elems = MetadataAt(task, 1, "vector");
        auto compute_channel = [&](int64_t channel) {
          float bias = in1_ptr[channel];
          int64_t base = channel * plane_elems;
          for (int64_t i = 0; i < plane_elems; ++i) {
            out_ptr[base + i] = in0_ptr[base + i] + bias;
          }
        };
        if (ShouldParallelize(channels, task.elem_count)) {
          tvm::runtime::parallel_for_with_threading_backend(compute_channel, 0, channels);
        } else {
          for (int64_t channel = 0; channel < channels; ++channel) {
            compute_channel(channel);
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

const CanonicalRegionSpec& FindCanonicalRegion(const char* tag) {
  for (const CanonicalRegionSpec& region : kCanonicalRegions) {
    if (std::strcmp(region.tag, tag) == 0) {
      return region;
    }
  }
  throw std::runtime_error(std::string("Unknown canonical Typhoon region `") + tag + "`");
}

}  // namespace

struct CapturedCall {
  int32_t num_handles{0};
  std::array<DLTensor, 3> handles{};
  std::array<bool, 3> present{{false, false, false}};
  std::array<std::vector<int64_t>, 3> shapes;
  std::array<std::vector<int64_t>, 3> strides;

  CapturedCall() = default;

  CapturedCall(const CapturedCall& other)
      : num_handles(other.num_handles),
        handles(other.handles),
        present(other.present),
        shapes(other.shapes),
        strides(other.strides) {
    RebindTensorMetadata();
  }

  CapturedCall(CapturedCall&& other) noexcept
      : num_handles(other.num_handles),
        handles(other.handles),
        present(other.present),
        shapes(std::move(other.shapes)),
        strides(std::move(other.strides)) {
    RebindTensorMetadata();
  }

  CapturedCall& operator=(const CapturedCall& other) {
    if (this != &other) {
      num_handles = other.num_handles;
      handles = other.handles;
      present = other.present;
      shapes = other.shapes;
      strides = other.strides;
      RebindTensorMetadata();
    }
    return *this;
  }

  CapturedCall& operator=(CapturedCall&& other) noexcept {
    if (this != &other) {
      num_handles = other.num_handles;
      handles = other.handles;
      present = other.present;
      shapes = std::move(other.shapes);
      strides = std::move(other.strides);
      RebindTensorMetadata();
    }
    return *this;
  }

 private:
  void RebindTensorMetadata() {
    for (size_t i = 0; i < handles.size(); ++i) {
      if (!present[i]) {
        handles[i].shape = nullptr;
        handles[i].strides = nullptr;
        continue;
      }
      handles[i].shape = shapes[i].data();
      handles[i].strides = strides[i].empty() ? nullptr : strides[i].data();
    }
  }
};

struct CanonicalReplayState {
  bool initialized{false};
  int32_t next_task_id{0};
  std::vector<int32_t> prev_segment_deps;
  std::unordered_set<int32_t> skipped_layer_ids;
};

struct GraphRuntimeState {
  explicit GraphRuntimeState(int32_t graph_id) : graph(graph_id) {}

  TyphoonGraphBuilder graph;
  std::vector<CapturedCall> captured_calls;
  std::unordered_map<std::string, int32_t> planned_capture_positions;
  CanonicalReplayState replay;
};

struct GraphTaskStats {
  int64_t total{0};
  int64_t dma{0};
  int64_t matmul{0};
  int64_t vector{0};
  int64_t reshape{0};
};

class TyphoonRuntimeState {
 public:
  static TyphoonRuntimeState& Global() {
    static TyphoonRuntimeState runtime;
    return runtime;
  }

  void ResetForTesting() {
    std::lock_guard<std::mutex> lock(mu_);
    graphs_.clear();
    keep_graph_state_for_testing_ = false;
    last_error_.clear();
    last_trace_json_ = "[]";
    last_graph_stats_json_ = "{}";
  }

  void SetKeepGraphStateForTesting(bool keep) {
    std::lock_guard<std::mutex> lock(mu_);
    keep_graph_state_for_testing_ = keep;
  }

  std::string LastError() const {
    std::lock_guard<std::mutex> lock(mu_);
    return last_error_;
  }

  std::string LastTraceJSON() const {
    std::lock_guard<std::mutex> lock(mu_);
    return last_trace_json_;
  }

  std::string LastGraphStatsJSON() const {
    std::lock_guard<std::mutex> lock(mu_);
    return last_graph_stats_json_;
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

  int AddBatchedDMATask(int32_t graph_id, int32_t task_id, int32_t direction, void* global_handle,
                        int64_t global_byte_offset, int64_t global_stride,
                        int32_t sram_region_id, int64_t sram_byte_offset, int64_t sram_stride,
                        int64_t bytes, int64_t batch_count, int32_t num_deps,
                        const int32_t* dep_ids) {
    return Call([&]() {
      GetOrCreateGraph(graph_id)
          .AddBatchedDMATask(task_id, direction, global_handle, global_byte_offset, global_stride,
                             sram_region_id, sram_byte_offset, sram_stride, bytes, batch_count,
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

  int CaptureCall(int32_t graph_id, int32_t num_handles, void* handle0, void* handle1, void* handle2) {
    return CaptureCallAt(graph_id, NextCaptureLayerId(graph_id), num_handles, handle0, handle1, handle2);
  }

  int CaptureCallAt(int32_t graph_id, int32_t layer_id, int32_t num_handles, void* handle0,
                    void* handle1, void* handle2) {
    return Call([&]() {
      if (layer_id < 0) {
        throw std::runtime_error("Typhoon capture layer_id must be non-negative");
      }
      if (num_handles < 0 || num_handles > 3) {
        throw std::runtime_error("Typhoon capture num_handles must be in [0, 3]");
      }
      CapturedCall call;
      call.num_handles = num_handles;
      std::array<DLTensor*, 3> handles = {static_cast<DLTensor*>(handle0),
                                          static_cast<DLTensor*>(handle1),
                                          static_cast<DLTensor*>(handle2)};
      for (int32_t i = 0; i < num_handles; ++i) {
        SnapshotTensor(handles[static_cast<size_t>(i)], &call, i);
      }
      StoreCapturedCall(graph_id, layer_id, std::move(call));
    });
  }

  int CaptureCallPlanned(int32_t graph_id, const char* layer_ids_csv, int32_t num_handles,
                         void* handle0, void* handle1, void* handle2) {
    return CaptureCallAt(graph_id, ResolvePlannedCaptureLayerId(graph_id, layer_ids_csv), num_handles,
                         handle0, handle1, handle2);
  }

  int CapturePackedArgs(int32_t graph_id, const TVMFFIAny* packed_args, int32_t num_args) {
    return CapturePackedArgsAt(graph_id, NextCaptureLayerId(graph_id), packed_args, num_args);
  }

  int CapturePackedArgsAt(int32_t graph_id, int32_t layer_id, const TVMFFIAny* packed_args,
                          int32_t num_args) {
    return Call([&]() {
      if (layer_id < 0) {
        throw std::runtime_error("Typhoon packed capture layer_id must be non-negative");
      }
      if (num_args < 0 || num_args > 3) {
        throw std::runtime_error("Typhoon packed capture num_args must be in [0, 3]");
      }
      if (num_args != 0 && packed_args == nullptr) {
        throw std::runtime_error("Typhoon packed capture args pointer is NULL");
      }
      CapturedCall call;
      call.num_handles = num_args;
      for (int32_t i = 0; i < num_args; ++i) {
        auto tensor = ffi::TypeTraits<DLTensor*>::TryCastFromAnyView(&packed_args[i]);
        if (!tensor.has_value() || tensor.value() == nullptr) {
          throw std::runtime_error("Typhoon packed capture arg " + std::to_string(i) +
                                   " is not a tensor handle");
        }
        SnapshotTensor(tensor.value(), &call, i);
      }
      StoreCapturedCall(graph_id, layer_id, std::move(call));
    });
  }

  int CapturePackedArgsPlanned(int32_t graph_id, const char* layer_ids_csv,
                               const TVMFFIAny* packed_args, int32_t num_args) {
    return CapturePackedArgsAt(graph_id, ResolvePlannedCaptureLayerId(graph_id, layer_ids_csv),
                               packed_args, num_args);
  }

  void* GetCapturedHandle(int32_t graph_id, int32_t layer_id, int32_t handle_index) {
    return CallHandle([&]() -> void* {
      if (handle_index < 0 || handle_index >= 3) {
        throw std::runtime_error("Typhoon captured handle index must be in [0, 2]");
      }
      auto it = graphs_.find(graph_id);
      if (it == graphs_.end()) {
        throw std::runtime_error("Typhoon graph " + std::to_string(graph_id) + " is unknown");
      }
      if (layer_id < 0 || layer_id >= static_cast<int32_t>(it->second.captured_calls.size())) {
        throw std::runtime_error("Typhoon captured layer_id " + std::to_string(layer_id) +
                                 " is out of range");
      }
      const CapturedCall& call = it->second.captured_calls[layer_id];
      if (handle_index >= call.num_handles) {
        throw std::runtime_error("Typhoon captured handle index " + std::to_string(handle_index) +
                                 " exceeds captured arity");
      }
      if (!call.present[handle_index]) {
        throw std::runtime_error("Typhoon captured tensor handle is NULL");
      }
      return call.handles[handle_index].data;
    });
  }

  void* GetCapturedTensorPtr(int32_t graph_id, int32_t layer_id, int32_t handle_index) {
    return CallHandle([&]() -> void* {
      if (handle_index < 0 || handle_index >= 3) {
        throw std::runtime_error("Typhoon captured tensor index must be in [0, 2]");
      }
      auto it = graphs_.find(graph_id);
      if (it == graphs_.end()) {
        throw std::runtime_error("Typhoon graph " + std::to_string(graph_id) + " is unknown");
      }
      if (layer_id < 0 || layer_id >= static_cast<int32_t>(it->second.captured_calls.size())) {
        throw std::runtime_error("Typhoon captured layer_id " + std::to_string(layer_id) +
                                 " is out of range");
      }
      const CapturedCall& call = it->second.captured_calls[layer_id];
      if (handle_index >= call.num_handles) {
        throw std::runtime_error("Typhoon captured tensor index " + std::to_string(handle_index) +
                                 " exceeds captured arity");
      }
      if (!call.present[handle_index]) {
        throw std::runtime_error("Typhoon captured tensor is NULL");
      }
      return reinterpret_cast<void*>(const_cast<DLTensor*>(&call.handles[handle_index]));
    });
  }

  int ReplayWholeGraphBegin(int32_t graph_id) {
    return Call([&]() {
      auto& state = GetOrCreateState(graph_id);
      for (const CanonicalRegionSpec& region : kCanonicalRegions) {
        state.graph.DeclareRegion(region.region_id, region.offset, region.size, region.alignment,
                                  region.preinitialized, region.tag);
      }
      state.replay = CanonicalReplayState{};
      state.replay.initialized = true;
    });
  }

  int ReplayCapturedLayer(int32_t graph_id, int32_t layer_id, int32_t replay_kind) {
    return Call([&]() {
      auto& state = GetOrCreateState(graph_id);
      EnsureReplayInitialized(state, graph_id);
      if (state.replay.skipped_layer_ids.count(layer_id) != 0) {
        return;
      }
      state.graph.SetCurrentLayerId(layer_id);
      struct ResetCurrentLayerId {
        TyphoonGraphBuilder* graph;
        ~ResetCurrentLayerId() { graph->SetCurrentLayerId(-1); }
      } reset_current_layer_id{&state.graph};
      switch (replay_kind) {
        case kReplayConv2D:
          ReplayConv2DLayer(&state, layer_id);
          return;
        case kReplayRelu:
          ReplayReluLayer(&state, layer_id);
          return;
        case kReplayAdd:
          ReplayAddLayer(&state, layer_id);
          return;
        case kReplayMaxPool:
          ReplayMaxPoolLayer(&state, layer_id);
          return;
        case kReplayGlobalAveragePool:
          ReplayGlobalAveragePoolLayer(&state, layer_id);
          return;
        case kReplayDenseMatmul:
          ReplayDenseMatmulLayer(&state, layer_id);
          return;
        case kReplayCopy:
          ReplayCopyLayer(&state, layer_id);
          return;
        default:
          throw std::runtime_error("Typhoon replay kind " + std::to_string(replay_kind) +
                                   " is unsupported");
      }
    });
  }

  int SubmitGraph(int32_t graph_id) {
    return Call([&]() {
      auto& graph = GetOrCreateState(graph_id).graph;
      graph.Submit();
      last_graph_stats_json_ = SerializeGraphStatsJSON(ComputeGraphTaskStats(graph));
      ExecuteGraph(graph);
      last_trace_json_ = SerializeTraceToJSON(TyphoonScheduler().Run(graph));
    });
  }

  int WaitGraph(int32_t graph_id) {
    if (!LastError().empty()) {
      std::lock_guard<std::mutex> lock(mu_);
      auto it = graphs_.find(graph_id);
      if (it != graphs_.end()) {
        try {
          it->second.graph.Wait();
        } catch (const std::exception&) {
        }
      }
      return 0;
    }
    return Call([&]() {
      auto it = graphs_.find(graph_id);
      if (it == graphs_.end()) {
        throw std::runtime_error("Typhoon graph " + std::to_string(graph_id) + " is unknown");
      }
      it->second.graph.Wait();
      if (keep_graph_state_for_testing_) {
        it->second.graph = TyphoonGraphBuilder(graph_id);
      } else {
        graphs_.erase(it);
      }
    });
  }

 private:
  static std::vector<int64_t> ShapeOf(const DLTensor* tensor) {
    if (tensor == nullptr) {
      throw std::runtime_error("Typhoon replay received a NULL DLTensor");
    }
    std::vector<int64_t> shape(static_cast<size_t>(tensor->ndim));
    for (int i = 0; i < tensor->ndim; ++i) {
      shape[static_cast<size_t>(i)] = tensor->shape[i];
    }
    return shape;
  }

  static int64_t TensorBytes(const DLTensor* tensor) {
    int64_t elems = 1;
    for (int i = 0; i < tensor->ndim; ++i) {
      elems *= tensor->shape[i];
    }
    return elems * (tensor->dtype.bits / 8) * tensor->dtype.lanes;
  }

  static bool SameShape(const DLTensor* lhs, const DLTensor* rhs) {
    if (lhs->ndim != rhs->ndim) {
      return false;
    }
    for (int i = 0; i < lhs->ndim; ++i) {
      if (lhs->shape[i] != rhs->shape[i]) {
        return false;
      }
    }
    return true;
  }

  static bool IsContiguousBiasVector(const DLTensor* tensor, int64_t elements) {
    if (tensor->dtype.code != kDLFloat || tensor->dtype.bits != 32 || tensor->dtype.lanes != 1) {
      return false;
    }
    if (tensor->strides != nullptr) {
      return false;
    }
    if (tensor->ndim == 1) {
      return tensor->shape[0] == elements;
    }
    if (tensor->ndim == 2) {
      return tensor->shape[0] == 1 && tensor->shape[1] == elements;
    }
    if (tensor->ndim == 4) {
      return tensor->shape[0] == 1 && tensor->shape[1] == elements && tensor->shape[2] == 1 &&
             tensor->shape[3] == 1;
    }
    return false;
  }

  static int64_t NCHWChannelOffsetBytes(int64_t channel, int64_t h_offset, int64_t height,
                                        int64_t width) {
    return ((channel * height) + h_offset) * width * 4;
  }

  static int64_t NCHWChannelPlaneBytes(int64_t height, int64_t width) {
    return height * width * 4;
  }

  static int64_t ComputeConvOutputExtent(int64_t input_extent, int64_t kernel_extent,
                                         int64_t stride, int64_t pad) {
    return (input_extent + 2 * pad - kernel_extent) / stride + 1;
  }

  static std::pair<int64_t, int64_t> InferConvStrideAndPad(int64_t input_extent,
                                                           int64_t output_extent,
                                                           int64_t kernel_extent) {
    for (int64_t stride = 1; stride <= 4; ++stride) {
      for (int64_t pad = 0; pad <= kernel_extent; ++pad) {
        int64_t numer = input_extent + 2 * pad - kernel_extent;
        if (numer < 0) {
          continue;
        }
        if (ComputeConvOutputExtent(input_extent, kernel_extent, stride, pad) == output_extent) {
          return {stride, pad};
        }
      }
    }
    throw std::runtime_error("Typhoon replay could not infer canonical conv stride/pad");
  }

  static int32_t NextTaskId(GraphRuntimeState* state) { return state->replay.next_task_id++; }

  static const CapturedCall& GetCapturedCall(const GraphRuntimeState& state, int32_t layer_id) {
    if (layer_id < 0 || layer_id >= static_cast<int32_t>(state.captured_calls.size())) {
      throw std::runtime_error("Typhoon captured layer_id " + std::to_string(layer_id) +
                               " is out of range");
    }
    return state.captured_calls[static_cast<size_t>(layer_id)];
  }

  static const DLTensor* GetCapturedTensor(const GraphRuntimeState& state, int32_t layer_id,
                                           int32_t handle_index) {
    const CapturedCall& call = GetCapturedCall(state, layer_id);
    if (handle_index < 0 || handle_index >= call.num_handles) {
      throw std::runtime_error("Typhoon captured tensor index " + std::to_string(handle_index) +
                               " exceeds captured arity");
    }
    if (!call.present[handle_index]) {
      throw std::runtime_error("Typhoon captured tensor is NULL");
    }
    return &call.handles[handle_index];
  }

  static void SnapshotTensor(const DLTensor* tensor, CapturedCall* call, int32_t handle_index) {
    if (tensor == nullptr) {
      throw std::runtime_error("Typhoon capture received a NULL DLTensor");
    }
    size_t index = static_cast<size_t>(handle_index);
    call->handles[index] = *tensor;
    call->present[index] = true;

    call->shapes[index].assign(tensor->shape, tensor->shape + tensor->ndim);
    call->handles[index].shape = call->shapes[index].data();

    if (tensor->strides != nullptr) {
      call->strides[index].assign(tensor->strides, tensor->strides + tensor->ndim);
      call->handles[index].strides = call->strides[index].data();
    } else {
      call->strides[index].clear();
      call->handles[index].strides = nullptr;
    }
  }

  static void EnsureReplayInitialized(const GraphRuntimeState& state, int32_t graph_id) {
    if (!state.replay.initialized) {
      throw std::runtime_error("Typhoon replay state for graph " + std::to_string(graph_id) +
                               " was not initialized");
    }
  }

  static void AddDMATask(GraphRuntimeState* state, int32_t direction, void* global_handle,
                         int64_t global_byte_offset, int32_t sram_region_id, int64_t bytes,
                         const std::vector<int32_t>& deps, int64_t sram_byte_offset = 0) {
    state->graph.AddDMATask(NextTaskId(state), direction, global_handle, global_byte_offset,
                            sram_region_id, sram_byte_offset, bytes, deps.size(),
                            deps.empty() ? nullptr : deps.data());
  }

  static int32_t AddBatchedDMATaskReturningId(GraphRuntimeState* state, int32_t direction,
                                              void* global_handle, int64_t global_byte_offset,
                                              int64_t global_stride, int32_t sram_region_id,
                                              int64_t sram_byte_offset, int64_t sram_stride,
                                              int64_t bytes, int64_t batch_count,
                                              const std::vector<int32_t>& deps) {
    int32_t task_id = NextTaskId(state);
    state->graph.AddBatchedDMATask(task_id, direction, global_handle, global_byte_offset,
                                   global_stride, sram_region_id, sram_byte_offset, sram_stride,
                                   bytes, batch_count, deps.size(),
                                   deps.empty() ? nullptr : deps.data());
    return task_id;
  }

  static int32_t AddDMATaskReturningId(GraphRuntimeState* state, int32_t direction,
                                       void* global_handle, int64_t global_byte_offset,
                                       int32_t sram_region_id, int64_t bytes,
                                       const std::vector<int32_t>& deps,
                                       int64_t sram_byte_offset = 0) {
    int32_t task_id = NextTaskId(state);
    state->graph.AddDMATask(task_id, direction, global_handle, global_byte_offset, sram_region_id,
                            sram_byte_offset, bytes, deps.size(),
                            deps.empty() ? nullptr : deps.data());
    return task_id;
  }

  static GraphTaskStats ComputeGraphTaskStats(const TyphoonGraphBuilder& graph) {
    GraphTaskStats stats;
    stats.total = static_cast<int64_t>(graph.tasks().size());
    for (const TyphoonTask& task : graph.tasks()) {
      switch (task.kind) {
        case TaskKind::kDMA:
          ++stats.dma;
          break;
        case TaskKind::kMatmul:
          ++stats.matmul;
          break;
        case TaskKind::kVector:
          ++stats.vector;
          break;
        case TaskKind::kReshape:
          ++stats.reshape;
          break;
      }
    }
    return stats;
  }

  static std::string SerializeGraphStatsJSON(const GraphTaskStats& stats) {
    std::ostringstream os;
    os << "{"
       << "\"total\":" << stats.total << ","
       << "\"dma\":" << stats.dma << ","
       << "\"matmul\":" << stats.matmul << ","
       << "\"vector\":" << stats.vector << ","
       << "\"reshape\":" << stats.reshape << "}";
    return os.str();
  }

  static int32_t AddReshapeTask(GraphRuntimeState* state, int32_t in_region_id,
                                int32_t out_region_id, int64_t elem_count, int32_t transform_code,
                                const std::vector<int64_t>& metadata,
                                const std::vector<int32_t>& deps) {
    int32_t task_id = NextTaskId(state);
    state->graph.AddReshapeTask(task_id, in_region_id, out_region_id, elem_count, transform_code,
                                metadata.size(), metadata.empty() ? nullptr : metadata.data(),
                                deps.size(), deps.empty() ? nullptr : deps.data());
    return task_id;
  }

  static int32_t AddVectorTask(GraphRuntimeState* state, int32_t op_code, int32_t in0_region_id,
                               int32_t in1_region_id, int32_t out_region_id, int64_t elem_count,
                               const std::vector<int64_t>& metadata,
                               const std::vector<int32_t>& deps) {
    int32_t task_id = NextTaskId(state);
    state->graph.AddVectorTask(task_id, op_code, in0_region_id, in1_region_id, out_region_id,
                               elem_count, 2, metadata.size(),
                               metadata.empty() ? nullptr : metadata.data(), deps.size(),
                               deps.empty() ? nullptr : deps.data());
    return task_id;
  }

  static int32_t AddMatmulTask(GraphRuntimeState* state, int32_t a_region_id, int32_t b_region_id,
                               int32_t c_region_id, int64_t m, int64_t n, int64_t k,
                               int32_t layout_code, const std::vector<int32_t>& deps) {
    int32_t task_id = NextTaskId(state);
    state->graph.AddMatmulTask(task_id, a_region_id, b_region_id, c_region_id, m, n, k, 2,
                               layout_code, deps.size(), deps.empty() ? nullptr : deps.data());
    return task_id;
  }

  static void ReplayReluLayer(GraphRuntimeState* state, int32_t layer_id) {
    const CanonicalRegionSpec& act0 = FindCanonicalRegion("act0");
    const CanonicalRegionSpec& act1 = FindCanonicalRegion("act1");
    const DLTensor* input = GetCapturedTensor(*state, layer_id, 0);
    const DLTensor* output = GetCapturedTensor(*state, layer_id, 1);
    int64_t total_bytes = TensorBytes(input);
    std::vector<int32_t> carry_deps = state->replay.prev_segment_deps;
    for (int64_t offset = 0; offset < total_bytes; offset += act0.size) {
      int64_t chunk_bytes = std::min<int64_t>(act0.size, total_bytes - offset);
      int32_t dma_in =
          AddDMATaskReturningId(state, 0, input->data, offset, act0.region_id, chunk_bytes, carry_deps);
      int32_t vector =
          AddVectorTask(state, 1, act0.region_id, -1, act1.region_id, chunk_bytes / 4, {}, {dma_in});
      int32_t dma_out = AddDMATaskReturningId(state, 1, output->data, offset, act1.region_id,
                                             chunk_bytes, {vector});
      carry_deps = {dma_out};
    }
    state->replay.prev_segment_deps = std::move(carry_deps);
  }

  static void ReplayCopyLayer(GraphRuntimeState* state, int32_t layer_id) {
    const CanonicalRegionSpec& act0 = FindCanonicalRegion("act0");
    const CanonicalRegionSpec& act1 = FindCanonicalRegion("act1");
    const DLTensor* input = GetCapturedTensor(*state, layer_id, 0);
    const DLTensor* output = GetCapturedTensor(*state, layer_id, 1);
    int64_t total_bytes = TensorBytes(input);
    std::vector<int32_t> carry_deps = state->replay.prev_segment_deps;
    for (int64_t offset = 0; offset < total_bytes; offset += act0.size) {
      int64_t chunk_bytes = std::min<int64_t>(act0.size, total_bytes - offset);
      int32_t dma_in =
          AddDMATaskReturningId(state, 0, input->data, offset, act0.region_id, chunk_bytes, carry_deps);
      int32_t reshape =
          AddReshapeTask(state, act0.region_id, act1.region_id, chunk_bytes, 0, {}, {dma_in});
      int32_t dma_out = AddDMATaskReturningId(state, 1, output->data, offset, act1.region_id,
                                             chunk_bytes, {reshape});
      carry_deps = {dma_out};
    }
    state->replay.prev_segment_deps = std::move(carry_deps);
  }

  static void ReplayAddLayer(GraphRuntimeState* state, int32_t layer_id) {
    const CanonicalRegionSpec& act0 = FindCanonicalRegion("act0");
    const CanonicalRegionSpec& act1 = FindCanonicalRegion("act1");
    const CanonicalRegionSpec& residual = FindCanonicalRegion("residual");
    const DLTensor* lhs = GetCapturedTensor(*state, layer_id, 0);
    const DLTensor* rhs = GetCapturedTensor(*state, layer_id, 1);
    const DLTensor* output = GetCapturedTensor(*state, layer_id, 2);
    int64_t lhs_bytes = TensorBytes(lhs);
    int64_t rhs_bytes = TensorBytes(rhs);
    std::vector<int64_t> lhs_shape = ShapeOf(lhs);
    std::vector<int64_t> rhs_shape = ShapeOf(rhs);
    std::vector<int64_t> out_shape = ShapeOf(output);
    auto format_shape = [](const DLTensor* tensor) {
      std::string text = "[";
      for (int i = 0; i < tensor->ndim; ++i) {
        if (i != 0) {
          text += ", ";
        }
        text += std::to_string(tensor->shape[i]);
      }
      text += "]";
      return text;
    };
    if (lhs_bytes <= 0 || rhs_bytes <= 0) {
      throw std::runtime_error("Typhoon replay add layer " + std::to_string(layer_id) +
                               " has non-positive tensor bytes: lhs_bytes=" +
                               std::to_string(lhs_bytes) + " rhs_bytes=" +
                               std::to_string(rhs_bytes) + " lhs_shape=" + format_shape(lhs) +
                               " rhs_shape=" + format_shape(rhs) + " out_shape=" +
                               format_shape(output) + " lhs_dtype_bits=" +
                               std::to_string(lhs->dtype.bits) + " rhs_dtype_bits=" +
                               std::to_string(rhs->dtype.bits) + " out_dtype_bits=" +
                               std::to_string(output->dtype.bits));
    }
    bool broadcast = rhs_bytes < lhs_bytes;
    std::vector<int32_t> carry_deps = state->replay.prev_segment_deps;
    bool channel_bias_broadcast =
        lhs_shape.size() == 4 && rhs_shape.size() == 4 && out_shape == lhs_shape &&
        rhs_shape[0] == 1 && rhs_shape[1] == lhs_shape[1] && rhs_shape[2] == 1 &&
        rhs_shape[3] == 1;
    if (!broadcast) {
      int64_t chunk_capacity =
          std::min<int64_t>(act0.size, std::min<int64_t>(residual.size, act1.size));
      for (int64_t offset = 0; offset < lhs_bytes; offset += chunk_capacity) {
        int64_t chunk_bytes = std::min<int64_t>(chunk_capacity, lhs_bytes - offset);
        int32_t dma_lhs = AddDMATaskReturningId(state, 0, lhs->data, offset, act0.region_id,
                                                chunk_bytes, carry_deps);
        int32_t dma_rhs = AddDMATaskReturningId(state, 0, rhs->data, offset, residual.region_id,
                                                chunk_bytes, carry_deps);
        int32_t vector = AddVectorTask(state, 0, act0.region_id, residual.region_id, act1.region_id,
                                       chunk_bytes / 4, {}, {dma_lhs, dma_rhs});
        int32_t dma_out = AddDMATaskReturningId(state, 1, output->data, offset, act1.region_id,
                                               chunk_bytes, {vector});
        carry_deps = {dma_out};
      }
    } else if (channel_bias_broadcast) {
      int64_t batch = lhs_shape[0];
      int64_t channels = lhs_shape[1];
      int64_t height = lhs_shape[2];
      int64_t width = lhs_shape[3];
      int64_t plane_elems = height * width;
      int64_t plane_bytes = plane_elems * 4;
      int64_t channel_chunk = std::max<int64_t>(1, act0.size / plane_bytes);
      channel_chunk = std::min<int64_t>(channel_chunk, channels);
      for (int64_t n = 0; n < batch; ++n) {
        int64_t batch_base_offset = n * channels * plane_bytes;
        for (int64_t c_offset = 0; c_offset < channels; c_offset += channel_chunk) {
          int64_t c = std::min<int64_t>(channel_chunk, channels - c_offset);
          int64_t plane_offset = batch_base_offset + c_offset * plane_bytes;
          int64_t bias_offset = c_offset * 4;
          int32_t dma_lhs = AddBatchedDMATaskReturningId(
              state, 0, lhs->data, plane_offset, plane_bytes, act0.region_id, 0, plane_bytes,
              plane_bytes, c, carry_deps);
          int32_t dma_rhs = AddBatchedDMATaskReturningId(
              state, 0, rhs->data, bias_offset, 4, residual.region_id, 0, 4, 4, c, carry_deps);
          int32_t vector = AddVectorTask(state, 4, act0.region_id, residual.region_id,
                                         act1.region_id, c * plane_elems, {c, plane_elems},
                                         {dma_lhs, dma_rhs});
          int32_t dma_out = AddBatchedDMATaskReturningId(
              state, 1, output->data, plane_offset, plane_bytes, act1.region_id, 0, plane_bytes,
              plane_bytes, c, {vector});
          carry_deps = {dma_out};
        }
      }
    } else {
      int64_t inner = rhs_bytes / 4;
      int64_t total_elems = lhs_bytes / 4;
      int64_t outer_total = total_elems / inner;
      int64_t outer_chunk = std::max<int64_t>(1, act0.size / rhs_bytes);
      for (int64_t outer_offset = 0; outer_offset < outer_total; outer_offset += outer_chunk) {
        int64_t outer = std::min<int64_t>(outer_chunk, outer_total - outer_offset);
        int64_t chunk_bytes = outer * rhs_bytes;
        int64_t global_offset = outer_offset * rhs_bytes;
        int32_t dma_lhs = AddDMATaskReturningId(state, 0, lhs->data, global_offset, act0.region_id,
                                                chunk_bytes, carry_deps);
        int32_t dma_rhs = AddDMATaskReturningId(state, 0, rhs->data, 0, residual.region_id,
                                                rhs_bytes, carry_deps);
        int32_t vector =
            AddVectorTask(state, 0, act0.region_id, residual.region_id, act1.region_id,
                          chunk_bytes / 4, {outer, inner}, {dma_lhs, dma_rhs});
        int32_t dma_out = AddDMATaskReturningId(state, 1, output->data, global_offset, act1.region_id,
                                               chunk_bytes, {vector});
        carry_deps = {dma_out};
      }
    }
    state->replay.prev_segment_deps = std::move(carry_deps);
  }

  static void ReplayMaxPoolLayer(GraphRuntimeState* state, int32_t layer_id) {
    const CanonicalRegionSpec& act0 = FindCanonicalRegion("act0");
    const CanonicalRegionSpec& act1 = FindCanonicalRegion("act1");
    const DLTensor* input = GetCapturedTensor(*state, layer_id, 0);
    const DLTensor* output = GetCapturedTensor(*state, layer_id, 1);
    std::vector<int64_t> in_shape = ShapeOf(input);
    std::vector<int64_t> out_shape = ShapeOf(output);
    int64_t in_h = in_shape[2];
    int64_t in_w = in_shape[3];
    int64_t out_h = out_shape[2];
    int64_t out_w = out_shape[3];
    int64_t c_total = in_shape[1];
    int64_t c_chunk = std::max<int64_t>(1, act0.size / (in_h * in_w * 4));
    c_chunk = std::min<int64_t>(c_chunk, c_total);
    std::vector<int32_t> carry_deps = state->replay.prev_segment_deps;
    for (int64_t c_offset = 0; c_offset < c_total; c_offset += c_chunk) {
      int64_t c = std::min<int64_t>(c_chunk, c_total - c_offset);
      int64_t input_offset = c_offset * in_h * in_w * 4;
      int64_t output_offset = c_offset * out_h * out_w * 4;
      int64_t input_bytes = c * in_h * in_w * 4;
      int64_t output_bytes = c * out_h * out_w * 4;
      int32_t dma_in =
          AddDMATaskReturningId(state, 0, input->data, input_offset, act0.region_id, input_bytes, carry_deps);
      int32_t vector = AddVectorTask(state, 2, act0.region_id, -1, act1.region_id, c * out_h * out_w,
                                     {1, c, in_h, in_w, 3, 3, 2, 2, 1, 1, out_h, out_w}, {dma_in});
      int32_t dma_out = AddDMATaskReturningId(state, 1, output->data, output_offset, act1.region_id,
                                             output_bytes, {vector});
      carry_deps = {dma_out};
    }
    state->replay.prev_segment_deps = std::move(carry_deps);
  }

  static void ReplayGlobalAveragePoolLayer(GraphRuntimeState* state, int32_t layer_id) {
    const CanonicalRegionSpec& act0 = FindCanonicalRegion("act0");
    const CanonicalRegionSpec& act1 = FindCanonicalRegion("act1");
    const DLTensor* input = GetCapturedTensor(*state, layer_id, 0);
    const DLTensor* output = GetCapturedTensor(*state, layer_id, 1);
    std::vector<int64_t> in_shape = ShapeOf(input);
    int64_t c_total = in_shape[1];
    int64_t in_h = in_shape[2];
    int64_t in_w = in_shape[3];
    int64_t c_chunk = std::max<int64_t>(1, act0.size / (in_h * in_w * 4));
    c_chunk = std::min<int64_t>(c_chunk, c_total);
    std::vector<int32_t> carry_deps = state->replay.prev_segment_deps;
    for (int64_t c_offset = 0; c_offset < c_total; c_offset += c_chunk) {
      int64_t c = std::min<int64_t>(c_chunk, c_total - c_offset);
      int64_t input_offset = c_offset * in_h * in_w * 4;
      int64_t output_offset = c_offset * 4;
      int64_t input_bytes = c * in_h * in_w * 4;
      int64_t output_bytes = c * 4;
      int32_t dma_in =
          AddDMATaskReturningId(state, 0, input->data, input_offset, act0.region_id, input_bytes, carry_deps);
      int32_t vector =
          AddVectorTask(state, 3, act0.region_id, -1, act1.region_id, c, {1, c, in_h, in_w}, {dma_in});
      int32_t dma_out = AddDMATaskReturningId(state, 1, output->data, output_offset, act1.region_id,
                                             output_bytes, {vector});
      carry_deps = {dma_out};
    }
    state->replay.prev_segment_deps = std::move(carry_deps);
  }

  static void ReplayDenseMatmulLayer(GraphRuntimeState* state, int32_t layer_id) {
    const CanonicalRegionSpec& act0 = FindCanonicalRegion("act0");
    const CanonicalRegionSpec& act1 = FindCanonicalRegion("act1");
    const CanonicalRegionSpec& residual = FindCanonicalRegion("residual");
    const CanonicalRegionSpec& wgt0 = FindCanonicalRegion("wgt0");
    const DLTensor* input = GetCapturedTensor(*state, layer_id, 0);
    const DLTensor* weight = GetCapturedTensor(*state, layer_id, 1);
    const DLTensor* output = GetCapturedTensor(*state, layer_id, 2);
    std::vector<int64_t> input_shape = ShapeOf(input);
    std::vector<int64_t> output_shape = ShapeOf(output);
    if (input_shape.size() != 2 || output_shape.size() != 2) {
      auto format_shape = [](const std::vector<int64_t>& shape) {
        std::ostringstream os;
        os << "[";
        for (size_t i = 0; i < shape.size(); ++i) {
          if (i != 0) {
            os << ", ";
          }
          os << shape[i];
        }
        os << "]";
        return os.str();
      };
      throw std::runtime_error("Typhoon replay dense_matmul layer " + std::to_string(layer_id) +
                               " requires rank-2 input/output tensors but saw input_shape=" +
                               format_shape(input_shape) + " output_shape=" +
                               format_shape(output_shape));
    }
    int64_t k = input_shape[1];
    int64_t n_total = output_shape[1];
    int64_t n_chunk = std::max<int64_t>(1, wgt0.size / (k * 4));
    n_chunk = std::min<int64_t>(n_chunk, n_total);
    const DLTensor* fused_bias = nullptr;
    const DLTensor* fused_output = output;
    int32_t fused_layer_id = -1;
    if (layer_id + 1 < static_cast<int32_t>(state->captured_calls.size()) &&
        state->replay.skipped_layer_ids.count(layer_id + 1) == 0) {
      const DLTensor* next_lhs = GetCapturedTensor(*state, layer_id + 1, 0);
      const DLTensor* next_rhs = GetCapturedTensor(*state, layer_id + 1, 1);
      const DLTensor* next_out = GetCapturedTensor(*state, layer_id + 1, 2);
      if (next_lhs->data == output->data && SameShape(next_lhs, output) &&
          SameShape(next_out, output) && IsContiguousBiasVector(next_rhs, n_total)) {
        fused_bias = next_rhs;
        fused_output = next_out;
        fused_layer_id = layer_id + 1;
      }
    }
    std::vector<int32_t> carry_deps = state->replay.prev_segment_deps;
    for (int64_t n_offset = 0; n_offset < n_total; n_offset += n_chunk) {
      int64_t n = std::min<int64_t>(n_chunk, n_total - n_offset);
      int64_t weight_offset = n_offset * k * 4;
      int64_t weight_bytes = n * k * 4;
      int64_t output_bytes = n * 4;
      int32_t dma_input =
          AddDMATaskReturningId(state, 0, input->data, 0, act0.region_id, k * 4, carry_deps);
      int32_t dma_weight = AddDMATaskReturningId(state, 0, weight->data, weight_offset,
                                                 wgt0.region_id, weight_bytes, carry_deps);
      int32_t matmul = AddMatmulTask(state, act0.region_id, wgt0.region_id, act1.region_id, 1, n, k, 1,
                                     {dma_input, dma_weight});
      int32_t dma_out = -1;
      if (fused_bias != nullptr) {
        int32_t dma_bias = AddDMATaskReturningId(state, 0, fused_bias->data, n_offset * 4,
                                                 residual.region_id, output_bytes, carry_deps);
        int32_t vector = AddVectorTask(state, 0, act1.region_id, residual.region_id, act0.region_id,
                                       n, {}, {matmul, dma_bias});
        dma_out = AddDMATaskReturningId(state, 1, fused_output->data, n_offset * 4, act0.region_id,
                                        output_bytes, {vector});
      } else {
        dma_out = AddDMATaskReturningId(state, 1, output->data, n_offset * 4, act1.region_id,
                                        output_bytes, {matmul});
      }
      carry_deps = {dma_out};
    }
    if (fused_layer_id >= 0) {
      state->replay.skipped_layer_ids.insert(fused_layer_id);
    }
    state->replay.prev_segment_deps = std::move(carry_deps);
  }

  static void ReplayConv2DLayer(GraphRuntimeState* state, int32_t layer_id) {
    const CanonicalRegionSpec& act0 = FindCanonicalRegion("act0");
    const CanonicalRegionSpec& act1 = FindCanonicalRegion("act1");
    const CanonicalRegionSpec& wgt0 = FindCanonicalRegion("wgt0");
    const CanonicalRegionSpec& col0 = FindCanonicalRegion("col0");
    const CanonicalRegionSpec& aux0 = FindCanonicalRegion("aux0");
    const DLTensor* input = GetCapturedTensor(*state, layer_id, 0);
    const DLTensor* weight = GetCapturedTensor(*state, layer_id, 1);
    const DLTensor* output = GetCapturedTensor(*state, layer_id, 2);
    std::vector<int64_t> input_shape = ShapeOf(input);
    std::vector<int64_t> weight_shape = ShapeOf(weight);
    std::vector<int64_t> output_shape = ShapeOf(output);
    int64_t in_channels = input_shape[1];
    int64_t in_h = input_shape[2];
    int64_t in_w = input_shape[3];
    int64_t out_channels = weight_shape[0];
    int64_t kernel_h = weight_shape[2];
    int64_t kernel_w = weight_shape[3];
    int64_t out_h = output_shape[2];
    int64_t out_w = output_shape[3];
    auto [stride_h, pad_h] = InferConvStrideAndPad(in_h, out_h, kernel_h);
    auto [stride_w, pad_w] = InferConvStrideAndPad(in_w, out_w, kernel_w);
    int64_t patch_size = in_channels * kernel_h * kernel_w;
    int64_t out_h_tile = 0;
    for (int64_t candidate = 1; candidate <= out_h; ++candidate) {
      int64_t worst_input_h = (candidate - 1) * stride_h + kernel_h;
      int64_t input_tile_bytes = in_channels * worst_input_h * in_w * 4;
      int64_t col_tile_bytes = candidate * out_w * patch_size * 4;
      if (input_tile_bytes <= act0.size && col_tile_bytes <= col0.size) {
        out_h_tile = candidate;
      }
    }
    if (out_h_tile == 0) {
      throw std::runtime_error("Typhoon replay could not fit canonical conv tile in SRAM");
    }
    int64_t max_weight_channels = wgt0.size / (patch_size * 4);
    int64_t max_output_channels = 0;
    std::vector<int32_t> carry_deps = state->replay.prev_segment_deps;
    for (int64_t out_h_start = 0; out_h_start < out_h; out_h_start += out_h_tile) {
      int64_t current_out_h_tile = std::min<int64_t>(out_h_tile, out_h - out_h_start);
      int64_t raw_input_h_start = out_h_start * stride_h - pad_h;
      int64_t raw_input_h_end = (out_h_start + current_out_h_tile - 1) * stride_h - pad_h + kernel_h;
      int64_t input_h_start = std::max<int64_t>(0, raw_input_h_start);
      int64_t input_h_end = std::min<int64_t>(in_h, raw_input_h_end);
      int64_t input_h_tile = input_h_end - input_h_start;
      int64_t input_plane_bytes = NCHWChannelPlaneBytes(input_h_tile, in_w);
      int32_t input_dma = AddBatchedDMATaskReturningId(
          state, 0, input->data, NCHWChannelOffsetBytes(0, input_h_start, in_h, in_w),
          NCHWChannelPlaneBytes(in_h, in_w), act0.region_id, 0, input_plane_bytes,
          input_plane_bytes, in_channels, carry_deps);
      int64_t rows = current_out_h_tile * out_w;
      max_output_channels = aux0.size / (rows * 4);
      int64_t out_channel_chunk =
          std::min<int64_t>(out_channels, std::min(max_weight_channels, max_output_channels));
      if (out_channel_chunk <= 0) {
        throw std::runtime_error("Typhoon replay could not fit conv output tile in SRAM");
      }
      int32_t reshape = AddReshapeTask(
          state, act0.region_id, col0.region_id, rows * patch_size * 4, 1,
          {1, in_channels, input_h_tile, in_w, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w,
           current_out_h_tile, out_w, input_h_start, 0, out_h_start, 0},
          {input_dma});
      std::vector<int32_t> chunk_deps = {reshape};
      for (int64_t oc_start = 0; oc_start < out_channels; oc_start += out_channel_chunk) {
        int64_t current_oc_chunk = std::min<int64_t>(out_channel_chunk, out_channels - oc_start);
        int32_t dma_weight = AddDMATaskReturningId(state, 0, weight->data, oc_start * patch_size * 4,
                                                   wgt0.region_id, current_oc_chunk * patch_size * 4,
                                                   chunk_deps);
        int32_t matmul =
            AddMatmulTask(state, col0.region_id, wgt0.region_id, aux0.region_id, rows,
                          current_oc_chunk, patch_size, 1, {reshape, dma_weight});
        int32_t transpose =
            AddReshapeTask(state, aux0.region_id, act1.region_id, rows * current_oc_chunk * 4, 2,
                           {rows, current_oc_chunk}, {matmul});
        int64_t output_plane_bytes = NCHWChannelPlaneBytes(current_out_h_tile, out_w);
        int32_t output_dma = AddBatchedDMATaskReturningId(
            state, 1, output->data, NCHWChannelOffsetBytes(oc_start, out_h_start, out_h, out_w),
            NCHWChannelPlaneBytes(out_h, out_w), act1.region_id, 0, output_plane_bytes,
            output_plane_bytes, current_oc_chunk, {transpose});
        chunk_deps = {output_dma};
      }
      carry_deps = std::move(chunk_deps);
    }
    state->replay.prev_segment_deps = std::move(carry_deps);
  }

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

  template <typename F>
  void* CallHandle(F&& f) {
    std::lock_guard<std::mutex> lock(mu_);
    try {
      void* handle = std::forward<F>(f)();
      last_error_.clear();
      return handle;
    } catch (const std::exception& e) {
      last_error_ = e.what();
      return nullptr;
    }
  }

  GraphRuntimeState& GetOrCreateState(int32_t graph_id) {
    auto it = graphs_.find(graph_id);
    if (it == graphs_.end()) {
      it = graphs_.emplace(graph_id, GraphRuntimeState(graph_id)).first;
    }
    return it->second;
  }

  int32_t NextCaptureLayerId(int32_t graph_id) {
    return static_cast<int32_t>(GetOrCreateState(graph_id).captured_calls.size());
  }

  void StoreCapturedCall(int32_t graph_id, int32_t layer_id, CapturedCall call) {
    auto& captured_calls = GetOrCreateState(graph_id).captured_calls;
    if (captured_calls.size() <= static_cast<size_t>(layer_id)) {
      captured_calls.resize(static_cast<size_t>(layer_id) + 1);
    }
    captured_calls[static_cast<size_t>(layer_id)] = std::move(call);
  }

  int32_t ResolvePlannedCaptureLayerId(int32_t graph_id, const char* layer_ids_csv) {
    if (layer_ids_csv == nullptr || layer_ids_csv[0] == '\0') {
      throw std::runtime_error("Typhoon planned capture requires non-empty layer_ids_csv");
    }

    std::vector<int32_t> layer_ids;
    std::stringstream ss(layer_ids_csv);
    std::string token;
    while (std::getline(ss, token, ',')) {
      if (!token.empty()) {
        layer_ids.push_back(static_cast<int32_t>(std::stoi(token)));
      }
    }
    if (layer_ids.empty()) {
      throw std::runtime_error("Typhoon planned capture could not parse layer_ids `" +
                               std::string(layer_ids_csv) + "`");
    }

    auto& state = GetOrCreateState(graph_id);
    std::string key(layer_ids_csv);
    auto& next_index = state.planned_capture_positions[key];
    if (next_index >= static_cast<int32_t>(layer_ids.size())) {
      throw std::runtime_error("Typhoon planned capture exhausted layer_ids `" + key + "`");
    }
    return layer_ids[static_cast<size_t>(next_index++)];
  }

  TyphoonGraphBuilder& GetOrCreateGraph(int32_t graph_id) { return GetOrCreateState(graph_id).graph; }

  mutable std::mutex mu_;
  std::unordered_map<int32_t, GraphRuntimeState> graphs_;
  bool keep_graph_state_for_testing_{false};
  std::string last_error_;
  std::string last_trace_json_{"[]"};
  std::string last_graph_stats_json_{"{}"};
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

extern "C" int TVMTyphoonAddBatchedDMATask(int32_t graph_id, int32_t task_id, int32_t direction,
                                           void* global_handle, int64_t global_byte_offset,
                                           int64_t global_stride, int32_t sram_region_id,
                                           int64_t sram_byte_offset, int64_t sram_stride,
                                           int64_t bytes, int64_t batch_count, int32_t num_deps,
                                           const int32_t* dep_ids) {
  return tvm::runtime::typhoon::TyphoonRuntimeState::Global().AddBatchedDMATask(
      graph_id, task_id, direction, global_handle, global_byte_offset, global_stride,
      sram_region_id, sram_byte_offset, sram_stride, bytes, batch_count, num_deps, dep_ids);
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

extern "C" int TVMTyphoonCaptureCall(int32_t graph_id, int32_t num_handles, void* handle0,
                                     void* handle1, void* handle2) {
  return tvm::runtime::typhoon::TyphoonRuntimeState::Global().CaptureCall(
      graph_id, num_handles, handle0, handle1, handle2);
}

extern "C" int TVMTyphoonCaptureCallAt(int32_t graph_id, int32_t layer_id, int32_t num_handles,
                                       void* handle0, void* handle1, void* handle2) {
  return tvm::runtime::typhoon::TyphoonRuntimeState::Global().CaptureCallAt(
      graph_id, layer_id, num_handles, handle0, handle1, handle2);
}

extern "C" int TVMTyphoonCaptureCallPlanned(int32_t graph_id, const char* layer_ids_csv,
                                            int32_t num_handles, void* handle0, void* handle1,
                                            void* handle2) {
  return tvm::runtime::typhoon::TyphoonRuntimeState::Global().CaptureCallPlanned(
      graph_id, layer_ids_csv, num_handles, handle0, handle1, handle2);
}

extern "C" int TVMTyphoonCapturePackedArgs(int32_t graph_id, void* packed_args, int32_t num_args) {
  return tvm::runtime::typhoon::TyphoonRuntimeState::Global().CapturePackedArgs(
      graph_id, static_cast<const TVMFFIAny*>(packed_args), num_args);
}

extern "C" int TVMTyphoonCapturePackedArgsAt(int32_t graph_id, int32_t layer_id,
                                             void* packed_args, int32_t num_args) {
  return tvm::runtime::typhoon::TyphoonRuntimeState::Global().CapturePackedArgsAt(
      graph_id, layer_id, static_cast<const TVMFFIAny*>(packed_args), num_args);
}

extern "C" int TVMTyphoonCapturePackedArgsPlanned(int32_t graph_id, const char* layer_ids_csv,
                                                  void* packed_args, int32_t num_args) {
  return tvm::runtime::typhoon::TyphoonRuntimeState::Global().CapturePackedArgsPlanned(
      graph_id, layer_ids_csv, static_cast<const TVMFFIAny*>(packed_args), num_args);
}

extern "C" void* TVMTyphoonGetCapturedHandle(int32_t graph_id, int32_t layer_id,
                                             int32_t handle_index) {
  return tvm::runtime::typhoon::TyphoonRuntimeState::Global().GetCapturedHandle(
      graph_id, layer_id, handle_index);
}

extern "C" void* TVMTyphoonGetCapturedTensorPtr(int32_t graph_id, int32_t layer_id,
                                                int32_t handle_index) {
  return tvm::runtime::typhoon::TyphoonRuntimeState::Global().GetCapturedTensorPtr(
      graph_id, layer_id, handle_index);
}

extern "C" int TVMTyphoonReplayWholeGraphBegin(int32_t graph_id) {
  return tvm::runtime::typhoon::TyphoonRuntimeState::Global().ReplayWholeGraphBegin(graph_id);
}

extern "C" int TVMTyphoonReplayCapturedLayer(int32_t graph_id, int32_t layer_id,
                                             int32_t replay_kind) {
  return tvm::runtime::typhoon::TyphoonRuntimeState::Global().ReplayCapturedLayer(
      graph_id, layer_id, replay_kind);
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
  refl::GlobalDef().def("runtime.typhoon.testing_keep_graph_state",
                        [](bool keep) { TyphoonRuntimeState::Global().SetKeepGraphStateForTesting(keep); });
  refl::GlobalDef().def("runtime.typhoon.testing_last_error",
                        []() { return TyphoonRuntimeState::Global().LastError(); });
  refl::GlobalDef().def("runtime.typhoon_get_last_trace_json",
                        []() { return TyphoonRuntimeState::Global().LastTraceJSON(); });
  refl::GlobalDef().def("runtime.typhoon_get_last_graph_stats_json",
                        []() { return TyphoonRuntimeState::Global().LastGraphStatsJSON(); });
}

}  // namespace typhoon
}  // namespace runtime
}  // namespace tvm
