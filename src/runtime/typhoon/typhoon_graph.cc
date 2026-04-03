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

#include <algorithm>
#include <stdexcept>
#include <string>
#include <utility>

namespace tvm {
namespace runtime {
namespace typhoon {

namespace {

[[noreturn]] void TyphoonError(const std::string& message) { throw std::runtime_error(message); }

}  // namespace

TyphoonGraphBuilder::TyphoonGraphBuilder(int32_t graph_id) : graph_id_(graph_id) {}

void TyphoonGraphBuilder::GraphBegin() { began_ = true; }

void TyphoonGraphBuilder::DeclareRegion(int32_t region_id, int64_t offset, int64_t size,
                                        int64_t alignment, bool preinitialized, const char* tag) {
  if (submitted_) {
    TyphoonError("Typhoon graph " + std::to_string(graph_id_) +
                 " cannot be mutated after submit");
  }
  if (region_index_.count(region_id)) {
    TyphoonError("Typhoon runtime duplicate region_id " + std::to_string(region_id));
  }

  TyphoonRegion region;
  region.region_id = region_id;
  region.offset = offset;
  region.size = size;
  region.alignment = alignment;
  region.preinitialized = preinitialized;
  region.tag = tag == nullptr ? "" : tag;

  CheckRegionBounds(region);
  CheckRegionOverlap(region);

  region_index_[region_id] = regions_.size();
  regions_.push_back(std::move(region));
}

void TyphoonGraphBuilder::AddDMATask(int32_t task_id, int32_t direction, void* global_handle,
                                     int64_t global_byte_offset, int32_t sram_region_id,
                                     int64_t bytes, int32_t num_deps, const int32_t* dep_ids) {
  TyphoonTask task;
  task.kind = TaskKind::kDMA;
  task.task_id = task_id;
  task.deps = CopyDeps(num_deps, dep_ids);
  task.global_endpoint = {global_handle, global_byte_offset};
  task.direction = direction;
  task.bytes = bytes;
  if (direction == 0) {
    task.writes = {sram_region_id};
  } else if (direction == 1) {
    task.reads = {sram_region_id};
  } else {
    TyphoonError("Typhoon DMA direction must be 0 or 1");
  }
  AddTask(std::move(task));
}

void TyphoonGraphBuilder::AddMatmulTask(int32_t task_id, int32_t a_region_id, int32_t b_region_id,
                                        int32_t c_region_id, int64_t m, int64_t n, int64_t k,
                                        int32_t dtype_code, int32_t layout_code, int32_t num_deps,
                                        const int32_t* dep_ids) {
  TyphoonTask task;
  task.kind = TaskKind::kMatmul;
  task.task_id = task_id;
  task.deps = CopyDeps(num_deps, dep_ids);
  task.reads = {a_region_id, b_region_id};
  task.writes = {c_region_id};
  task.m = m;
  task.n = n;
  task.k = k;
  task.dtype_code = dtype_code;
  (void)layout_code;
  AddTask(std::move(task));
}

void TyphoonGraphBuilder::AddVectorTask(int32_t task_id, int32_t op_code, int32_t in0_region_id,
                                        int32_t in1_region_id, int32_t out_region_id,
                                        int64_t elem_count, int32_t dtype_code, int32_t num_deps,
                                        const int32_t* dep_ids) {
  TyphoonTask task;
  task.kind = TaskKind::kVector;
  task.task_id = task_id;
  task.deps = CopyDeps(num_deps, dep_ids);
  task.reads = {in0_region_id, in1_region_id};
  task.writes = {out_region_id};
  task.op_code = op_code;
  task.elem_count = elem_count;
  task.dtype_code = dtype_code;
  AddTask(std::move(task));
}

void TyphoonGraphBuilder::AddReshapeTask(int32_t task_id, int32_t in_region_id,
                                         int32_t out_region_id, int64_t elem_count,
                                         int32_t transform_code, int32_t num_deps,
                                         const int32_t* dep_ids) {
  TyphoonTask task;
  task.kind = TaskKind::kReshape;
  task.task_id = task_id;
  task.deps = CopyDeps(num_deps, dep_ids);
  task.reads = {in_region_id};
  task.writes = {out_region_id};
  task.elem_count = elem_count;
  (void)transform_code;
  AddTask(std::move(task));
}

void TyphoonGraphBuilder::Submit() {
  ValidateTaskRegions();
  ValidateTaskDependencies();
  ValidateTaskInitialization();
  ValidateWriteHazards();
  ValidateTaskFootprints();
  submitted_ = true;
}

void TyphoonGraphBuilder::Wait() const {
  if (!submitted_) {
    TyphoonError("Typhoon graph " + std::to_string(graph_id_) + " has not been submitted");
  }
}

std::vector<int32_t> TyphoonGraphBuilder::CopyDeps(int32_t num_deps, const int32_t* dep_ids) const {
  if (num_deps < 0) {
    TyphoonError("Typhoon dependency count must be non-negative");
  }
  if (num_deps > 0 && dep_ids == nullptr) {
    TyphoonError("Typhoon dependency array cannot be null when num_deps > 0");
  }
  std::vector<int32_t> deps;
  deps.reserve(num_deps);
  for (int32_t i = 0; i < num_deps; ++i) {
    deps.push_back(dep_ids[i]);
  }
  return deps;
}

void TyphoonGraphBuilder::AddTask(TyphoonTask task) {
  if (submitted_) {
    TyphoonError("Typhoon graph " + std::to_string(graph_id_) +
                 " cannot be mutated after submit");
  }
  if (task.task_id < 0) {
    TyphoonError("Typhoon task_id must be non-negative");
  }
  if (task_index_.count(task.task_id)) {
    TyphoonError("Typhoon runtime duplicate task_id " + std::to_string(task.task_id));
  }
  task_index_[task.task_id] = tasks_.size();
  tasks_.push_back(std::move(task));
}

void TyphoonGraphBuilder::CheckRegionBounds(const TyphoonRegion& region) const {
  if (region.region_id < 0) {
    TyphoonError("Typhoon region_id must be non-negative");
  }
  if (region.offset < 0 || region.size <= 0 || region.alignment <= 0) {
    TyphoonError("Typhoon region bounds must be positive and non-negative");
  }
  if (region.offset % region.alignment != 0) {
    TyphoonError("Typhoon region " + std::to_string(region.region_id) +
                 " violates alignment");
  }
  if (region.offset + region.size > kTyphoonDefaultSRAMSize) {
    TyphoonError("Typhoon region " + std::to_string(region.region_id) + " exceeds SRAM bounds");
  }
}

void TyphoonGraphBuilder::CheckRegionOverlap(const TyphoonRegion& region) const {
  for (const auto& other : regions_) {
    bool overlap =
        std::max(region.offset, other.offset) < std::min(region.offset + region.size, other.offset + other.size);
    if (overlap) {
      TyphoonError("Typhoon regions " + std::to_string(other.region_id) + " and " +
                   std::to_string(region.region_id) + " overlap");
    }
  }
}

void TyphoonGraphBuilder::ValidateTaskRegions() const {
  auto ensure_region = [this](int32_t region_id, const TyphoonTask& task) {
    if (!region_index_.count(region_id)) {
      TyphoonError("Typhoon task " + std::to_string(task.task_id) +
                   " must access declared SRAM regions");
    }
  };

  for (const auto& task : tasks_) {
    for (int32_t region_id : task.reads) {
      ensure_region(region_id, task);
    }
    for (int32_t region_id : task.writes) {
      ensure_region(region_id, task);
    }
  }
}

void TyphoonGraphBuilder::ValidateTaskDependencies() {
  reverse_edges_.assign(tasks_.size(), {});
  for (size_t i = 0; i < tasks_.size(); ++i) {
    for (int32_t dep_id : tasks_[i].deps) {
      auto it = task_index_.find(dep_id);
      if (it == task_index_.end()) {
        TyphoonError("Typhoon task " + std::to_string(tasks_[i].task_id) +
                     " references unknown task dependency " + std::to_string(dep_id));
      }
      reverse_edges_[i].push_back(it->second);
    }
  }

  std::vector<int> state(tasks_.size(), 0);
  for (size_t i = 0; i < tasks_.size(); ++i) {
    DetectCycle(i, &state);
  }

  ancestors_.assign(tasks_.size(), {});
  ancestor_built_.assign(tasks_.size(), false);
  for (size_t i = 0; i < tasks_.size(); ++i) {
    BuildAncestors(i);
  }
}

void TyphoonGraphBuilder::ValidateTaskInitialization() const {
  std::unordered_map<int32_t, std::vector<int32_t>> writers_by_region;
  for (const auto& task : tasks_) {
    for (int32_t region_id : task.writes) {
      writers_by_region[region_id].push_back(task.task_id);
    }
  }

  for (size_t i = 0; i < tasks_.size(); ++i) {
    const auto& task = tasks_[i];
    for (int32_t region_id : task.reads) {
      const auto& region = GetRegion(region_id);
      if (region.preinitialized) {
        continue;
      }

      bool initialized = false;
      auto it = writers_by_region.find(region_id);
      if (it != writers_by_region.end()) {
        for (int32_t writer_task_id : it->second) {
          if (ancestors_[i].count(writer_task_id)) {
            initialized = true;
            break;
          }
        }
      }
      if (!initialized) {
        TyphoonError("Typhoon region " + std::to_string(region_id) +
                     " must be initialized before task " + std::to_string(task.task_id) +
                     " reads it");
      }
    }
  }
}

void TyphoonGraphBuilder::ValidateWriteHazards() const {
  std::unordered_map<int32_t, std::vector<int32_t>> writers_by_region;
  for (const auto& task : tasks_) {
    for (int32_t region_id : task.writes) {
      writers_by_region[region_id].push_back(task.task_id);
    }
  }

  for (const auto& [region_id, writers] : writers_by_region) {
    for (size_t i = 0; i < writers.size(); ++i) {
      for (size_t j = i + 1; j < writers.size(); ++j) {
        size_t lhs = task_index_.at(writers[i]);
        size_t rhs = task_index_.at(writers[j]);
        bool ordered =
            ancestors_[lhs].count(writers[j]) || ancestors_[rhs].count(writers[i]);
        if (!ordered) {
          TyphoonError("Typhoon write hazard on region " + std::to_string(region_id) +
                       " between task " + std::to_string(writers[i]) + " and task " +
                       std::to_string(writers[j]));
        }
      }
    }
  }
}

void TyphoonGraphBuilder::ValidateTaskFootprints() const {
  for (const auto& task : tasks_) {
    if (task.kind == TaskKind::kDMA) {
      const auto& region = task.direction == 0 ? GetRegion(task.writes.at(0)) : GetRegion(task.reads.at(0));
      if (task.bytes > region.size) {
        TyphoonError("Typhoon DMA task " + std::to_string(task.task_id) +
                     " has out-of-bounds size mismatch");
      }
      continue;
    }

    if (task.kind == TaskKind::kVector) {
      int64_t bytes = task.elem_count * DTypeBytes(task.dtype_code);
      for (int32_t region_id : task.reads) {
        if (bytes > GetRegion(region_id).size) {
          TyphoonError("Typhoon vector task " + std::to_string(task.task_id) +
                       " has out-of-bounds size mismatch");
        }
      }
      for (int32_t region_id : task.writes) {
        if (bytes > GetRegion(region_id).size) {
          TyphoonError("Typhoon vector task " + std::to_string(task.task_id) +
                       " has out-of-bounds size mismatch");
        }
      }
      continue;
    }

    if (task.kind == TaskKind::kReshape) {
      int64_t bytes = task.elem_count;
      for (int32_t region_id : task.reads) {
        if (bytes > GetRegion(region_id).size) {
          TyphoonError("Typhoon reshape task " + std::to_string(task.task_id) +
                       " has out-of-bounds size mismatch");
        }
      }
      for (int32_t region_id : task.writes) {
        if (bytes > GetRegion(region_id).size) {
          TyphoonError("Typhoon reshape task " + std::to_string(task.task_id) +
                       " has out-of-bounds size mismatch");
        }
      }
      continue;
    }

    if (task.kind == TaskKind::kMatmul) {
      int64_t dtype_bytes = DTypeBytes(task.dtype_code);
      int64_t a_bytes = task.m * task.k * dtype_bytes;
      int64_t b_bytes = task.k * task.n * dtype_bytes;
      int64_t c_bytes = task.m * task.n * dtype_bytes;
      if (a_bytes > GetRegion(task.reads.at(0)).size || b_bytes > GetRegion(task.reads.at(1)).size ||
          c_bytes > GetRegion(task.writes.at(0)).size) {
        TyphoonError("Typhoon matmul task " + std::to_string(task.task_id) +
                     " has out-of-bounds size mismatch");
      }
    }
  }
}

const TyphoonRegion& TyphoonGraphBuilder::GetRegion(int32_t region_id) const {
  return regions_.at(region_index_.at(region_id));
}

int64_t TyphoonGraphBuilder::DTypeBytes(int32_t dtype_code) const {
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
      TyphoonError("Typhoon dtype_code " + std::to_string(dtype_code) + " is unsupported");
  }
}

const std::unordered_set<int32_t>& TyphoonGraphBuilder::BuildAncestors(size_t task_index) {
  if (ancestor_built_[task_index]) {
    return ancestors_[task_index];
  }

  auto& ancestors = ancestors_[task_index];
  for (size_t dep_index : reverse_edges_[task_index]) {
    int32_t dep_task_id = tasks_[dep_index].task_id;
    ancestors.insert(dep_task_id);
    const auto& dep_ancestors = BuildAncestors(dep_index);
    ancestors.insert(dep_ancestors.begin(), dep_ancestors.end());
  }
  ancestor_built_[task_index] = true;
  return ancestors;
}

void TyphoonGraphBuilder::DetectCycle(size_t task_index, std::vector<int>* state) const {
  if ((*state)[task_index] == 2) {
    return;
  }
  if ((*state)[task_index] == 1) {
    TyphoonError("Typhoon dependency cycle detected at task " +
                 std::to_string(tasks_[task_index].task_id));
  }
  (*state)[task_index] = 1;
  for (size_t dep_index : reverse_edges_[task_index]) {
    DetectCycle(dep_index, state);
  }
  (*state)[task_index] = 2;
}

}  // namespace typhoon
}  // namespace runtime
}  // namespace tvm
