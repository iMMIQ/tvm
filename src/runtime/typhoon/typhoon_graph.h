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

#ifndef TVM_RUNTIME_TYPHOON_GRAPH_H_
#define TVM_RUNTIME_TYPHOON_GRAPH_H_

#include "typhoon_types.h"

#include <stdint.h>

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace tvm {
namespace runtime {
namespace typhoon {

class TyphoonGraphBuilder {
 public:
  explicit TyphoonGraphBuilder(int32_t graph_id);

  void GraphBegin();
  void DeclareRegion(int32_t region_id, int64_t offset, int64_t size, int64_t alignment,
                     bool preinitialized, const char* tag);
  void AddDMATask(int32_t task_id, int32_t direction, void* global_handle,
                  int64_t global_byte_offset, int32_t sram_region_id,
                  int64_t sram_byte_offset, int64_t bytes, int32_t num_deps,
                  const int32_t* dep_ids);
  void AddMatmulTask(int32_t task_id, int32_t a_region_id, int32_t b_region_id,
                     int32_t c_region_id, int64_t m, int64_t n, int64_t k, int32_t dtype_code,
                     int32_t layout_code, int32_t num_deps, const int32_t* dep_ids);
  void AddVectorTask(int32_t task_id, int32_t op_code, int32_t in0_region_id,
                     int32_t in1_region_id, int32_t out_region_id, int64_t elem_count,
                     int32_t dtype_code, int32_t num_metadata, const int64_t* metadata,
                     int32_t num_deps, const int32_t* dep_ids);
  void AddReshapeTask(int32_t task_id, int32_t in_region_id, int32_t out_region_id,
                      int64_t elem_count, int32_t transform_code, int32_t num_metadata,
                      const int64_t* metadata, int32_t num_deps, const int32_t* dep_ids);
  void Submit();
  void Wait() const;
  int32_t graph_id() const { return graph_id_; }
  bool began() const { return began_; }
  const std::vector<TyphoonRegion>& regions() const { return regions_; }
  const std::vector<TyphoonTask>& tasks() const { return tasks_; }
  const std::unordered_map<int32_t, size_t>& task_index() const { return task_index_; }
  const TyphoonRegion& GetRegion(int32_t region_id) const;
  const std::vector<size_t>& topo_order() const { return topo_order_; }

 private:
  std::vector<int32_t> CopyDeps(int32_t num_deps, const int32_t* dep_ids) const;
  std::vector<int64_t> CopyMetadata(int32_t num_metadata, const int64_t* metadata) const;
  void AddTask(TyphoonTask task);
  void CheckRegionBounds(const TyphoonRegion& region) const;
  void CheckRegionOverlap(const TyphoonRegion& region) const;
  void ValidateTaskRegions() const;
  void ValidateTaskDependencies();
  void ValidateTaskInitialization() const;
  void ValidateWriteHazards() const;
  void ValidateTaskFootprints() const;
  int64_t DTypeBytes(int32_t dtype_code) const;
  const std::unordered_set<int32_t>& BuildAncestors(size_t task_index);

  int32_t graph_id_;
  bool began_{false};
  bool submitted_{false};
  std::vector<TyphoonRegion> regions_;
  std::vector<TyphoonTask> tasks_;
  std::unordered_map<int32_t, size_t> region_index_;
  std::unordered_map<int32_t, size_t> task_index_;
  std::vector<std::vector<size_t>> reverse_edges_;
  std::vector<size_t> topo_order_;
  std::vector<std::unordered_set<int32_t>> ancestors_;
  std::vector<bool> ancestor_built_;
};

}  // namespace typhoon
}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_TYPHOON_GRAPH_H_
