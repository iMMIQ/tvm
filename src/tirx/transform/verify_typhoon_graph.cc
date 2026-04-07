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

/*!
 * \file verify_typhoon_graph.cc
 * \brief Verify Typhoon graph structure before intrinsic lowering.
 */
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/target/target.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/function.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <algorithm>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace tvm {
namespace tirx {

namespace {

struct TyphoonRegionInfo {
  int64_t region_id;
  int64_t offset;
  int64_t size;
  int64_t alignment;
  bool preinitialized;
};

struct TyphoonTaskInfo {
  int64_t task_id;
  std::string kind;
  std::vector<int64_t> deps;
  std::vector<int64_t> reads;
  std::vector<int64_t> writes;
  int64_t elem_count{0};
  int64_t dtype_code{0};
  int64_t op_code{0};
  int64_t transform_code{0};
  int64_t sram_byte_offset{0};
  int64_t bytes{0};
  std::vector<int64_t> metadata;
};

class TyphoonGraphVerifier : public StmtExprVisitor {
 public:
  explicit TyphoonGraphVerifier(int64_t sram_size) : sram_size_(sram_size) {}

  void Verify(const PrimFunc& func) {
    VisitStmt(func->body);
    VerifyTaskRegions();
    VerifyDependencies();
    VerifyInitialization();
    VerifyWriteHazards();
    VerifyTaskPayloads();
  }

 private:
  using StmtExprVisitor::VisitExpr_;

  void VisitStmt_(const EvaluateNode* op) final {
    const auto* call = op->value.as<CallNode>();
    if (call == nullptr) {
      StmtExprVisitor::VisitStmt_(op);
      return;
    }

    if (call->op.same_as(region_decl_op_)) {
      CollectRegion(call);
    } else if (call->op.same_as(submit_graph_op_)) {
      CollectGraphOp(call, "submit_graph");
    } else if (call->op.same_as(wait_graph_op_)) {
      CollectGraphOp(call, "wait_graph");
    } else if (call->op.same_as(task_dma_op_)) {
      CollectDMATask(call);
    } else if (call->op.same_as(task_matmul_op_)) {
      CollectTask(call, "matmul", /*num_deps_index=*/10, {2, 3}, {4});
    } else if (call->op.same_as(task_vector_op_)) {
      CollectVectorTask(call);
    } else if (call->op.same_as(task_reshape_op_)) {
      CollectReshapeTask(call);
    }

    StmtExprVisitor::VisitStmt_(op);
  }

  int64_t ExpectInt(const PrimExpr& expr, const char* field) const {
    if (const auto* imm = expr.as<IntImmNode>()) {
      return imm->value;
    }
    TVM_FFI_THROW(ValueError) << "Typhoon graph requires constant integer " << field;
  }

  struct ParsedMetadataAndDeps {
    std::vector<int64_t> metadata;
    std::vector<int64_t> deps;
  };

  ParsedMetadataAndDeps ParseMetadataAndDeps(const CallNode* call, int num_metadata_index,
                                             const std::string& task_kind) const {
    TVM_FFI_CHECK_LT(num_metadata_index, static_cast<int>(call->args.size()), ValueError)
        << "Typhoon " << task_kind << " task is missing metadata count";
    int64_t num_metadata = ExpectInt(call->args[num_metadata_index], "num_metadata");
    TVM_FFI_CHECK_GE(num_metadata, 0, ValueError)
        << "Typhoon " << task_kind << " task metadata count must be non-negative";
    int64_t num_deps_index = num_metadata_index + 1 + num_metadata;
    TVM_FFI_CHECK_LT(num_deps_index, static_cast<int64_t>(call->args.size()), ValueError)
        << "Typhoon " << task_kind << " task is missing dependency metadata";

    ParsedMetadataAndDeps parsed;
    parsed.metadata.reserve(num_metadata);
    for (int64_t i = 0; i < num_metadata; ++i) {
      parsed.metadata.push_back(
          ExpectInt(call->args[num_metadata_index + 1 + i], "metadata_value"));
    }

    int64_t num_deps = ExpectInt(call->args[num_deps_index], "num_deps");
    TVM_FFI_CHECK_GE(num_deps, 0, ValueError)
        << "Typhoon " << task_kind << " task dependency count must be non-negative";
    TVM_FFI_CHECK_EQ(static_cast<int64_t>(call->args.size()), num_deps_index + 1 + num_deps,
                     ValueError)
        << "Typhoon " << task_kind << " task dependency list length mismatch";

    parsed.deps.reserve(num_deps);
    for (int64_t i = 0; i < num_deps; ++i) {
      parsed.deps.push_back(ExpectInt(call->args[num_deps_index + 1 + i], "dep_task_id"));
    }
    return parsed;
  }

  std::vector<int64_t> ParseDeps(const CallNode* call, int num_deps_index,
                                 const std::string& task_kind) const {
    TVM_FFI_CHECK_LT(num_deps_index, static_cast<int>(call->args.size()), ValueError)
        << "Typhoon " << task_kind << " task is missing dependency metadata";
    int64_t num_deps = ExpectInt(call->args[num_deps_index], "num_deps");
    TVM_FFI_CHECK_GE(num_deps, 0, ValueError)
        << "Typhoon " << task_kind << " task dependency count must be non-negative";
    TVM_FFI_CHECK_EQ(static_cast<int64_t>(call->args.size()), num_deps_index + 1 + num_deps,
                     ValueError)
        << "Typhoon " << task_kind << " task dependency list length mismatch";

    std::vector<int64_t> deps;
    deps.reserve(num_deps);
    for (int64_t i = 0; i < num_deps; ++i) {
      deps.push_back(ExpectInt(call->args[num_deps_index + 1 + i], "dep_task_id"));
    }
    return deps;
  }

  int64_t RequireGraphId(const PrimExpr& expr, const char* op_name) {
    int64_t graph_id = ExpectInt(expr, "graph_id");
    if (!graph_id_seen_) {
      graph_id_ = graph_id;
      graph_id_seen_ = true;
      return graph_id;
    }
    TVM_FFI_CHECK_EQ(graph_id_, graph_id, ValueError)
        << "Typhoon PrimFunc must use a single graph_id, but " << op_name << " uses "
        << graph_id << " after graph_id " << graph_id_;
    return graph_id;
  }

  void CollectGraphOp(const CallNode* call, const char* op_name) {
    TVM_FFI_CHECK_EQ(call->args.size(), 1U, ValueError)
        << "tirx.typhoon." << op_name << " expects graph_id only";
    RequireGraphId(call->args[0], op_name);
  }

  void CollectRegion(const CallNode* call) {
    TVM_FFI_CHECK_EQ(call->args.size(), 7U, ValueError)
        << "tirx.typhoon.region_decl expects 7 arguments";

    RequireGraphId(call->args[0], "region_decl");
    int64_t region_id = ExpectInt(call->args[1], "region_id");
    int64_t offset = ExpectInt(call->args[2], "offset");
    int64_t size = ExpectInt(call->args[3], "size");
    int64_t alignment = ExpectInt(call->args[4], "alignment");
    int64_t preinitialized = ExpectInt(call->args[5], "preinitialized");

    TVM_FFI_CHECK_GE(region_id, 0, ValueError) << "Typhoon region_id must be non-negative";
    TVM_FFI_CHECK_GE(offset, 0, ValueError) << "Typhoon region offset must be non-negative";
    TVM_FFI_CHECK_GT(size, 0, ValueError) << "Typhoon region size must be positive";
    TVM_FFI_CHECK_GT(alignment, 0, ValueError) << "Typhoon region alignment must be positive";
    TVM_FFI_CHECK_EQ(offset % alignment, 0, ValueError)
        << "Typhoon region " << region_id << " offset does not satisfy alignment";
    TVM_FFI_CHECK_LE(offset + size, sram_size_, ValueError)
        << "Typhoon region " << region_id << " exceeds SRAM bounds";
    TVM_FFI_CHECK(!regions_.count(region_id), ValueError)
        << "Typhoon graph has duplicate region_id " << region_id;

    for (const auto& [other_id, other] : regions_) {
      bool overlap = std::max(offset, other.offset) < std::min(offset + size, other.offset + other.size);
      TVM_FFI_CHECK(!overlap, ValueError)
          << "Typhoon regions " << other_id << " and " << region_id << " overlap";
    }

    regions_.emplace(region_id,
                     TyphoonRegionInfo{region_id, offset, size, alignment, preinitialized != 0});
  }

  void CollectDMATask(const CallNode* call) {
    TVM_FFI_CHECK_GE(call->args.size(), 9U, ValueError)
        << "tirx.typhoon.task_dma expects dependency metadata";

    TyphoonTaskInfo task;
    RequireGraphId(call->args[0], "task_dma");
    task.task_id = ExpectInt(call->args[1], "task_id");
    task.kind = "dma";
    task.deps = ParseDeps(call, /*num_deps_index=*/8, task.kind);

    int64_t direction = ExpectInt(call->args[2], "direction");
    int64_t sram_region_id = ExpectInt(call->args[5], "sram_region_id");
    task.sram_byte_offset = ExpectInt(call->args[6], "sram_byte_offset");
    int64_t bytes = ExpectInt(call->args[7], "bytes");
    task.bytes = bytes;
    TVM_FFI_CHECK_GE(task.task_id, 0, ValueError) << "Typhoon task_id must be non-negative";
    TVM_FFI_CHECK_GE(task.sram_byte_offset, 0, ValueError)
        << "Typhoon DMA SRAM byte offset must be non-negative";
    TVM_FFI_CHECK_GT(bytes, 0, ValueError) << "Typhoon DMA bytes must be positive";
    TVM_FFI_CHECK_GE(sram_region_id, 0, ValueError)
        << "Typhoon DMA SRAM region id must be non-negative";
    if (direction == 0) {
      task.writes = {sram_region_id};
    } else if (direction == 1) {
      task.reads = {sram_region_id};
    } else {
      TVM_FFI_THROW(ValueError) << "Typhoon DMA direction must be 0 or 1";
    }

    AddTask(std::move(task));
  }

  void CollectVectorTask(const CallNode* call) {
    TVM_FFI_CHECK_GT(static_cast<int>(call->args.size()), 8, ValueError)
        << "tirx.typhoon.task_vector is missing required operands";

    TyphoonTaskInfo task;
    RequireGraphId(call->args[0], "vector");
    task.task_id = ExpectInt(call->args[1], "task_id");
    task.kind = "vector";
    task.op_code = ExpectInt(call->args[2], "op_code");
    task.elem_count = ExpectInt(call->args[6], "elem_count");
    task.dtype_code = ExpectInt(call->args[7], "dtype_code");
    TVM_FFI_CHECK_GE(task.task_id, 0, ValueError) << "Typhoon task_id must be non-negative";

    auto parsed = ParseMetadataAndDeps(call, /*num_metadata_index=*/8, task.kind);
    task.metadata = std::move(parsed.metadata);
    task.deps = std::move(parsed.deps);
    task.reads = {ExpectInt(call->args[3], "region_id")};
    if (task.op_code == 0) {
      task.reads.push_back(ExpectInt(call->args[4], "region_id"));
    }
    task.writes = {ExpectInt(call->args[5], "region_id")};

    AddTask(std::move(task));
  }

  void CollectReshapeTask(const CallNode* call) {
    TVM_FFI_CHECK_GT(static_cast<int>(call->args.size()), 6, ValueError)
        << "tirx.typhoon.task_reshape is missing required operands";

    TyphoonTaskInfo task;
    RequireGraphId(call->args[0], "reshape");
    task.task_id = ExpectInt(call->args[1], "task_id");
    task.kind = "reshape";
    task.elem_count = ExpectInt(call->args[4], "elem_count");
    task.transform_code = ExpectInt(call->args[5], "transform_code");
    TVM_FFI_CHECK_GE(task.task_id, 0, ValueError) << "Typhoon task_id must be non-negative";

    auto parsed = ParseMetadataAndDeps(call, /*num_metadata_index=*/6, task.kind);
    task.metadata = std::move(parsed.metadata);
    task.deps = std::move(parsed.deps);
    task.reads = {ExpectInt(call->args[2], "region_id")};
    task.writes = {ExpectInt(call->args[3], "region_id")};

    AddTask(std::move(task));
  }

  void CollectTask(const CallNode* call, const char* kind, int num_deps_index,
                   std::initializer_list<int> read_arg_indices,
                   std::initializer_list<int> write_arg_indices) {
    int max_arg_index = num_deps_index;
    for (int index : read_arg_indices) {
      max_arg_index = std::max(max_arg_index, index);
    }
    for (int index : write_arg_indices) {
      max_arg_index = std::max(max_arg_index, index);
    }
    TVM_FFI_CHECK_GT(static_cast<int>(call->args.size()), max_arg_index, ValueError)
        << "tirx.typhoon.task_" << kind << " is missing required operands";

    TyphoonTaskInfo task;
    RequireGraphId(call->args[0], kind);
    task.task_id = ExpectInt(call->args[1], "task_id");
    task.kind = kind;
    task.deps = ParseDeps(call, num_deps_index, task.kind);
    TVM_FFI_CHECK_GE(task.task_id, 0, ValueError) << "Typhoon task_id must be non-negative";

    for (int index : read_arg_indices) {
      task.reads.push_back(ExpectInt(call->args[index], "region_id"));
    }
    for (int index : write_arg_indices) {
      task.writes.push_back(ExpectInt(call->args[index], "region_id"));
    }

    AddTask(std::move(task));
  }

  void AddTask(TyphoonTaskInfo task) {
    TVM_FFI_CHECK(!task_id_to_index_.count(task.task_id), ValueError)
        << "Typhoon graph has duplicate task_id " << task.task_id;
    task_id_to_index_[task.task_id] = tasks_.size();
    tasks_.push_back(std::move(task));
  }

  void VerifyTaskRegions() const {
    auto check_region = [this](int64_t region_id, const TyphoonTaskInfo& task, const char* role) {
      TVM_FFI_CHECK(regions_.count(region_id), ValueError)
          << "Typhoon task " << task.task_id << " references unknown " << role << " region_id "
          << region_id;
    };

    for (const auto& task : tasks_) {
      for (int64_t region_id : task.reads) {
        check_region(region_id, task, "input");
      }
      for (int64_t region_id : task.writes) {
        check_region(region_id, task, "output");
      }
    }
  }

  void VerifyDependencies() {
    forward_edges_.assign(tasks_.size(), {});
    reverse_edges_.assign(tasks_.size(), {});

    for (size_t i = 0; i < tasks_.size(); ++i) {
      for (int64_t dep_id : tasks_[i].deps) {
        auto it = task_id_to_index_.find(dep_id);
        TVM_FFI_CHECK(it != task_id_to_index_.end(), ValueError)
            << "Typhoon task " << tasks_[i].task_id << " references unknown dependency "
            << dep_id;
        reverse_edges_[i].push_back(it->second);
        forward_edges_[it->second].push_back(i);
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

  void DetectCycle(size_t index, std::vector<int>* state) const {
    if ((*state)[index] == 2) {
      return;
    }
    TVM_FFI_CHECK_NE((*state)[index], 1, ValueError)
        << "Typhoon task dependency cycle detected at task_id " << tasks_[index].task_id;

    (*state)[index] = 1;
    for (size_t dep_index : reverse_edges_[index]) {
      DetectCycle(dep_index, state);
    }
    (*state)[index] = 2;
  }

  const std::unordered_set<int64_t>& BuildAncestors(size_t index) {
    if (ancestor_built_[index]) {
      return ancestors_[index];
    }

    auto& ancestors = ancestors_[index];
    for (size_t dep_index : reverse_edges_[index]) {
      int64_t dep_task_id = tasks_[dep_index].task_id;
      ancestors.insert(dep_task_id);
      const auto& dep_ancestors = BuildAncestors(dep_index);
      ancestors.insert(dep_ancestors.begin(), dep_ancestors.end());
    }
    ancestor_built_[index] = true;
    return ancestors;
  }

  void VerifyInitialization() const {
    std::unordered_map<int64_t, std::vector<int64_t>> writers_by_region;
    for (const auto& task : tasks_) {
      for (int64_t region_id : task.writes) {
        writers_by_region[region_id].push_back(task.task_id);
      }
    }

    for (size_t i = 0; i < tasks_.size(); ++i) {
      const auto& task = tasks_[i];
      for (int64_t region_id : task.reads) {
        const auto& region = regions_.at(region_id);
        if (region.preinitialized) {
          continue;
        }

        bool initialized = false;
        auto it = writers_by_region.find(region_id);
        if (it != writers_by_region.end()) {
          for (int64_t writer_task_id : it->second) {
            if (ancestors_[i].count(writer_task_id)) {
              initialized = true;
              break;
            }
          }
        }

        TVM_FFI_CHECK(initialized, ValueError)
            << "Typhoon region " << region_id << " must be initialized before task "
            << task.task_id << " reads it";
      }
    }
  }

  void VerifyWriteHazards() const {
    std::unordered_map<int64_t, std::vector<int64_t>> writers_by_region;
    for (const auto& task : tasks_) {
      for (int64_t region_id : task.writes) {
        writers_by_region[region_id].push_back(task.task_id);
      }
    }

    for (const auto& [region_id, writers] : writers_by_region) {
      for (size_t i = 0; i < writers.size(); ++i) {
        for (size_t j = i + 1; j < writers.size(); ++j) {
          size_t lhs_index = task_id_to_index_.at(writers[i]);
          size_t rhs_index = task_id_to_index_.at(writers[j]);
          bool ordered = ancestors_[lhs_index].count(writers[j]) || ancestors_[rhs_index].count(writers[i]);
          TVM_FFI_CHECK(ordered, ValueError)
              << "Typhoon write hazard on region " << region_id << " between task "
              << writers[i] << " and task " << writers[j];
        }
      }
    }
  }

  void VerifyTaskPayloads() const {
    auto expect_positive = [](int64_t value, const char* field, const TyphoonTaskInfo& task) {
      TVM_FFI_CHECK_GT(value, 0, ValueError)
          << "Typhoon " << task.kind << " task " << task.task_id << " requires positive "
          << field;
    };
    auto expect_non_negative = [](int64_t value, const char* field, const TyphoonTaskInfo& task) {
      TVM_FFI_CHECK_GE(value, 0, ValueError)
          << "Typhoon " << task.kind << " task " << task.task_id << " requires non-negative "
          << field;
    };

    for (const auto& task : tasks_) {
      if (task.kind == "dma") {
        const auto& region =
            task.reads.empty() ? regions_.at(task.writes[0]) : regions_.at(task.reads[0]);
        TVM_FFI_CHECK_LE(task.sram_byte_offset + task.bytes, region.size, ValueError)
            << "Typhoon DMA task " << task.task_id << " has out-of-bounds size mismatch";
        continue;
      }
      if (task.kind == "vector") {
        switch (task.op_code) {
          case 0:
          case 1: {
            int64_t bytes = task.elem_count * DTypeBytes(task.dtype_code);
            TVM_FFI_CHECK_LE(bytes, regions_.at(task.reads[0]).size, ValueError)
                << "Typhoon vector task " << task.task_id << " has out-of-bounds size mismatch";
            if (task.op_code == 0 && task.metadata.empty()) {
              TVM_FFI_CHECK_LE(bytes, regions_.at(task.reads[1]).size, ValueError)
                  << "Typhoon vector task " << task.task_id
                  << " has out-of-bounds size mismatch";
            } else if (task.op_code == 0) {
              TVM_FFI_CHECK_EQ(task.metadata.size(), 2U, ValueError)
                  << "Typhoon vector broadcast-add task " << task.task_id
                  << " expects [outer, inner] metadata";
              int64_t outer = task.metadata[0];
              int64_t inner = task.metadata[1];
              expect_positive(outer, "outer", task);
              expect_positive(inner, "inner", task);
              TVM_FFI_CHECK_EQ(task.elem_count, outer * inner, ValueError)
                  << "Typhoon vector broadcast-add task " << task.task_id
                  << " has elem_count mismatch";
              TVM_FFI_CHECK_LE(inner * DTypeBytes(task.dtype_code), regions_.at(task.reads[1]).size,
                               ValueError)
                  << "Typhoon vector task " << task.task_id
                  << " has out-of-bounds size mismatch";
            } else {
              TVM_FFI_CHECK(task.metadata.empty(), ValueError)
                  << "Typhoon vector task " << task.task_id
                  << " does not accept metadata for relu";
            }
            TVM_FFI_CHECK_LE(bytes, regions_.at(task.writes[0]).size, ValueError)
                << "Typhoon vector task " << task.task_id << " has out-of-bounds size mismatch";
            break;
          }
          case 2: {
            TVM_FFI_CHECK_EQ(task.metadata.size(), 12U, ValueError)
                << "Typhoon vector maxpool task " << task.task_id
                << " expects 12 metadata values";
            TVM_FFI_CHECK_EQ(task.dtype_code, 2, ValueError)
                << "Typhoon vector maxpool task " << task.task_id
                << " currently requires dtype_code=2";
            int64_t n = task.metadata[0];
            int64_t c = task.metadata[1];
            int64_t in_h = task.metadata[2];
            int64_t in_w = task.metadata[3];
            int64_t kernel_h = task.metadata[4];
            int64_t kernel_w = task.metadata[5];
            int64_t stride_h = task.metadata[6];
            int64_t stride_w = task.metadata[7];
            int64_t pad_h = task.metadata[8];
            int64_t pad_w = task.metadata[9];
            int64_t out_h = task.metadata[10];
            int64_t out_w = task.metadata[11];
            expect_positive(n, "n", task);
            expect_positive(c, "c", task);
            expect_positive(in_h, "in_h", task);
            expect_positive(in_w, "in_w", task);
            expect_positive(kernel_h, "kernel_h", task);
            expect_positive(kernel_w, "kernel_w", task);
            expect_positive(stride_h, "stride_h", task);
            expect_positive(stride_w, "stride_w", task);
            expect_non_negative(pad_h, "pad_h", task);
            expect_non_negative(pad_w, "pad_w", task);
            int64_t numer_h = in_h + 2 * pad_h - kernel_h;
            int64_t numer_w = in_w + 2 * pad_w - kernel_w;
            TVM_FFI_CHECK_GE(numer_h, 0, ValueError)
                << "Typhoon vector maxpool task " << task.task_id << " has invalid height";
            TVM_FFI_CHECK_GE(numer_w, 0, ValueError)
                << "Typhoon vector maxpool task " << task.task_id << " has invalid width";
            TVM_FFI_CHECK_EQ(out_h, numer_h / stride_h + 1, ValueError)
                << "Typhoon vector maxpool task " << task.task_id
                << " has maxpool output height mismatch";
            TVM_FFI_CHECK_EQ(out_w, numer_w / stride_w + 1, ValueError)
                << "Typhoon vector maxpool task " << task.task_id
                << " has maxpool output width mismatch";
            TVM_FFI_CHECK_EQ(task.elem_count, n * c * out_h * out_w, ValueError)
                << "Typhoon vector maxpool task " << task.task_id
                << " has elem_count mismatch";
            TVM_FFI_CHECK_LE(n * c * in_h * in_w * 4, regions_.at(task.reads[0]).size, ValueError)
                << "Typhoon vector task " << task.task_id << " has out-of-bounds size mismatch";
            TVM_FFI_CHECK_LE(task.elem_count * 4, regions_.at(task.writes[0]).size, ValueError)
                << "Typhoon vector task " << task.task_id << " has out-of-bounds size mismatch";
            break;
          }
          case 3: {
            TVM_FFI_CHECK_EQ(task.metadata.size(), 4U, ValueError)
                << "Typhoon vector global_average_pool task " << task.task_id
                << " expects 4 metadata values";
            TVM_FFI_CHECK_EQ(task.dtype_code, 2, ValueError)
                << "Typhoon vector global_average_pool task " << task.task_id
                << " currently requires dtype_code=2";
            int64_t n = task.metadata[0];
            int64_t c = task.metadata[1];
            int64_t in_h = task.metadata[2];
            int64_t in_w = task.metadata[3];
            expect_positive(n, "n", task);
            expect_positive(c, "c", task);
            expect_positive(in_h, "in_h", task);
            expect_positive(in_w, "in_w", task);
            TVM_FFI_CHECK_EQ(task.elem_count, n * c, ValueError)
                << "Typhoon vector global_average_pool task " << task.task_id
                << " has elem_count mismatch";
            TVM_FFI_CHECK_LE(n * c * in_h * in_w * 4, regions_.at(task.reads[0]).size, ValueError)
                << "Typhoon vector task " << task.task_id << " has out-of-bounds size mismatch";
            TVM_FFI_CHECK_LE(task.elem_count * 4, regions_.at(task.writes[0]).size, ValueError)
                << "Typhoon vector task " << task.task_id << " has out-of-bounds size mismatch";
            break;
          }
          default:
            TVM_FFI_THROW(ValueError)
                << "Typhoon vector task " << task.task_id << " uses unsupported op_code "
                << task.op_code;
        }
      } else if (task.kind == "reshape") {
        switch (task.transform_code) {
          case 0:
            TVM_FFI_CHECK(task.metadata.empty(), ValueError)
                << "Typhoon reshape task " << task.task_id
                << " does not accept metadata for copy/reorder";
            TVM_FFI_CHECK_LE(task.elem_count, regions_.at(task.reads[0]).size, ValueError)
                << "Typhoon reshape task " << task.task_id << " has out-of-bounds size mismatch";
            TVM_FFI_CHECK_LE(task.elem_count, regions_.at(task.writes[0]).size, ValueError)
                << "Typhoon reshape task " << task.task_id << " has out-of-bounds size mismatch";
            break;
          case 1: {
            TVM_FFI_CHECK(task.metadata.size() == 12U || task.metadata.size() == 16U, ValueError)
                << "Typhoon reshape im2col task " << task.task_id
                << " expects 12 or 16 metadata values";
            int64_t n = task.metadata[0];
            int64_t c = task.metadata[1];
            int64_t in_h = task.metadata[2];
            int64_t in_w = task.metadata[3];
            int64_t kernel_h = task.metadata[4];
            int64_t kernel_w = task.metadata[5];
            int64_t stride_h = task.metadata[6];
            int64_t stride_w = task.metadata[7];
            int64_t pad_h = task.metadata[8];
            int64_t pad_w = task.metadata[9];
            int64_t out_h = task.metadata[10];
            int64_t out_w = task.metadata[11];
            expect_positive(n, "n", task);
            expect_positive(c, "c", task);
            expect_positive(in_h, "in_h", task);
            expect_positive(in_w, "in_w", task);
            expect_positive(kernel_h, "kernel_h", task);
            expect_positive(kernel_w, "kernel_w", task);
            expect_positive(stride_h, "stride_h", task);
            expect_positive(stride_w, "stride_w", task);
            expect_non_negative(pad_h, "pad_h", task);
            expect_non_negative(pad_w, "pad_w", task);
            bool tiled = task.metadata.size() == 16U;
            if (!tiled) {
              int64_t numer_h = in_h + 2 * pad_h - kernel_h;
              int64_t numer_w = in_w + 2 * pad_w - kernel_w;
              TVM_FFI_CHECK_GE(numer_h, 0, ValueError)
                  << "Typhoon reshape im2col task " << task.task_id << " has invalid height";
              TVM_FFI_CHECK_GE(numer_w, 0, ValueError)
                  << "Typhoon reshape im2col task " << task.task_id << " has invalid width";
              TVM_FFI_CHECK_EQ(out_h, numer_h / stride_h + 1, ValueError)
                  << "Typhoon reshape im2col task " << task.task_id
                  << " has im2col output height mismatch";
              TVM_FFI_CHECK_EQ(out_w, numer_w / stride_w + 1, ValueError)
                  << "Typhoon reshape im2col task " << task.task_id
                  << " has im2col output width mismatch";
            } else {
              for (size_t i = 12; i < 16; ++i) {
                TVM_FFI_CHECK_GE(task.metadata[i], 0, ValueError)
                    << "Typhoon reshape im2col task " << task.task_id
                    << " requires non-negative tile origins";
              }
              expect_positive(out_h, "out_h", task);
              expect_positive(out_w, "out_w", task);
            }
            TVM_FFI_CHECK_EQ(task.elem_count, n * c * kernel_h * kernel_w * out_h * out_w * 4,
                             ValueError)
                << "Typhoon reshape im2col task " << task.task_id
                << " has elem_count mismatch";
            TVM_FFI_CHECK_LE(n * c * in_h * in_w * 4, regions_.at(task.reads[0]).size, ValueError)
                << "Typhoon reshape task " << task.task_id << " has out-of-bounds size mismatch";
            TVM_FFI_CHECK_LE(task.elem_count, regions_.at(task.writes[0]).size, ValueError)
                << "Typhoon reshape task " << task.task_id << " has out-of-bounds size mismatch";
            break;
          }
          case 2: {
            TVM_FFI_CHECK_EQ(task.metadata.size(), 2U, ValueError)
                << "Typhoon reshape transpose task " << task.task_id
                << " expects [rows, cols] metadata";
            int64_t rows = task.metadata[0];
            int64_t cols = task.metadata[1];
            expect_positive(rows, "rows", task);
            expect_positive(cols, "cols", task);
            TVM_FFI_CHECK_EQ(task.elem_count, rows * cols * 4, ValueError)
                << "Typhoon reshape transpose task " << task.task_id
                << " has elem_count mismatch";
            TVM_FFI_CHECK_LE(task.elem_count, regions_.at(task.reads[0]).size, ValueError)
                << "Typhoon reshape task " << task.task_id << " has out-of-bounds size mismatch";
            TVM_FFI_CHECK_LE(task.elem_count, regions_.at(task.writes[0]).size, ValueError)
                << "Typhoon reshape task " << task.task_id << " has out-of-bounds size mismatch";
            break;
          }
          default:
            TVM_FFI_THROW(ValueError)
                << "Typhoon reshape task " << task.task_id
                << " uses unsupported transform_code " << task.transform_code;
        }
      }
    }
  }

  int64_t sram_size_;
  std::unordered_map<int64_t, TyphoonRegionInfo> regions_;
  std::vector<TyphoonTaskInfo> tasks_;
  std::unordered_map<int64_t, size_t> task_id_to_index_;
  std::vector<std::vector<size_t>> forward_edges_;
  std::vector<std::vector<size_t>> reverse_edges_;
  std::vector<std::unordered_set<int64_t>> ancestors_;
  std::vector<bool> ancestor_built_;

  const Op& region_decl_op_ = Op::Get("tirx.typhoon.region_decl");
  const Op& submit_graph_op_ = Op::Get("tirx.typhoon.submit_graph");
  const Op& task_dma_op_ = Op::Get("tirx.typhoon.task_dma");
  const Op& task_matmul_op_ = Op::Get("tirx.typhoon.task_matmul");
  const Op& task_vector_op_ = Op::Get("tirx.typhoon.task_vector");
  const Op& task_reshape_op_ = Op::Get("tirx.typhoon.task_reshape");
  const Op& wait_graph_op_ = Op::Get("tirx.typhoon.wait_graph");

  bool graph_id_seen_{false};
  int64_t graph_id_{-1};

  int64_t DTypeBytes(int64_t dtype_code) const {
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
        TVM_FFI_THROW(ValueError) << "Typhoon dtype_code " << dtype_code << " is unsupported";
    }
  }
};

}  // namespace

namespace transform {

Pass VerifyTyphoonGraph() {
  auto pass_func = [](IRModule mod, PassContext ctx) {
    for (const auto& [gvar, base_func] : mod->functions) {
      const auto* func = base_func.as<PrimFuncNode>();
      if (func == nullptr) {
        continue;
      }

      auto target = func->GetAttr<Target>(tvm::attr::kTarget);
      if (!target.defined() || target.value()->kind->name != "typhoon") {
        continue;
      }

      int64_t sram_size = target.value()->GetAttr<Integer>("sram_size", Integer(1048576)).value().IntValue();
      TyphoonGraphVerifier(sram_size).Verify(ffi::GetRef<PrimFunc>(func));
    }
    return mod;
  };
  return CreateModulePass(pass_func, 0, "tirx.VerifyTyphoonGraph", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tirx.transform.VerifyTyphoonGraph", VerifyTyphoonGraph);
}

}  // namespace transform

}  // namespace tirx
}  // namespace tvm
