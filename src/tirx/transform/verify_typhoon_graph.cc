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
      CollectTask(call, "vector", /*num_deps_index=*/8, {3, 4}, {5});
    } else if (call->op.same_as(task_reshape_op_)) {
      CollectTask(call, "reshape", /*num_deps_index=*/6, {2}, {3});
    }

    StmtExprVisitor::VisitStmt_(op);
  }

  int64_t ExpectInt(const PrimExpr& expr, const char* field) const {
    if (const auto* imm = expr.as<IntImmNode>()) {
      return imm->value;
    }
    TVM_FFI_THROW(ValueError) << "Typhoon graph requires constant integer " << field;
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
    TVM_FFI_CHECK_GE(call->args.size(), 8U, ValueError)
        << "tirx.typhoon.task_dma expects dependency metadata";

    TyphoonTaskInfo task;
    RequireGraphId(call->args[0], "task_dma");
    task.task_id = ExpectInt(call->args[1], "task_id");
    task.kind = "dma";
    task.deps = ParseDeps(call, /*num_deps_index=*/7, task.kind);

    int64_t direction = ExpectInt(call->args[2], "direction");
    int64_t sram_region_id = ExpectInt(call->args[5], "sram_region_id");
    int64_t bytes = ExpectInt(call->args[6], "bytes");
    TVM_FFI_CHECK_GE(task.task_id, 0, ValueError) << "Typhoon task_id must be non-negative";
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
