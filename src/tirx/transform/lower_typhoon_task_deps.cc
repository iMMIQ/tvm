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
 * \file lower_typhoon_task_deps.cc
 * \brief Materialize Typhoon task dependency lists for the runtime ABI.
 */
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/type.h>
#include <tvm/target/target.h>
#include <tvm/tirx/buffer.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/function.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <string>

#include "ir_utils.h"

namespace tvm {
namespace tirx {

class TyphoonTaskDepLowerer : public StmtExprMutator {
 public:
  using Parent = StmtExprMutator;

  Stmt VisitStmt_(const EvaluateNode* op) final {
    Stmt stmt = Parent::VisitStmt_(op);
    const auto* eval = stmt.as<EvaluateNode>();
    if (eval == nullptr) {
      return stmt;
    }
    const auto* call = eval->value.as<CallNode>();
    if (call == nullptr) {
      return stmt;
    }

    if (call->op.same_as(task_dma_op_)) {
      return LowerTask(call, /*num_deps_index=*/8, "TVMTyphoonAddDMATask", op->span);
    }
    if (call->op.same_as(task_batched_dma_op_)) {
      return LowerTask(call, /*num_deps_index=*/11, "TVMTyphoonAddBatchedDMATask", op->span);
    }
    if (call->op.same_as(task_matmul_op_)) {
      return LowerTask(call, /*num_deps_index=*/10, "TVMTyphoonAddMatmulTask", op->span);
    }
    if (call->op.same_as(task_vector_op_)) {
      return LowerTaskWithMetadata(call, /*num_metadata_index=*/8, "TVMTyphoonAddVectorTask",
                                   op->span);
    }
    if (call->op.same_as(task_reshape_op_)) {
      return LowerTaskWithMetadata(call, /*num_metadata_index=*/6, "TVMTyphoonAddReshapeTask",
                                   op->span);
    }
    return stmt;
  }

 private:
  Stmt LowerTask(const CallNode* call, int num_deps_index, const char* symbol_name, Span span) {
    TVM_FFI_ICHECK_LT(num_deps_index, static_cast<int>(call->args.size()));
    const auto* num_deps_imm = call->args[num_deps_index].as<IntImmNode>();
    TVM_FFI_ICHECK(num_deps_imm != nullptr)
        << "Typhoon task dependency count must be a constant integer";
    TVM_FFI_ICHECK_GE(num_deps_imm->value, 0);
    int num_deps = static_cast<int>(num_deps_imm->value);
    TVM_FFI_ICHECK_EQ(static_cast<int>(call->args.size()), num_deps_index + 1 + num_deps)
        << "Typhoon task dependency list length mismatch";

    ffi::Array<PrimExpr> extern_args{StringImm(symbol_name)};
    for (int i = 0; i <= num_deps_index; ++i) {
      extern_args.push_back(call->args[i]);
    }

    if (num_deps == 0) {
      extern_args.push_back(make_zero(DataType::Handle()));
      return Evaluate(Call(call->dtype, builtin::call_extern(), extern_args), span);
    }

    Var dep_ids_var("typhoon_dep_ids_" + std::to_string(counter_++),
                    PointerType(PrimType(DataType::Int(32))));
    Buffer dep_ids_buffer(dep_ids_var, DataType::Int(32), {ConstInt32(num_deps)}, {},
                          IntImm(DataType::Int(32), 0), dep_ids_var->name_hint, 0, 0, kDefault);

    ffi::Array<Stmt> seq{
        Bind(dep_ids_var, StackAlloca("int32", num_deps), span), DeclBuffer(dep_ids_buffer, span)};
    for (int i = 0; i < num_deps; ++i) {
      seq.push_back(BufferStore(dep_ids_buffer, cast(DataType::Int(32), call->args[num_deps_index + 1 + i]),
                                {ConstInt32(i)}));
    }

    extern_args.push_back(cast(DataType::Handle(), dep_ids_var));
    seq.push_back(Evaluate(Call(call->dtype, builtin::call_extern(), extern_args), span));
    return SeqStmt(seq, span);
  }

  Stmt LowerTaskWithMetadata(const CallNode* call, int num_metadata_index, const char* symbol_name,
                             Span span) {
    TVM_FFI_ICHECK_LT(num_metadata_index, static_cast<int>(call->args.size()));
    const auto* num_metadata_imm = call->args[num_metadata_index].as<IntImmNode>();
    TVM_FFI_ICHECK(num_metadata_imm != nullptr)
        << "Typhoon task metadata count must be a constant integer";
    TVM_FFI_ICHECK_GE(num_metadata_imm->value, 0);
    int num_metadata = static_cast<int>(num_metadata_imm->value);
    int num_deps_index = num_metadata_index + 1 + num_metadata;
    TVM_FFI_ICHECK_LT(num_deps_index, static_cast<int>(call->args.size()));

    const auto* num_deps_imm = call->args[num_deps_index].as<IntImmNode>();
    TVM_FFI_ICHECK(num_deps_imm != nullptr)
        << "Typhoon task dependency count must be a constant integer";
    TVM_FFI_ICHECK_GE(num_deps_imm->value, 0);
    int num_deps = static_cast<int>(num_deps_imm->value);
    TVM_FFI_ICHECK_EQ(static_cast<int>(call->args.size()), num_deps_index + 1 + num_deps)
        << "Typhoon task dependency list length mismatch";

    ffi::Array<PrimExpr> extern_args{StringImm(symbol_name)};
    for (int i = 0; i <= num_metadata_index; ++i) {
      extern_args.push_back(call->args[i]);
    }

    ffi::Array<Stmt> seq;
    if (num_metadata == 0) {
      extern_args.push_back(make_zero(DataType::Handle()));
    } else {
      Var metadata_var("typhoon_metadata_" + std::to_string(counter_++),
                       PointerType(PrimType(DataType::Int(64))));
      Buffer metadata_buffer(metadata_var, DataType::Int(64), {ConstInt32(num_metadata)}, {},
                             IntImm(DataType::Int(32), 0), metadata_var->name_hint, 0, 0,
                             kDefault);
      seq.push_back(Bind(metadata_var, StackAlloca("int64", num_metadata), span));
      seq.push_back(DeclBuffer(metadata_buffer, span));
      for (int i = 0; i < num_metadata; ++i) {
        seq.push_back(BufferStore(metadata_buffer, cast(DataType::Int(64),
                                                        call->args[num_metadata_index + 1 + i]),
                                  {ConstInt32(i)}));
      }
      extern_args.push_back(cast(DataType::Handle(), metadata_var));
    }

    extern_args.push_back(call->args[num_deps_index]);
    if (num_deps == 0) {
      extern_args.push_back(make_zero(DataType::Handle()));
      Stmt call_stmt = Evaluate(Call(call->dtype, builtin::call_extern(), extern_args), span);
      if (seq.empty()) {
        return call_stmt;
      }
      seq.push_back(call_stmt);
      return SeqStmt(seq, span);
    }

    Var dep_ids_var("typhoon_dep_ids_" + std::to_string(counter_++),
                    PointerType(PrimType(DataType::Int(32))));
    Buffer dep_ids_buffer(dep_ids_var, DataType::Int(32), {ConstInt32(num_deps)}, {},
                          IntImm(DataType::Int(32), 0), dep_ids_var->name_hint, 0, 0, kDefault);
    seq.push_back(Bind(dep_ids_var, StackAlloca("int32", num_deps), span));
    seq.push_back(DeclBuffer(dep_ids_buffer, span));
    for (int i = 0; i < num_deps; ++i) {
      seq.push_back(BufferStore(dep_ids_buffer, cast(DataType::Int(32), call->args[num_deps_index + 1 + i]),
                                {ConstInt32(i)}));
    }

    extern_args.push_back(cast(DataType::Handle(), dep_ids_var));
    seq.push_back(Evaluate(Call(call->dtype, builtin::call_extern(), extern_args), span));
    return SeqStmt(seq, span);
  }

  int counter_{0};
  const Op& task_dma_op_ = Op::Get("tirx.typhoon.task_dma");
  const Op& task_batched_dma_op_ = Op::Get("tirx.typhoon.task_batched_dma");
  const Op& task_matmul_op_ = Op::Get("tirx.typhoon.task_matmul");
  const Op& task_vector_op_ = Op::Get("tirx.typhoon.task_vector");
  const Op& task_reshape_op_ = Op::Get("tirx.typhoon.task_reshape");
};

namespace transform {

Pass LowerTyphoonTaskDeps() {
  auto pass_func = [](PrimFunc f, IRModule m, PassContext ctx) {
    auto target = f->GetAttr<Target>(tvm::attr::kTarget);
    if (!target.defined() || target.value()->kind->name != "typhoon") {
      return f;
    }
    auto* n = f.CopyOnWrite();
    n->body = TyphoonTaskDepLowerer()(std::move(n->body));
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tirx.LowerTyphoonTaskDeps", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tirx.transform.LowerTyphoonTaskDeps", LowerTyphoonTaskDeps);
}

}  // namespace transform

}  // namespace tirx
}  // namespace tvm
