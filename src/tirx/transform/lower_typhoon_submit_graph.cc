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
 * \file lower_typhoon_submit_graph.cc
 * \brief Expand Typhoon submit_graph nodes into explicit graph-begin statements.
 */
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/target/target.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/function.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

namespace tvm {
namespace tirx {

class TyphoonSubmitGraphLowerer : public StmtExprMutator {
 public:
  using Parent = StmtExprMutator;

  std::optional<PrimExpr> graph_id() const { return graph_id_; }

  Stmt VisitStmt_(const EvaluateNode* op) final {
    Stmt stmt = Parent::VisitStmt_(op);
    const auto* eval = stmt.as<EvaluateNode>();
    if (eval == nullptr) {
      return stmt;
    }
    const auto* call = eval->value.as<CallNode>();
    if (call == nullptr || !call->op.same_as(submit_graph_op_)) {
      return stmt;
    }
    TVM_FFI_ICHECK_EQ(call->args.size(), 1U) << "tirx.typhoon.submit_graph expects graph_id only";
    if (!graph_id_.has_value()) {
      graph_id_ = call->args[0];
    }

    ffi::Array<PrimExpr> submit_args{StringImm("TVMTyphoonSubmitGraph"), call->args[0]};
    PrimExpr submit_call = Call(eval->value.dtype(), builtin::call_extern(), submit_args);
    return Evaluate(submit_call, op->span);
  }

 private:
  std::optional<PrimExpr> graph_id_;
  const Op& submit_graph_op_ = Op::Get("tirx.typhoon.submit_graph");
};

namespace transform {

Pass LowerTyphoonSubmitGraph() {
  auto pass_func = [](PrimFunc f, IRModule m, PassContext ctx) {
    auto target = f->GetAttr<Target>(tvm::attr::kTarget);
    if (!target.defined() || target.value()->kind->name != "typhoon") {
      return f;
    }
    TyphoonSubmitGraphLowerer lowerer;
    auto* n = f.CopyOnWrite();
    n->body = lowerer(std::move(n->body));
    if (lowerer.graph_id().has_value()) {
      ffi::Array<PrimExpr> begin_args{StringImm("TVMTyphoonGraphBegin"), lowerer.graph_id().value()};
      Stmt begin_stmt = Evaluate(Call(DataType::Int(32), builtin::call_extern(), begin_args));
      if (const auto* seq = n->body.as<SeqStmtNode>()) {
        ffi::Array<Stmt> stmts;
        stmts.push_back(begin_stmt);
        for (const Stmt& stmt : seq->seq) {
          stmts.push_back(stmt);
        }
        n->body = SeqStmt(stmts, seq->span);
      } else {
        n->body = SeqStmt({begin_stmt, n->body});
      }
    }
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tirx.LowerTyphoonSubmitGraph", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tirx.transform.LowerTyphoonSubmitGraph", LowerTyphoonSubmitGraph);
}

}  // namespace transform

}  // namespace tirx
}  // namespace tvm
