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
 * \file intrin_rule_typhoon.cc
 * \brief Typhoon intrinsic lowering rules.
 */
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/op_attr_types.h>

namespace tvm {
namespace codegen {
namespace intrin {
using tirx::Call;
using tirx::FLowerIntrinsic;

inline FLowerIntrinsic LowerToTyphoonExternCall(const char* symbol_name) {
  return [symbol_name](const PrimExpr& e) {
    const auto* call = e.as<tirx::CallNode>();
    TVM_FFI_ICHECK(call != nullptr);
    ffi::Array<PrimExpr> args{tirx::StringImm(symbol_name)};
    for (const auto& arg : call->args) {
      args.push_back(arg);
    }
    return Call(call->dtype, tirx::builtin::call_extern(), args);
  };
}

TVM_REGISTER_OP("tirx.typhoon.region_decl")
    .set_attr<FLowerIntrinsic>("typhoon.FLowerIntrinsic",
                               LowerToTyphoonExternCall("TVMTyphoonDeclareRegion"));

TVM_REGISTER_OP("tirx.typhoon.task_dma")
    .set_attr<FLowerIntrinsic>("typhoon.FLowerIntrinsic",
                               LowerToTyphoonExternCall("TVMTyphoonAddDMATask"));

TVM_REGISTER_OP("tirx.typhoon.task_batched_dma")
    .set_attr<FLowerIntrinsic>("typhoon.FLowerIntrinsic",
                               LowerToTyphoonExternCall("TVMTyphoonAddBatchedDMATask"));

TVM_REGISTER_OP("tirx.typhoon.task_matmul")
    .set_attr<FLowerIntrinsic>("typhoon.FLowerIntrinsic",
                               LowerToTyphoonExternCall("TVMTyphoonAddMatmulTask"));

TVM_REGISTER_OP("tirx.typhoon.task_vector")
    .set_attr<FLowerIntrinsic>("typhoon.FLowerIntrinsic",
                               LowerToTyphoonExternCall("TVMTyphoonAddVectorTask"));

TVM_REGISTER_OP("tirx.typhoon.task_reshape")
    .set_attr<FLowerIntrinsic>("typhoon.FLowerIntrinsic",
                               LowerToTyphoonExternCall("TVMTyphoonAddReshapeTask"));

TVM_REGISTER_OP("tirx.typhoon.submit_graph")
    .set_attr<FLowerIntrinsic>("typhoon.FLowerIntrinsic",
                               LowerToTyphoonExternCall("TVMTyphoonSubmitGraph"));

TVM_REGISTER_OP("tirx.typhoon.wait_graph")
    .set_attr<FLowerIntrinsic>("typhoon.FLowerIntrinsic",
                               LowerToTyphoonExternCall("TVMTyphoonWaitGraph"));

}  // namespace intrin
}  // namespace codegen
}  // namespace tvm
