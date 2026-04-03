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
 * \file tirx/op/typhoon.cc
 * \brief Typhoon graph-building ops.
 */
#include <tvm/tirx/op.h>
#include <tvm/tirx/op_attr_types.h>

namespace tvm {
namespace tirx {

#define TVM_TIR_REGISTER_TYPHOON_OP(OpName)                                               \
  TVM_TIR_REGISTER_OP("typhoon." OpName)                                                   \
      .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque))

TVM_TIR_REGISTER_TYPHOON_OP("region_decl").set_num_inputs(6);
TVM_TIR_REGISTER_TYPHOON_OP("task_dma");
TVM_TIR_REGISTER_TYPHOON_OP("task_matmul");
TVM_TIR_REGISTER_TYPHOON_OP("task_vector");
TVM_TIR_REGISTER_TYPHOON_OP("task_reshape");
TVM_TIR_REGISTER_TYPHOON_OP("submit_graph").set_num_inputs(1);
TVM_TIR_REGISTER_TYPHOON_OP("wait_graph").set_num_inputs(1);

#undef TVM_TIR_REGISTER_TYPHOON_OP

}  // namespace tirx
}  // namespace tvm
