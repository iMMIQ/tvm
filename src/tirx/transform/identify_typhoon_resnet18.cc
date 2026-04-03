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
 * \file identify_typhoon_resnet18.cc
 * \brief Identify fixed-shape ResNet18 stem patterns for Typhoon lowering.
 */
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/attrs.h>
#include <tvm/target/target.h>
#include <tvm/tirx/buffer.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/function.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <sstream>
#include <set>
#include <string>
#include <vector>

namespace tvm {
namespace tirx {

namespace {

std::vector<int64_t> ExpectConstantShape(const Buffer& buffer, const char* name) {
  std::vector<int64_t> shape;
  shape.reserve(buffer->shape.size());
  for (const PrimExpr& dim : buffer->shape) {
    const auto* imm = dim.as<IntImmNode>();
    TVM_FFI_CHECK(imm != nullptr, ValueError)
        << "IdentifyTyphoonResNet18 requires constant " << name << " buffer shape";
    shape.push_back(imm->value);
  }
  return shape;
}

bool IsShape(const std::vector<int64_t>& actual, std::initializer_list<int64_t> expected) {
  if (actual.size() != expected.size()) {
    return false;
  }
  size_t index = 0;
  for (int64_t value : expected) {
    if (actual[index++] != value) {
      return false;
    }
  }
  return true;
}

class LoadSourceVerifier : public ExprVisitor {
 public:
  LoadSourceVerifier(Buffer input, Buffer weight)
      : input_(std::move(input)), weight_(std::move(weight)) {}

  bool saw_input() const { return saw_input_; }
  bool saw_weight() const { return saw_weight_; }

 private:
  void VisitExpr_(const BufferLoadNode* op) final {
    if (op->buffer.same_as(input_)) {
      saw_input_ = true;
    }
    if (op->buffer.same_as(weight_)) {
      saw_weight_ = true;
    }
    ExprVisitor::VisitExpr_(op);
  }

  Buffer input_;
  Buffer weight_;
  bool saw_input_{false};
  bool saw_weight_{false};
};

class StemPatternVerifier : public StmtExprVisitor {
 public:
  StemPatternVerifier(Buffer input, Buffer weight, Buffer output)
      : input_(std::move(input)), weight_(std::move(weight)), output_(std::move(output)) {}

  bool matched() const {
    return matched_store_ && loop_extents_.count(64) >= 1 && loop_extents_.count(112) >= 2;
  }

 private:
  void VisitStmt_(const ForNode* op) final {
    if (const auto* imm = op->extent.as<IntImmNode>()) {
      loop_extents_.insert(imm->value);
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const BufferStoreNode* op) final {
    if (op->buffer.same_as(output_)) {
      LoadSourceVerifier verifier(input_, weight_);
      verifier(op->value);
      if (verifier.saw_input() && verifier.saw_weight()) {
        matched_store_ = true;
      }
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  Buffer input_;
  Buffer weight_;
  Buffer output_;
  bool matched_store_{false};
  std::multiset<int64_t> loop_extents_;
};

bool IsFixedShapeResNet18Stem(const PrimFunc& func) {
  Buffer input;
  Buffer weight;
  Buffer output;
  for (const auto& [param, buffer] : func->buffer_map) {
    if (buffer->dtype != DataType::Float(32)) {
      continue;
    }

    std::vector<int64_t> shape = ExpectConstantShape(buffer, "candidate");
    if (IsShape(shape, {1, 3, 224, 224})) {
      TVM_FFI_CHECK(!input.defined(), ValueError)
          << "IdentifyTyphoonResNet18 found multiple input-shaped buffers";
      input = buffer;
    } else if (IsShape(shape, {64, 3, 7, 7})) {
      TVM_FFI_CHECK(!weight.defined(), ValueError)
          << "IdentifyTyphoonResNet18 found multiple weight-shaped buffers";
      weight = buffer;
    } else if (IsShape(shape, {1, 64, 112, 112})) {
      TVM_FFI_CHECK(!output.defined(), ValueError)
          << "IdentifyTyphoonResNet18 found multiple output-shaped buffers";
      output = buffer;
    }
  }

  if (!input.defined() || !weight.defined() || !output.defined()) {
    return false;
  }

  StemPatternVerifier verifier(input, weight, output);
  verifier(func->body);
  return verifier.matched();
}

ffi::String BuildResNet18PlanJSON() {
  std::ostringstream os;
  os << "{"
     << "\"model\":\"resnet18\","
     << "\"recognized_scope\":\"stem\","
     << "\"input_shape\":[1,3,224,224],"
     << "\"dtype\":\"float32\","
     << "\"blocks\":[{"
     << "\"stage_id\":0,"
     << "\"block_id\":0,"
     << "\"layer_ids\":[0]"
     << "}],"
     << "\"layers\":[{"
     << "\"layer_id\":0,"
     << "\"stage_id\":0,"
     << "\"block_id\":0,"
     << "\"kind\":\"conv2d\","
     << "\"op_name\":\"stem_conv\","
     << "\"logical_input_shape\":[1,3,224,224],"
     << "\"logical_output_shape\":[1,64,112,112],"
     << "\"requires_im2col\":true,"
     << "\"preferred_output_layout\":\"zZ\""
     << "}]"
     << "}";
  return ffi::String(os.str());
}

}  // namespace

namespace transform {

Pass IdentifyTyphoonResNet18() {
  auto pass_func = [](IRModule mod, PassContext ctx) {
    std::vector<PrimFunc> typhoon_funcs;
    for (const auto& [gvar, base_func] : mod->functions) {
      const auto* func = base_func.as<PrimFuncNode>();
      if (func == nullptr) {
        continue;
      }

      auto target = func->GetAttr<Target>(tvm::attr::kTarget);
      if (!target.defined() || target.value()->kind->name != "typhoon") {
        continue;
      }
      typhoon_funcs.push_back(ffi::GetRef<PrimFunc>(func));
    }

    if (typhoon_funcs.empty()) {
      return mod;
    }

    TVM_FFI_CHECK_EQ(typhoon_funcs.size(), 1U, ValueError)
        << "IdentifyTyphoonResNet18 currently requires a single typhoon PrimFunc";

    if (IsFixedShapeResNet18Stem(typhoon_funcs.front())) {
      return WithAttr(std::move(mod), "typhoon_resnet18_plan", BuildResNet18PlanJSON());
    }

    if (!typhoon_funcs.empty()) {
      TVM_FFI_THROW(ValueError)
          << "IdentifyTyphoonResNet18 only supports fixed-shape ResNet18 stem graphs "
          << "(1x3x224x224 -> 1x64x112x112, float32)";
    }
    return mod;
  };
  return CreateModulePass(pass_func, 0, "tirx.IdentifyTyphoonResNet18", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tirx.transform.IdentifyTyphoonResNet18", IdentifyTyphoonResNet18);
}

}  // namespace transform

}  // namespace tirx
}  // namespace tvm
