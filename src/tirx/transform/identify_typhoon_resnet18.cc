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
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <algorithm>
#include <cctype>
#include <map>
#include <sstream>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace tvm {
namespace tirx {

namespace {

class ExistingTyphoonGraphDetector : public StmtExprVisitor {
 public:
  bool found_graph_ops() const { return found_graph_ops_; }

 private:
  using Parent = StmtExprVisitor;

  void VisitExpr_(const CallNode* op) final {
    if (op->op.same_as(region_decl_op_) || op->op.same_as(task_dma_op_) ||
        op->op.same_as(task_matmul_op_) || op->op.same_as(task_vector_op_) ||
        op->op.same_as(task_reshape_op_) || op->op.same_as(submit_graph_op_) ||
        op->op.same_as(wait_graph_op_)) {
      found_graph_ops_ = true;
    }
    Parent::VisitExpr_(op);
  }

  bool found_graph_ops_{false};
  const Op& region_decl_op_ = Op::Get("tirx.typhoon.region_decl");
  const Op& task_dma_op_ = Op::Get("tirx.typhoon.task_dma");
  const Op& task_matmul_op_ = Op::Get("tirx.typhoon.task_matmul");
  const Op& task_vector_op_ = Op::Get("tirx.typhoon.task_vector");
  const Op& task_reshape_op_ = Op::Get("tirx.typhoon.task_reshape");
  const Op& submit_graph_op_ = Op::Get("tirx.typhoon.submit_graph");
  const Op& wait_graph_op_ = Op::Get("tirx.typhoon.wait_graph");
};

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

std::string ToLower(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return value;
}

bool ContainsExplicitBatchNorm(const std::string& symbol) {
  std::string normalized = ToLower(symbol);
  return normalized.find("batch_norm") != std::string::npos ||
         normalized.find("batchnormalization") != std::string::npos ||
         normalized.find("batchnorm") != std::string::npos;
}

std::string GetGlobalSymbol(const GlobalVar& gvar, const PrimFunc& func) {
  auto global_symbol = func->GetAttr<ffi::String>("global_symbol");
  if (global_symbol.has_value()) {
    return global_symbol.value();
  }
  return gvar->name_hint;
}

std::vector<std::vector<int64_t>> GetOrderedBufferShapes(const PrimFunc& func) {
  std::vector<std::vector<int64_t>> shapes;
  shapes.reserve(func->params.size());
  for (const Var& param : func->params) {
    auto it = func->buffer_map.find(param);
    TVM_FFI_CHECK(it != func->buffer_map.end(), ValueError)
        << "IdentifyTyphoonResNet18 requires PrimFunc params to be bound to buffers";
    const Buffer& buffer = (*it).second;
    TVM_FFI_CHECK(buffer->dtype == DataType::Float(32), ValueError)
        << "IdentifyTyphoonResNet18 only supports float32 canonical ResNet18 buffers";
    shapes.push_back(ExpectConstantShape(buffer, buffer->name.c_str()));
  }
  return shapes;
}

struct NamedPrimFunc {
  std::string symbol;
  PrimFunc func;
};

struct ExpectedFuncSpec {
  const char* symbol;
  std::vector<std::vector<int64_t>> buffer_shapes;
};

const std::vector<ExpectedFuncSpec>& GetCanonicalFunctionSpecs() {
  static const std::vector<ExpectedFuncSpec> specs = {
      {"conv2d", {{1, 3, 224, 224}, {64, 3, 7, 7}, {1, 64, 112, 112}}},
      {"reshape", {{64}, {1, 64, 1, 1}}},
      {"add", {{1, 64, 112, 112}, {1, 64, 1, 1}, {1, 64, 112, 112}}},
      {"relu", {{1, 64, 112, 112}, {1, 64, 112, 112}}},
      {"max_pool2d", {{1, 64, 112, 112}, {1, 64, 56, 56}}},
      {"conv2d1", {{1, 64, 56, 56}, {64, 64, 3, 3}, {1, 64, 56, 56}}},
      {"add1", {{1, 64, 56, 56}, {1, 64, 1, 1}, {1, 64, 56, 56}}},
      {"relu1", {{1, 64, 56, 56}, {1, 64, 56, 56}}},
      {"add2", {{1, 64, 56, 56}, {1, 64, 56, 56}, {1, 64, 56, 56}}},
      {"conv2d2", {{1, 64, 56, 56}, {128, 64, 3, 3}, {1, 128, 28, 28}}},
      {"reshape1", {{128}, {1, 128, 1, 1}}},
      {"add3", {{1, 128, 28, 28}, {1, 128, 1, 1}, {1, 128, 28, 28}}},
      {"relu2", {{1, 128, 28, 28}, {1, 128, 28, 28}}},
      {"conv2d3", {{1, 128, 28, 28}, {128, 128, 3, 3}, {1, 128, 28, 28}}},
      {"conv2d4", {{1, 64, 56, 56}, {128, 64, 1, 1}, {1, 128, 28, 28}}},
      {"add4", {{1, 128, 28, 28}, {1, 128, 28, 28}, {1, 128, 28, 28}}},
      {"conv2d5", {{1, 128, 28, 28}, {256, 128, 3, 3}, {1, 256, 14, 14}}},
      {"reshape2", {{256}, {1, 256, 1, 1}}},
      {"add5", {{1, 256, 14, 14}, {1, 256, 1, 1}, {1, 256, 14, 14}}},
      {"relu3", {{1, 256, 14, 14}, {1, 256, 14, 14}}},
      {"conv2d6", {{1, 256, 14, 14}, {256, 256, 3, 3}, {1, 256, 14, 14}}},
      {"conv2d7", {{1, 128, 28, 28}, {256, 128, 1, 1}, {1, 256, 14, 14}}},
      {"add6", {{1, 256, 14, 14}, {1, 256, 14, 14}, {1, 256, 14, 14}}},
      {"conv2d8", {{1, 256, 14, 14}, {512, 256, 3, 3}, {1, 512, 7, 7}}},
      {"reshape3", {{512}, {1, 512, 1, 1}}},
      {"add7", {{1, 512, 7, 7}, {1, 512, 1, 1}, {1, 512, 7, 7}}},
      {"relu4", {{1, 512, 7, 7}, {1, 512, 7, 7}}},
      {"conv2d9", {{1, 512, 7, 7}, {512, 512, 3, 3}, {1, 512, 7, 7}}},
      {"conv2d10", {{1, 256, 14, 14}, {512, 256, 1, 1}, {1, 512, 7, 7}}},
      {"add8", {{1, 512, 7, 7}, {1, 512, 7, 7}, {1, 512, 7, 7}}},
      {"mean", {{1, 512, 7, 7}, {1, 512, 1, 1}}},
      {"reshape4", {{1, 512, 1, 1}, {1, 512}}},
      {"transpose", {{1000, 512}, {512, 1000}}},
      {"matmul", {{1, 512}, {512, 1000}, {1, 1000}}},
      {"add9", {{1, 1000}, {1000}, {1, 1000}}},
  };
  return specs;
}

bool MatchesExpectedBufferShapes(const PrimFunc& func,
                                 const std::vector<std::vector<int64_t>>& expected_shapes) {
  return GetOrderedBufferShapes(func) == expected_shapes;
}

bool IsCanonicalResNet18FullGraph(const std::vector<NamedPrimFunc>& typhoon_funcs) {
  const auto& specs = GetCanonicalFunctionSpecs();
  std::set<std::string> expected_symbols;
  for (const ExpectedFuncSpec& spec : specs) {
    expected_symbols.insert(spec.symbol);
  }
  std::map<std::string, PrimFunc> funcs_by_symbol;
  for (const NamedPrimFunc& entry : typhoon_funcs) {
    if (ContainsExplicitBatchNorm(entry.symbol)) {
      TVM_FFI_THROW(ValueError)
          << "IdentifyTyphoonResNet18 does not support explicit BatchNormalization; "
          << "expected canonical batch-norm-folded ResNet18";
    }
    if (!funcs_by_symbol.emplace(entry.symbol, entry.func).second) {
      TVM_FFI_THROW(ValueError)
          << "IdentifyTyphoonResNet18 requires unique canonical operator symbols, found duplicate `"
          << entry.symbol << "`";
    }
    TVM_FFI_CHECK(expected_symbols.count(entry.symbol) != 0, ValueError)
        << "IdentifyTyphoonResNet18 requires the canonical ResNet18 operator family; found `"
        << entry.symbol << "`";
  }

  if (funcs_by_symbol.size() != specs.size()) {
    return false;
  }

  for (const ExpectedFuncSpec& spec : specs) {
    auto it = funcs_by_symbol.find(spec.symbol);
    if (it == funcs_by_symbol.end()) {
      return false;
    }
    if (!MatchesExpectedBufferShapes(it->second, spec.buffer_shapes)) {
      return false;
    }
  }
  return true;
}

struct StagePlanSpec {
  int stage_id;
  const char* name;
  std::vector<int> block_ids;
};

struct BlockPlanSpec {
  int block_id;
  int stage_id;
  std::vector<int> layer_ids;
};

struct LayerPlanSpec {
  int layer_id;
  int stage_id;
  int block_id;
  const char* kind;
  const char* op_name;
  std::vector<int64_t> logical_input_shape;
  std::vector<int64_t> logical_output_shape;
  std::vector<int64_t> weight_shape;
  bool has_weight_shape{false};
  bool requires_im2col{false};
  bool has_layout_transform{false};
  const char* input_layout{nullptr};
  const char* output_layout{nullptr};
};

struct EdgePlanSpec {
  int src_layer_id;
  int dst_layer_id;
  const char* edge_kind;
};

void AppendIntArray(std::ostringstream& os, const std::vector<int64_t>& values) {
  os << "[";
  for (size_t i = 0; i < values.size(); ++i) {
    if (i != 0) {
      os << ",";
    }
    os << values[i];
  }
  os << "]";
}

void AppendIntArray(std::ostringstream& os, const std::vector<int>& values) {
  os << "[";
  for (size_t i = 0; i < values.size(); ++i) {
    if (i != 0) {
      os << ",";
    }
    os << values[i];
  }
  os << "]";
}

ffi::String BuildResNet18PlanJSON() {
  static const std::vector<StagePlanSpec> stages = {
      {0, "stage0", {0}}, {1, "stage1", {1}}, {2, "stage2", {2}}, {3, "stage3", {3}}};
  static const std::vector<BlockPlanSpec> blocks = {
      {0, 0, {0, 1, 2, 3, 4, 5, 6, 7}},
      {1, 1, {8, 9, 10, 11, 12, 13}},
      {2, 2, {14, 15, 16, 17, 18, 19}},
      {3, 3, {20, 21, 22, 23, 24, 25, 26, 27, 28}},
  };
  static const std::vector<LayerPlanSpec> layers = {
      {0, 0, 0, "conv2d", "stem_conv", {1, 3, 224, 224}, {1, 64, 112, 112}, {64, 3, 7, 7},
       true, true, true, "NCHW", "zZ"},
      {1, 0, 0, "bias_add", "stem_bias_add", {1, 64, 112, 112}, {1, 64, 112, 112}},
      {2, 0, 0, "relu", "stem_relu", {1, 64, 112, 112}, {1, 64, 112, 112}},
      {3, 0, 0, "max_pool2d", "stem_max_pool", {1, 64, 112, 112}, {1, 64, 56, 56}},
      {4, 0, 0, "conv2d", "stage0_conv", {1, 64, 56, 56}, {1, 64, 56, 56}, {64, 64, 3, 3},
       true, true, true, "NCHW", "zZ"},
      {5, 0, 0, "bias_add", "stage0_bias_add", {1, 64, 56, 56}, {1, 64, 56, 56}},
      {6, 0, 0, "relu", "stage0_relu", {1, 64, 56, 56}, {1, 64, 56, 56}},
      {7, 0, 0, "residual_add", "stage0_residual_add", {1, 64, 56, 56}, {1, 64, 56, 56}},
      {8, 1, 1, "conv2d", "stage1_conv0", {1, 64, 56, 56}, {1, 128, 28, 28}, {128, 64, 3, 3},
       true, true, true, "NCHW", "zZ"},
      {9, 1, 1, "bias_add", "stage1_bias_add", {1, 128, 28, 28}, {1, 128, 28, 28}},
      {10, 1, 1, "relu", "stage1_relu", {1, 128, 28, 28}, {1, 128, 28, 28}},
      {11, 1, 1, "conv2d", "stage1_conv1", {1, 128, 28, 28}, {1, 128, 28, 28},
       {128, 128, 3, 3}, true, true, true, "NCHW", "zZ"},
      {12, 1, 1, "conv2d", "stage1_skip_conv", {1, 64, 56, 56}, {1, 128, 28, 28},
       {128, 64, 1, 1}, true, false, true, "NCHW", "zZ"},
      {13, 1, 1, "residual_add", "stage1_residual_add", {1, 128, 28, 28}, {1, 128, 28, 28}},
      {14, 2, 2, "conv2d", "stage2_conv0", {1, 128, 28, 28}, {1, 256, 14, 14},
       {256, 128, 3, 3}, true, true, true, "NCHW", "zZ"},
      {15, 2, 2, "bias_add", "stage2_bias_add", {1, 256, 14, 14}, {1, 256, 14, 14}},
      {16, 2, 2, "relu", "stage2_relu", {1, 256, 14, 14}, {1, 256, 14, 14}},
      {17, 2, 2, "conv2d", "stage2_conv1", {1, 256, 14, 14}, {1, 256, 14, 14},
       {256, 256, 3, 3}, true, true, true, "NCHW", "zZ"},
      {18, 2, 2, "conv2d", "stage2_skip_conv", {1, 128, 28, 28}, {1, 256, 14, 14},
       {256, 128, 1, 1}, true, false, true, "NCHW", "zZ"},
      {19, 2, 2, "residual_add", "stage2_residual_add", {1, 256, 14, 14}, {1, 256, 14, 14}},
      {20, 3, 3, "conv2d", "stage3_conv0", {1, 256, 14, 14}, {1, 512, 7, 7},
       {512, 256, 3, 3}, true, true, true, "NCHW", "zZ"},
      {21, 3, 3, "bias_add", "stage3_bias_add", {1, 512, 7, 7}, {1, 512, 7, 7}},
      {22, 3, 3, "relu", "stage3_relu", {1, 512, 7, 7}, {1, 512, 7, 7}},
      {23, 3, 3, "conv2d", "stage3_conv1", {1, 512, 7, 7}, {1, 512, 7, 7},
       {512, 512, 3, 3}, true, true, true, "NCHW", "zZ"},
      {24, 3, 3, "conv2d", "stage3_skip_conv", {1, 256, 14, 14}, {1, 512, 7, 7},
       {512, 256, 1, 1}, true, false, true, "NCHW", "zZ"},
      {25, 3, 3, "residual_add", "stage3_residual_add", {1, 512, 7, 7}, {1, 512, 7, 7}},
      {26, 3, 3, "global_avg_pool", "global_avg_pool", {1, 512, 7, 7}, {1, 512, 1, 1}},
      {27, 3, 3, "flatten", "flatten", {1, 512, 1, 1}, {1, 512}},
      {28, 3, 3, "dense", "dense", {1, 512}, {1, 1000}, {1000, 512}, true, false, true, "NC",
       "zZ"},
  };
  static const std::vector<EdgePlanSpec> edges = {
      {0, 1, "data"},  {1, 2, "data"},     {2, 3, "data"},      {3, 4, "data"},
      {4, 5, "data"},  {5, 6, "data"},     {6, 7, "data"},      {3, 7, "residual"},
      {7, 8, "data"},  {8, 9, "data"},     {9, 10, "data"},     {10, 11, "data"},
      {7, 12, "residual_projection"},      {11, 13, "data"},    {12, 13, "residual"},
      {13, 14, "data"},                    {14, 15, "data"},    {15, 16, "data"},
      {16, 17, "data"},                    {13, 18, "residual_projection"},
      {17, 19, "data"},                    {18, 19, "residual"}, {19, 20, "data"},
      {20, 21, "data"},                    {21, 22, "data"},    {22, 23, "data"},
      {19, 24, "residual_projection"},     {23, 25, "data"},    {24, 25, "residual"},
      {25, 26, "data"},                    {26, 27, "data"},    {27, 28, "data"},
  };

  std::ostringstream os;
  os << "{"
     << "\"model\":\"resnet18\","
     << "\"recognized_scope\":\"full_graph\","
     << "\"input_shape\":[1,3,224,224],"
     << "\"dtype\":\"float32\",";

  os << "\"stages\":[";
  for (size_t i = 0; i < stages.size(); ++i) {
    if (i != 0) {
      os << ",";
    }
    const StagePlanSpec& stage = stages[i];
    os << "{"
       << "\"stage_id\":" << stage.stage_id << ","
       << "\"name\":\"" << stage.name << "\","
       << "\"block_ids\":";
    AppendIntArray(os, stage.block_ids);
    os << "}";
  }
  os << "],";

  os << "\"blocks\":[";
  for (size_t i = 0; i < blocks.size(); ++i) {
    if (i != 0) {
      os << ",";
    }
    const BlockPlanSpec& block = blocks[i];
    os << "{"
       << "\"block_id\":" << block.block_id << ","
       << "\"stage_id\":" << block.stage_id << ","
       << "\"layer_ids\":";
    AppendIntArray(os, block.layer_ids);
    os << "}";
  }
  os << "],";

  os << "\"layers\":[";
  for (size_t i = 0; i < layers.size(); ++i) {
    if (i != 0) {
      os << ",";
    }
    const LayerPlanSpec& layer = layers[i];
    os << "{"
       << "\"layer_id\":" << layer.layer_id << ","
       << "\"stage_id\":" << layer.stage_id << ","
       << "\"block_id\":" << layer.block_id << ","
       << "\"kind\":\"" << layer.kind << "\","
       << "\"op_name\":\"" << layer.op_name << "\","
       << "\"logical_input_shape\":";
    AppendIntArray(os, layer.logical_input_shape);
    os << ",\"logical_output_shape\":";
    AppendIntArray(os, layer.logical_output_shape);
    if (layer.has_weight_shape) {
      os << ",\"weight_shape\":";
      AppendIntArray(os, layer.weight_shape);
    }
    if (layer.kind == std::string("conv2d")) {
      os << ",\"requires_im2col\":" << (layer.requires_im2col ? "true" : "false");
    }
    if (layer.has_layout_transform) {
      os << ",\"layout_transform\":{"
         << "\"input_layout\":\"" << layer.input_layout << "\","
         << "\"output_layout\":\"" << layer.output_layout << "\""
         << "}";
    }
    os << "}";
  }
  os << "],";

  os << "\"edges\":[";
  for (size_t i = 0; i < edges.size(); ++i) {
    if (i != 0) {
      os << ",";
    }
    const EdgePlanSpec& edge = edges[i];
    os << "{"
       << "\"src_layer_id\":" << edge.src_layer_id << ","
       << "\"dst_layer_id\":" << edge.dst_layer_id << ","
       << "\"edge_kind\":\"" << edge.edge_kind << "\""
       << "}";
  }
  os << "]"
     << "}";
  return ffi::String(os.str());
}

class LoadSourceVerifier : public ExprVisitor {
 public:
  LoadSourceVerifier(Buffer primary_input, Buffer secondary_input, Buffer weight)
      : primary_input_(std::move(primary_input)),
        secondary_input_(std::move(secondary_input)),
        weight_(std::move(weight)) {}

  bool saw_primary_input() const { return saw_primary_input_; }
  bool saw_secondary_input() const { return saw_secondary_input_; }
  bool saw_weight() const { return saw_weight_; }

 private:
  void VisitExpr_(const BufferLoadNode* op) final {
    if (op->buffer.same_as(primary_input_)) {
      saw_primary_input_ = true;
    }
    if (secondary_input_.defined() && op->buffer.same_as(secondary_input_)) {
      saw_secondary_input_ = true;
    }
    if (op->buffer.same_as(weight_)) {
      saw_weight_ = true;
    }
    ExprVisitor::VisitExpr_(op);
  }

  Buffer primary_input_;
  Buffer secondary_input_;
  Buffer weight_;
  bool saw_primary_input_{false};
  bool saw_secondary_input_{false};
  bool saw_weight_{false};
};

}  // namespace

namespace transform {

Pass IdentifyTyphoonResNet18() {
  auto pass_func = [](IRModule mod, PassContext ctx) {
    std::vector<NamedPrimFunc> typhoon_funcs;
    for (const auto& [gvar, base_func] : mod->functions) {
      const auto* func = base_func.as<PrimFuncNode>();
      if (func == nullptr) {
        continue;
      }

      auto target = func->GetAttr<Target>(tvm::attr::kTarget);
      if (!target.defined() || target.value()->kind->name != "typhoon") {
        continue;
      }
      typhoon_funcs.push_back(
          NamedPrimFunc{GetGlobalSymbol(gvar, ffi::GetRef<PrimFunc>(func)), ffi::GetRef<PrimFunc>(func)});
    }

    if (typhoon_funcs.empty()) {
      return mod;
    }

    for (const NamedPrimFunc& entry : typhoon_funcs) {
      ExistingTyphoonGraphDetector detector;
      detector(entry.func->body);
      if (detector.found_graph_ops()) {
        return mod;
      }
    }

    if (IsCanonicalResNet18FullGraph(typhoon_funcs)) {
      return WithAttr(std::move(mod), "typhoon_resnet18_plan", BuildResNet18PlanJSON());
    }

    TVM_FFI_THROW(ValueError)
        << "IdentifyTyphoonResNet18 only supports the canonical fixed-shape full ResNet18 graph "
        << "(batch-norm-folded float32 1x3x224x224)";
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
