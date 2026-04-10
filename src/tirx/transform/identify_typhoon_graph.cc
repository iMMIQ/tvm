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
 * \file identify_typhoon_graph.cc
 * \brief Identify general standalone Typhoon-compatible fused PrimFuncs.
 */

#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ffi/string.h>
#include <tvm/target/target.h>
#include <tvm/tirx/buffer.h>
#include <tvm/tirx/function.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <algorithm>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace tvm {
namespace tirx {

namespace {

struct TyphoonSemanticNode {
  int64_t graph_id{0};
  int64_t node_id{0};
  std::string symbol;
  std::string op_family;
  std::vector<std::string> input_roles;
  std::vector<std::string> post_ops;
  std::vector<std::vector<int64_t>> param_shapes;
};

std::vector<int64_t> ExpectConstantShape(const Buffer& buffer, const char* name) {
  std::vector<int64_t> shape;
  shape.reserve(buffer->shape.size());
  for (const PrimExpr& dim : buffer->shape) {
    const auto* imm = dim.as<IntImmNode>();
    TVM_FFI_CHECK(imm != nullptr, ValueError)
        << "IdentifyTyphoonGraph requires constant " << name << " buffer shape";
    shape.push_back(imm->value);
  }
  return shape;
}

bool IsTyphoonPrimFunc(const BaseFunc& base_func) {
  const auto* func = base_func.as<PrimFuncNode>();
  if (func == nullptr) {
    return false;
  }
  auto target = func->GetAttr<Target>(tvm::attr::kTarget);
  return target.defined() && target.value()->kind->name == "typhoon";
}

std::string GetGlobalSymbol(const GlobalVar& gvar, const PrimFunc& func) {
  auto global_symbol = func->GetAttr<ffi::String>("global_symbol");
  if (global_symbol.has_value()) {
    return global_symbol.value();
  }
  return gvar->name_hint;
}

class SBlockNameCollector : public StmtVisitor {
 public:
  std::vector<std::string> block_names;

 private:
  void VisitStmt_(const SBlockNode* op) final {
    block_names.push_back(op->name_hint);
    StmtVisitor::VisitStmt_(op);
  }
};

bool HasBlock(const std::vector<std::string>& names, const std::string& needle) {
  return std::find(names.begin(), names.end(), needle) != names.end();
}

int CountBlocksWithPrefix(const std::vector<std::string>& names, const std::string& prefix) {
  return std::count_if(names.begin(), names.end(), [&](const std::string& name) {
    return name.rfind(prefix, 0) == 0;
  });
}

bool StartsWith(const std::string& value, const std::string& prefix) {
  return value.rfind(prefix, 0) == 0;
}

std::optional<TyphoonSemanticNode> TryClassifyTyphoonFunc(const GlobalVar& gvar, const PrimFunc& func,
                                                          int64_t graph_id) {
  std::string symbol = GetGlobalSymbol(gvar, func);
  if (symbol.empty()) {
    return std::nullopt;
  }

  std::vector<std::vector<int64_t>> param_shapes;
  param_shapes.reserve(func->params.size());
  for (const Var& param : func->params) {
    auto it = func->buffer_map.find(param);
    if (it == func->buffer_map.end()) {
      return std::nullopt;
    }
    const Buffer& buffer = (*it).second;
    if (buffer->dtype != DataType::Float(32)) {
      return std::nullopt;
    }
    param_shapes.push_back(ExpectConstantShape(buffer, buffer->name.c_str()));
  }

  SBlockNameCollector collector;
  collector(func->body);
  const std::vector<std::string>& block_names = collector.block_names;
  int add_count = CountBlocksWithPrefix(block_names, "T_add");
  bool has_conv2d = HasBlock(block_names, "conv2d_nchw");
  bool has_matmul = HasBlock(block_names, "matmul");
  bool has_relu_compute = HasBlock(block_names, "compute");

  TyphoonSemanticNode node;
  node.graph_id = graph_id;
  node.node_id = 0;
  node.symbol = symbol;
  node.param_shapes = std::move(param_shapes);

  if (has_conv2d) {
    if (func->params.size() == 4 && add_count >= 1 && has_relu_compute) {
      node.op_family = "conv_bias_relu";
      node.input_roles = {"data", "weight", "bias"};
      node.post_ops = {"relu"};
      return node;
    }
    if (func->params.size() == 4 && add_count >= 1) {
      node.op_family = "conv_bias";
      node.input_roles = {"data", "weight", "bias"};
      return node;
    }
    if (func->params.size() == 5 && add_count >= 2 && has_relu_compute) {
      node.op_family = "conv_bias_residual_relu";
      node.input_roles = {"data", "weight", "bias", "residual"};
      node.post_ops = {"relu"};
      return node;
    }
    if (func->params.size() == 5 && add_count >= 2) {
      node.op_family = "conv_bias_residual";
      node.input_roles = {"data", "weight", "bias", "residual"};
      return node;
    }
  }

  if (has_matmul && func->params.size() == 4 && add_count >= 1) {
    node.op_family = "matmul_bias";
    node.input_roles = {"data", "weight", "bias"};
    return node;
  }

  if (symbol == "max_pool2d") {
    node.op_family = "max_pool";
    node.input_roles = {"data"};
    return node;
  }
  if (symbol == "mean") {
    node.op_family = "global_avg_pool";
    node.input_roles = {"data"};
    return node;
  }
  if (symbol == "transpose") {
    node.op_family = "transpose";
    node.input_roles = {"data"};
    return node;
  }
  if (StartsWith(symbol, "reshape")) {
    node.op_family = "reshape";
    node.input_roles = {"data"};
    return node;
  }
  if (StartsWith(symbol, "copy")) {
    node.op_family = "copy";
    node.input_roles = {"data"};
    return node;
  }
  return std::nullopt;
}

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

void AppendStringArray(std::ostringstream& os, const std::vector<std::string>& values) {
  os << "[";
  for (size_t i = 0; i < values.size(); ++i) {
    if (i != 0) {
      os << ",";
    }
    os << "\"" << values[i] << "\"";
  }
  os << "]";
}

ffi::String BuildTyphoonGraphPlanJSON(const std::vector<TyphoonSemanticNode>& nodes) {
  std::ostringstream os;
  os << "{"
     << "\"version\":1,"
     << "\"target\":\"typhoon\","
     << "\"graphs\":[";
  for (size_t i = 0; i < nodes.size(); ++i) {
    if (i != 0) {
      os << ",";
    }
    const TyphoonSemanticNode& node = nodes[i];
    os << "{"
       << "\"graph_id\":" << node.graph_id << ","
       << "\"mode\":\"standalone\","
       << "\"symbol\":\"" << node.symbol << "\","
       << "\"inputs\":";
    AppendStringArray(os, node.input_roles);
    os << ",\"outputs\":[\"output\"],"
       << "\"nodes\":[{"
       << "\"node_id\":" << node.node_id << ","
       << "\"symbol\":\"" << node.symbol << "\","
       << "\"op_family\":\"" << node.op_family << "\","
       << "\"input_roles\":";
    AppendStringArray(os, node.input_roles);
    os << ",\"post_ops\":";
    AppendStringArray(os, node.post_ops);
    os << ",\"param_shapes\":[";
    for (size_t j = 0; j < node.param_shapes.size(); ++j) {
      if (j != 0) {
        os << ",";
      }
      AppendIntArray(os, node.param_shapes[j]);
    }
    os << "],\"attrs\":{}}],"
       << "\"edges\":[]}";
  }
  os << "]}";
  return ffi::String(os.str());
}

}  // namespace

namespace transform {

Pass IdentifyTyphoonGraph() {
  auto pass_func = [](IRModule mod, PassContext ctx) {
    if (mod->GetAttr<ffi::String>("typhoon_graph_plan").has_value()) {
      return mod;
    }

    std::vector<TyphoonSemanticNode> nodes;
    int64_t next_graph_id = 0;
    for (const auto& [gvar, base_func] : mod->functions) {
      if (!IsTyphoonPrimFunc(base_func)) {
        continue;
      }
      PrimFunc func = Downcast<PrimFunc>(base_func);
      auto node = TryClassifyTyphoonFunc(gvar, func, next_graph_id);
      if (!node.has_value()) {
        continue;
      }
      nodes.push_back(std::move(node.value()));
      ++next_graph_id;
    }
    if (nodes.empty()) {
      return mod;
    }
    return WithAttr(std::move(mod), "typhoon_graph_plan", BuildTyphoonGraphPlanJSON(nodes));
  };
  return CreateModulePass(pass_func, 0, "tirx.IdentifyTyphoonGraph", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tirx.transform.IdentifyTyphoonGraph", IdentifyTyphoonGraph);
}

}  // namespace transform

}  // namespace tirx
}  // namespace tvm
