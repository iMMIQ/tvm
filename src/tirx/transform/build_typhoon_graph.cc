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
 * \file build_typhoon_graph.cc
 * \brief Emit Typhoon graph IR from fixed ResNet18 and SRAM plans.
 */
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ffi/string.h>
#include <tvm/target/target.h>
#include <tvm/tirx/buffer.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/function.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/transform.h>

#include <optional>
#include <regex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace tvm {
namespace tirx {

namespace {

struct RegionSpec {
  int64_t region_id;
  std::string name;
  int64_t offset;
  int64_t size;
  int64_t alignment;
};

struct Conv2DConfig {
  std::vector<int64_t> input_shape;
  std::vector<int64_t> weight_shape;
  std::vector<int64_t> output_shape;
  int64_t in_channels{0};
  int64_t out_channels{0};
  int64_t in_h{0};
  int64_t in_w{0};
  int64_t out_h{0};
  int64_t out_w{0};
  int64_t kernel_h{0};
  int64_t kernel_w{0};
  int64_t stride_h{0};
  int64_t stride_w{0};
  int64_t pad_h{0};
  int64_t pad_w{0};
  int64_t patch_size{0};
};

struct PlannedLayer {
  int64_t layer_id{0};
  std::string kind;
  std::string symbol;
  std::vector<int64_t> logical_input_shape;
  std::vector<int64_t> logical_output_shape;
  std::vector<int64_t> weight_shape;
  std::vector<int64_t> secondary_input_shape;
  bool requires_im2col{false};
};

const std::vector<RegionSpec>& GetCanonicalRegions() {
  static const std::vector<RegionSpec> regions = {
      {0, "act0", 0, 65536, 64},
      {1, "act1", 65536, 65536, 64},
      {2, "residual", 131072, 65536, 64},
      {3, "wgt0", 196608, 589824, 64},
      {4, "col0", 786432, 147456, 64},
      {5, "aux0", 933888, 32768, 64},
  };
  return regions;
}

std::vector<int64_t> ExpectConstantShape(const Buffer& buffer, const char* name) {
  std::vector<int64_t> shape;
  shape.reserve(buffer->shape.size());
  for (const PrimExpr& dim : buffer->shape) {
    const auto* imm = dim.as<IntImmNode>();
    TVM_FFI_CHECK(imm != nullptr, ValueError)
        << "BuildTyphoonGraph requires constant " << name << " buffer shape";
    shape.push_back(imm->value);
  }
  return shape;
}

bool IsShape(const std::vector<int64_t>& actual, const std::vector<int64_t>& expected) {
  return actual == expected;
}

std::vector<int64_t> ParseIntList(const std::string& text) {
  std::vector<int64_t> values;
  std::regex int_re("-?[0-9]+");
  for (auto it = std::sregex_iterator(text.begin(), text.end(), int_re); it != std::sregex_iterator();
       ++it) {
    values.push_back(std::stoll((*it).str()));
  }
  return values;
}

std::vector<int64_t> ExtractArray(const std::string& json, const char* key) {
  std::regex pattern(std::string("\"") + key + "\"\\s*:\\s*\\[([^\\]]+)\\]");
  std::smatch match;
  TVM_FFI_CHECK(std::regex_search(json, match, pattern), ValueError)
      << "BuildTyphoonGraph could not find `" << key << "`";
  return ParseIntList(match[1].str());
}

bool TryExtractArray(const std::string& json, const char* key, std::vector<int64_t>* value) {
  std::regex pattern(std::string("\"") + key + "\"\\s*:\\s*\\[([^\\]]*)\\]");
  std::smatch match;
  if (!std::regex_search(json, match, pattern)) {
    return false;
  }
  *value = ParseIntList(match[1].str());
  return true;
}

std::string ExtractString(const std::string& json, const char* key) {
  std::regex pattern(std::string("\"") + key + "\"\\s*:\\s*\"([^\"]+)\"");
  std::smatch match;
  TVM_FFI_CHECK(std::regex_search(json, match, pattern), ValueError)
      << "BuildTyphoonGraph could not find `" << key << "`";
  return match[1].str();
}

bool TryExtractBool(const std::string& json, const char* key, bool* value) {
  std::regex pattern(std::string("\"") + key + "\"\\s*:\\s*(true|false)");
  std::smatch match;
  if (!std::regex_search(json, match, pattern)) {
    return false;
  }
  *value = match[1].str() == "true";
  return true;
}

int64_t ExtractInt(const std::string& json, const char* key) {
  std::regex pattern(std::string("\"") + key + "\"\\s*:\\s*(-?[0-9]+)");
  std::smatch match;
  TVM_FFI_CHECK(std::regex_search(json, match, pattern), ValueError)
      << "BuildTyphoonGraph could not find `" << key << "`";
  return std::stoll(match[1].str());
}

std::string ExtractJSONArrayText(const std::string& json, const char* key) {
  std::string needle = std::string("\"") + key + "\"";
  size_t key_pos = json.find(needle);
  TVM_FFI_CHECK_NE(key_pos, std::string::npos, ValueError)
      << "BuildTyphoonGraph could not find `" << key << "`";
  size_t array_start = json.find('[', key_pos);
  TVM_FFI_CHECK_NE(array_start, std::string::npos, ValueError)
      << "BuildTyphoonGraph could not find array `" << key << "`";

  int depth = 0;
  bool in_string = false;
  for (size_t i = array_start; i < json.size(); ++i) {
    char c = json[i];
    if (c == '"' && (i == 0 || json[i - 1] != '\\')) {
      in_string = !in_string;
    }
    if (in_string) {
      continue;
    }
    if (c == '[') {
      ++depth;
    } else if (c == ']') {
      --depth;
      if (depth == 0) {
        return json.substr(array_start, i - array_start + 1);
      }
    }
  }
  TVM_FFI_THROW(ValueError) << "BuildTyphoonGraph found unterminated array `" << key << "`";
}

std::vector<std::string> SplitTopLevelObjects(const std::string& array_json) {
  std::vector<std::string> objects;
  int depth = 0;
  bool in_string = false;
  size_t object_start = std::string::npos;
  for (size_t i = 0; i < array_json.size(); ++i) {
    char c = array_json[i];
    if (c == '"' && (i == 0 || array_json[i - 1] != '\\')) {
      in_string = !in_string;
    }
    if (in_string) {
      continue;
    }
    if (c == '{') {
      if (depth == 0) {
        object_start = i;
      }
      ++depth;
    } else if (c == '}') {
      --depth;
      if (depth == 0 && object_start != std::string::npos) {
        objects.push_back(array_json.substr(object_start, i - object_start + 1));
      }
    }
  }
  return objects;
}

std::vector<PlannedLayer> ParsePlannedLayers(const std::string& plan_json) {
  std::vector<PlannedLayer> layers;
  for (const std::string& object : SplitTopLevelObjects(ExtractJSONArrayText(plan_json, "layers"))) {
    PlannedLayer layer;
    layer.layer_id = ExtractInt(object, "layer_id");
    layer.kind = ExtractString(object, "kind");
    layer.symbol = ExtractString(object, "symbol");
    layer.logical_input_shape = ExtractArray(object, "logical_input_shape");
    layer.logical_output_shape = ExtractArray(object, "logical_output_shape");
    TryExtractArray(object, "weight_shape", &layer.weight_shape);
    TryExtractArray(object, "secondary_input_shape", &layer.secondary_input_shape);
    TryExtractBool(object, "requires_im2col", &layer.requires_im2col);
    layers.push_back(std::move(layer));
  }
  TVM_FFI_CHECK(!layers.empty(), ValueError) << "BuildTyphoonGraph requires planned layers";
  return layers;
}

std::vector<RegionSpec> ExtractRegions(const std::string& json) {
  std::vector<RegionSpec> regions;
  std::regex pattern(
      "\\{\\s*\"region_id\"\\s*:\\s*(-?[0-9]+)\\s*,\\s*\"name\"\\s*:\\s*\"([^\"]+)\"\\s*,\\s*"
      "\"offset\"\\s*:\\s*(-?[0-9]+)\\s*,\\s*\"size\"\\s*:\\s*(-?[0-9]+)\\s*,\\s*"
      "\"alignment\"\\s*:\\s*(-?[0-9]+)\\s*\\}");
  for (auto it = std::sregex_iterator(json.begin(), json.end(), pattern);
       it != std::sregex_iterator(); ++it) {
    regions.push_back(RegionSpec{
        std::stoll((*it)[1].str()),
        (*it)[2].str(),
        std::stoll((*it)[3].str()),
        std::stoll((*it)[4].str()),
        std::stoll((*it)[5].str()),
    });
  }
  TVM_FFI_CHECK(!regions.empty(), ValueError)
      << "BuildTyphoonGraph could not find any SRAM regions";
  return regions;
}

const RegionSpec& FindRegion(const std::vector<RegionSpec>& regions, const char* name) {
  for (const RegionSpec& region : regions) {
    if (region.name == name) {
      return region;
    }
  }
  TVM_FFI_THROW(ValueError) << "BuildTyphoonGraph could not find SRAM region `" << name << "`";
  return regions.front();
}

struct MatchedBuffers {
  Buffer input;
  Buffer weight;
  Buffer output;
};

MatchedBuffers MatchBuffers(const PrimFunc& func, const std::vector<int64_t>& input_shape,
                            const std::vector<int64_t>& weight_shape,
                            const std::vector<int64_t>& output_shape) {
  MatchedBuffers matched;
  for (const auto& [param, buffer] : func->buffer_map) {
    std::vector<int64_t> shape = ExpectConstantShape(buffer, buffer->name.c_str());
    if (!matched.input.defined() && IsShape(shape, input_shape)) {
      matched.input = buffer;
    } else if (!matched.weight.defined() && IsShape(shape, weight_shape)) {
      matched.weight = buffer;
    } else if (!matched.output.defined() && IsShape(shape, output_shape)) {
      matched.output = buffer;
    }
  }

  TVM_FFI_CHECK(matched.input.defined(), ValueError)
      << "BuildTyphoonGraph could not match planned input buffer";
  TVM_FFI_CHECK(matched.weight.defined(), ValueError)
      << "BuildTyphoonGraph could not match planned weight buffer";
  TVM_FFI_CHECK(matched.output.defined(), ValueError)
      << "BuildTyphoonGraph could not match planned output buffer";
  return matched;
}

PrimExpr Int32Const(int64_t value) { return IntImm(DataType::Int(32), value); }

PrimExpr Int64Const(int64_t value) { return IntImm(DataType::Int(64), value); }

Stmt RegionDeclStmt(int64_t graph_id, const RegionSpec& region) {
  return Evaluate(Call(DataType::Int(32), Op::Get("tirx.typhoon.region_decl"),
                       {Int32Const(graph_id), Int32Const(region.region_id), Int64Const(region.offset),
                        Int64Const(region.size), Int64Const(region.alignment), Int32Const(0),
                        StringImm(region.name)}));
}

Stmt DMATaskStmt(int64_t graph_id, int64_t task_id, int64_t direction, PrimExpr global_handle,
                 int64_t global_byte_offset, int64_t sram_region_id, int64_t bytes,
                 const std::vector<int64_t>& deps, int64_t sram_byte_offset = 0) {
  ffi::Array<PrimExpr> args{Int32Const(graph_id), Int32Const(task_id), Int32Const(direction),
                            global_handle, Int64Const(global_byte_offset), Int32Const(sram_region_id),
                            Int64Const(sram_byte_offset), Int64Const(bytes), Int32Const(deps.size())};
  for (int64_t dep : deps) {
    args.push_back(Int32Const(dep));
  }
  return Evaluate(Call(DataType::Int(32), Op::Get("tirx.typhoon.task_dma"), args));
}

Stmt MatmulTaskStmt(int64_t graph_id, int64_t task_id, int64_t a_region_id, int64_t b_region_id,
                    int64_t c_region_id, const std::vector<int64_t>& tile,
                    int64_t layout_code, const std::vector<int64_t>& deps) {
  TVM_FFI_CHECK_EQ(tile.size(), 3U, ValueError)
      << "BuildTyphoonGraph expects matmul_tile=[Mt,Nt,Kt]";
  ffi::Array<PrimExpr> args{Int32Const(graph_id),
                            Int32Const(task_id),
                            Int32Const(a_region_id),
                            Int32Const(b_region_id),
                            Int32Const(c_region_id),
                            Int64Const(tile[0]),
                            Int64Const(tile[1]),
                            Int64Const(tile[2]),
                            Int32Const(2),
                            Int32Const(layout_code),
                            Int32Const(deps.size())};
  for (int64_t dep : deps) {
    args.push_back(Int32Const(dep));
  }
  return Evaluate(Call(DataType::Int(32), Op::Get("tirx.typhoon.task_matmul"), args));
}

Stmt VectorTaskStmt(int64_t graph_id, int64_t task_id, int64_t op_code, int64_t in0_region_id,
                    int64_t in1_region_id, int64_t out_region_id, int64_t elem_count,
                    const std::vector<int64_t>& metadata, const std::vector<int64_t>& deps) {
  ffi::Array<PrimExpr> args{Int32Const(graph_id), Int32Const(task_id), Int32Const(op_code),
                            Int32Const(in0_region_id), Int32Const(in1_region_id),
                            Int32Const(out_region_id), Int64Const(elem_count), Int32Const(2),
                            Int32Const(metadata.size())};
  for (int64_t value : metadata) {
    args.push_back(Int64Const(value));
  }
  args.push_back(Int32Const(deps.size()));
  for (int64_t dep : deps) {
    args.push_back(Int32Const(dep));
  }
  return Evaluate(Call(DataType::Int(32), Op::Get("tirx.typhoon.task_vector"), args));
}

Stmt WaitGraphStmt(int64_t graph_id) {
  return Evaluate(Call(DataType::Int(32), Op::Get("tirx.typhoon.wait_graph"),
                       {Int32Const(graph_id)}));
}

Stmt ReshapeTaskStmt(int64_t graph_id, int64_t task_id, int64_t in_region_id, int64_t out_region_id,
                     int64_t transform_code, const std::vector<int64_t>& metadata, int64_t elem_count,
                     const std::vector<int64_t>& deps);

Stmt SubmitGraphStmt(int64_t graph_id);

int64_t BufferBytes(const Buffer& buffer) {
  int64_t bytes = 4;
  for (int64_t dim : ExpectConstantShape(buffer, buffer->name.c_str())) {
    bytes *= dim;
  }
  return bytes;
}

std::string GlobalSymbol(const PrimFunc& func) {
  auto global_symbol = func->GetAttr<ffi::String>("global_symbol");
  return global_symbol.has_value() ? std::string(global_symbol.value()) : "";
}

bool IsTyphoonTargeted(const PrimFunc& func) {
  auto target = func->GetAttr<Target>(tvm::attr::kTarget);
  return target.defined() && target.value()->kind->name == "typhoon";
}

struct StandaloneGraphSegment {
  ffi::Array<Stmt> region_stmts;
  ffi::Array<Stmt> task_stmts;
  int64_t task_count{0};
  std::vector<int64_t> root_task_ids;
  std::vector<int64_t> sink_task_ids;
};

struct LayerSegmentTemplate {
  PrimFunc func;
  StandaloneGraphSegment segment;
  std::unordered_map<const VarNode*, int> buffer_data_slot;
};

bool IsEvaluateCallOp(const Stmt& stmt, const Op& op, const CallNode** out_call = nullptr) {
  const auto* eval = stmt.as<EvaluateNode>();
  if (eval == nullptr) {
    return false;
  }
  const auto* call = eval->value.as<CallNode>();
  if (call == nullptr || !call->op.same_as(op)) {
    return false;
  }
  if (out_call != nullptr) {
    *out_call = call;
  }
  return true;
}

int64_t ExpectIntImm(const PrimExpr& expr, const char* field) {
  const auto* imm = expr.as<IntImmNode>();
  TVM_FFI_CHECK(imm != nullptr, ValueError)
      << "BuildTyphoonGraph requires constant integer " << field;
  return imm->value;
}

int64_t GetTaskDepIndex(const CallNode* call) {
  static const Op& task_dma_op = Op::Get("tirx.typhoon.task_dma");
  static const Op& task_matmul_op = Op::Get("tirx.typhoon.task_matmul");
  static const Op& task_vector_op = Op::Get("tirx.typhoon.task_vector");
  static const Op& task_reshape_op = Op::Get("tirx.typhoon.task_reshape");
  if (call->op.same_as(task_dma_op)) {
    return 8;
  }
  if (call->op.same_as(task_matmul_op)) {
    return 10;
  }
  if (call->op.same_as(task_vector_op)) {
    return 9 + ExpectIntImm(call->args[8], "num_metadata");
  }
  if (call->op.same_as(task_reshape_op)) {
    return 7 + ExpectIntImm(call->args[6], "num_metadata");
  }
  TVM_FFI_THROW(ValueError) << "BuildTyphoonGraph could not identify Typhoon task dependency slot";
  return -1;
}

std::vector<int64_t> ParseTaskDeps(const CallNode* call) {
  int64_t dep_index = GetTaskDepIndex(call);
  int64_t num_deps = ExpectIntImm(call->args[dep_index], "num_deps");
  std::vector<int64_t> deps;
  deps.reserve(num_deps);
  for (int64_t i = 0; i < num_deps; ++i) {
    deps.push_back(ExpectIntImm(call->args[dep_index + 1 + i], "dep_task_id"));
  }
  return deps;
}

StandaloneGraphSegment ExtractStandaloneGraphSegment(const PrimFunc& func) {
  static const Op& region_decl_op = Op::Get("tirx.typhoon.region_decl");
  static const Op& task_dma_op = Op::Get("tirx.typhoon.task_dma");
  static const Op& task_matmul_op = Op::Get("tirx.typhoon.task_matmul");
  static const Op& task_vector_op = Op::Get("tirx.typhoon.task_vector");
  static const Op& task_reshape_op = Op::Get("tirx.typhoon.task_reshape");
  static const Op& submit_graph_op = Op::Get("tirx.typhoon.submit_graph");
  static const Op& wait_graph_op = Op::Get("tirx.typhoon.wait_graph");

  ffi::Array<Stmt> seq;
  if (const auto* seq_stmt = func->body.as<SeqStmtNode>()) {
    seq = seq_stmt->seq;
  } else {
    seq.push_back(func->body);
  }

  StandaloneGraphSegment segment;
  std::vector<int64_t> task_ids;
  std::unordered_set<int64_t> consumed_task_ids;
  for (const Stmt& stmt : seq) {
    const CallNode* call = nullptr;
    if (IsEvaluateCallOp(stmt, region_decl_op, &call)) {
      segment.region_stmts.push_back(stmt);
      continue;
    }
    if (IsEvaluateCallOp(stmt, submit_graph_op, &call) || IsEvaluateCallOp(stmt, wait_graph_op, &call)) {
      continue;
    }
    if (IsEvaluateCallOp(stmt, task_dma_op, &call) || IsEvaluateCallOp(stmt, task_matmul_op, &call) ||
        IsEvaluateCallOp(stmt, task_vector_op, &call) || IsEvaluateCallOp(stmt, task_reshape_op, &call)) {
      int64_t task_id = ExpectIntImm(call->args[1], "task_id");
      std::vector<int64_t> deps = ParseTaskDeps(call);
      task_ids.push_back(task_id);
      if (deps.empty()) {
        segment.root_task_ids.push_back(task_id);
      }
      for (int64_t dep : deps) {
        consumed_task_ids.insert(dep);
      }
      segment.task_stmts.push_back(stmt);
      continue;
    }
    TVM_FFI_THROW(ValueError)
        << "BuildTyphoonGraph standalone canonical body contains unsupported statement";
  }

  TVM_FFI_CHECK(!task_ids.empty(), ValueError)
      << "BuildTyphoonGraph expected canonical Typhoon segment to contain tasks";
  segment.task_count = *std::max_element(task_ids.begin(), task_ids.end()) + 1;
  for (int64_t task_id : task_ids) {
    if (!consumed_task_ids.count(task_id)) {
      segment.sink_task_ids.push_back(task_id);
    }
  }
  TVM_FFI_CHECK(!segment.sink_task_ids.empty(), ValueError)
      << "BuildTyphoonGraph expected canonical Typhoon segment to have terminal tasks";
  return segment;
}

Stmt RewriteStandaloneTaskStmt(const Stmt& stmt, int64_t task_id_offset,
                               const std::vector<int64_t>& prev_segment_deps) {
  const auto* eval = stmt.as<EvaluateNode>();
  TVM_FFI_CHECK(eval != nullptr, ValueError)
      << "BuildTyphoonGraph expected Evaluate Typhoon task statement";
  const auto* call = eval->value.as<CallNode>();
  TVM_FFI_CHECK(call != nullptr, ValueError)
      << "BuildTyphoonGraph expected Typhoon task call";

  int64_t dep_index = GetTaskDepIndex(call);
  int64_t original_num_deps = ExpectIntImm(call->args[dep_index], "num_deps");
  std::vector<int64_t> rewritten_deps;
  rewritten_deps.reserve(original_num_deps + prev_segment_deps.size());
  if (original_num_deps == 0) {
    rewritten_deps.insert(rewritten_deps.end(), prev_segment_deps.begin(), prev_segment_deps.end());
  }
  for (int64_t i = 0; i < original_num_deps; ++i) {
    rewritten_deps.push_back(
        ExpectIntImm(call->args[dep_index + 1 + i], "dep_task_id") + task_id_offset);
  }

  ffi::Array<PrimExpr> args;
  for (int64_t i = 0; i < dep_index; ++i) {
    if (i == 1) {
      args.push_back(Int32Const(ExpectIntImm(call->args[i], "task_id") + task_id_offset));
    } else {
      args.push_back(call->args[i]);
    }
  }
  args.push_back(Int32Const(rewritten_deps.size()));
  for (int64_t dep : rewritten_deps) {
    args.push_back(Int32Const(dep));
  }
  return Evaluate(Call(DataType::Int(32), Downcast<Op>(call->op), args));
}

PrimFunc RewriteStandaloneSegmentBody(PrimFunc func, const StandaloneGraphSegment& segment,
                                      int64_t task_id_offset,
                                      const std::vector<int64_t>& prev_segment_deps,
                                      bool emit_regions, bool emit_submit_wait) {
  ffi::Array<Stmt> stmts;
  if (emit_regions) {
    for (const Stmt& stmt : segment.region_stmts) {
      stmts.push_back(stmt);
    }
  }
  for (const Stmt& stmt : segment.task_stmts) {
    stmts.push_back(RewriteStandaloneTaskStmt(stmt, task_id_offset, prev_segment_deps));
  }
  if (emit_submit_wait) {
    stmts.push_back(SubmitGraphStmt(0));
    stmts.push_back(WaitGraphStmt(0));
  }
  auto* n = func.CopyOnWrite();
  n->body = SeqStmt(stmts);
  return func;
}

std::vector<int64_t> OffsetTaskIds(const std::vector<int64_t>& task_ids, int64_t offset);

PrimExpr CapturedHandleExpr(int64_t graph_id, int64_t layer_id, int64_t slot) {
  ffi::Array<PrimExpr> args{StringImm("TVMTyphoonGetCapturedHandle"), Int32Const(graph_id),
                            Int32Const(layer_id), Int32Const(slot)};
  return Call(DataType::Handle(), builtin::call_extern(), args);
}

std::string SerializeLayerIdCSV(const std::vector<int64_t>& layer_ids);
Stmt CaptureCallStmt(int64_t graph_id, const std::vector<int64_t>& layer_ids, const PrimFunc& func);
Stmt CapturePackedArgsStmt(int64_t graph_id, const std::vector<int64_t>& layer_ids,
                           const PrimFunc& func);

Stmt CaptureCallStmt(int64_t graph_id, int64_t layer_id, const PrimFunc& func) {
  return CaptureCallStmt(graph_id, std::vector<int64_t>{layer_id}, func);
}

std::string SerializeLayerIdCSV(const std::vector<int64_t>& layer_ids) {
  std::ostringstream os;
  for (size_t i = 0; i < layer_ids.size(); ++i) {
    if (i != 0) {
      os << ",";
    }
    os << layer_ids[i];
  }
  return os.str();
}

Stmt CaptureCallStmt(int64_t graph_id, const std::vector<int64_t>& layer_ids, const PrimFunc& func) {
  TVM_FFI_CHECK(!layer_ids.empty(), ValueError)
      << "BuildTyphoonGraph planned capture requires at least one layer_id";
  ffi::Array<PrimExpr> args{StringImm("TVMTyphoonCaptureCallPlanned"), Int32Const(graph_id),
                            StringImm(SerializeLayerIdCSV(layer_ids)),
                            Int32Const(func->params.size())};
  for (size_t i = 0; i < 3; ++i) {
    if (i < func->params.size()) {
      args.push_back(func->params[i]);
    } else {
      args.push_back(make_zero(DataType::Handle()));
    }
  }
  return Evaluate(Call(DataType::Int(32), builtin::call_extern(), args));
}

Stmt CapturePackedArgsStmt(int64_t graph_id, int64_t layer_id, const PrimFunc& func) {
  return CapturePackedArgsStmt(graph_id, std::vector<int64_t>{layer_id}, func);
}

Stmt CapturePackedArgsStmt(int64_t graph_id, const std::vector<int64_t>& layer_ids,
                           const PrimFunc& func) {
  TVM_FFI_CHECK(!layer_ids.empty(), ValueError)
      << "BuildTyphoonGraph planned packed capture requires at least one layer_id";
  TVM_FFI_CHECK_GE(func->params.size(), 3, ValueError)
      << "BuildTyphoonGraph expected packed FFI function to have args and num_args params";
  ffi::Array<PrimExpr> args{StringImm("TVMTyphoonCapturePackedArgsPlanned"), Int32Const(graph_id),
                            StringImm(SerializeLayerIdCSV(layer_ids)), func->params[1],
                            func->params[2]};
  return Evaluate(Call(DataType::Int(32), builtin::call_extern(), args));
}

LayerSegmentTemplate MakeLayerSegmentTemplate(PrimFunc func) {
  LayerSegmentTemplate templ;
  templ.func = func;
  templ.segment = ExtractStandaloneGraphSegment(func);
  for (size_t i = 0; i < func->params.size(); ++i) {
    auto it = func->buffer_map.find(func->params[i]);
    TVM_FFI_CHECK(it != func->buffer_map.end(), ValueError)
        << "BuildTyphoonGraph expected template params to be buffer-bound";
    const Buffer& buffer = (*it).second;
    templ.buffer_data_slot[buffer->data.get()] = static_cast<int>(i);
  }
  return templ;
}

Stmt RewritePlannedTaskStmt(const Stmt& stmt, int64_t graph_id, int64_t layer_id,
                           const LayerSegmentTemplate& templ, int64_t task_id_offset,
                           const std::vector<int64_t>& prev_segment_deps) {
  const auto* eval = stmt.as<EvaluateNode>();
  TVM_FFI_CHECK(eval != nullptr, ValueError)
      << "BuildTyphoonGraph expected Evaluate Typhoon task statement";
  const auto* call = eval->value.as<CallNode>();
  TVM_FFI_CHECK(call != nullptr, ValueError) << "BuildTyphoonGraph expected Typhoon task call";

  int64_t dep_index = GetTaskDepIndex(call);
  int64_t original_num_deps = ExpectIntImm(call->args[dep_index], "num_deps");
  std::vector<int64_t> rewritten_deps;
  rewritten_deps.reserve(original_num_deps + prev_segment_deps.size());
  if (original_num_deps == 0) {
    rewritten_deps.insert(rewritten_deps.end(), prev_segment_deps.begin(), prev_segment_deps.end());
  }
  for (int64_t i = 0; i < original_num_deps; ++i) {
    rewritten_deps.push_back(
        ExpectIntImm(call->args[dep_index + 1 + i], "dep_task_id") + task_id_offset);
  }

  ffi::Array<PrimExpr> args;
  for (int64_t i = 0; i < dep_index; ++i) {
    if (i == 0) {
      args.push_back(Int32Const(graph_id));
      continue;
    }
    if (i == 1) {
      args.push_back(Int32Const(ExpectIntImm(call->args[i], "task_id") + task_id_offset));
      continue;
    }
    if (call->op.same_as(Op::Get("tirx.typhoon.task_dma")) && i == 3) {
      if (const auto* var = call->args[i].as<VarNode>()) {
        auto it = templ.buffer_data_slot.find(var);
        if (it != templ.buffer_data_slot.end()) {
          args.push_back(CapturedHandleExpr(graph_id, layer_id, it->second));
          continue;
        }
      }
    }
    args.push_back(call->args[i]);
  }
  args.push_back(Int32Const(rewritten_deps.size()));
  for (int64_t dep : rewritten_deps) {
    args.push_back(Int32Const(dep));
  }
  return Evaluate(Call(DataType::Int(32), Downcast<Op>(call->op), args));
}

ffi::Array<Stmt> BuildPlannedWholeGraphStmts(
    int64_t graph_id, const std::vector<PlannedLayer>& layers,
    const std::unordered_map<std::string, LayerSegmentTemplate>& templates) {
  ffi::Array<Stmt> stmts;
  for (const RegionSpec& region : GetCanonicalRegions()) {
    stmts.push_back(RegionDeclStmt(graph_id, region));
  }

  int64_t task_id_offset = 0;
  std::vector<int64_t> prev_segment_deps;
  for (const PlannedLayer& layer : layers) {
    auto it = templates.find(layer.symbol);
    TVM_FFI_CHECK(it != templates.end(), ValueError)
        << "BuildTyphoonGraph missing canonical segment template for `" << layer.symbol << "`";
    const LayerSegmentTemplate& templ = it->second;
    for (const Stmt& task_stmt : templ.segment.task_stmts) {
      stmts.push_back(
          RewritePlannedTaskStmt(task_stmt, graph_id, layer.layer_id, templ, task_id_offset,
                                 prev_segment_deps));
    }
    prev_segment_deps = OffsetTaskIds(templ.segment.sink_task_ids, task_id_offset);
    task_id_offset += templ.segment.task_count;
  }
  stmts.push_back(SubmitGraphStmt(graph_id));
  stmts.push_back(WaitGraphStmt(graph_id));
  return stmts;
}

int32_t CanonicalReplayKindCode(const PlannedLayer& layer) {
  if (layer.kind == "conv2d") {
    return 1;
  }
  if (layer.kind == "relu") {
    return 2;
  }
  if (layer.kind == "bias_add" || layer.kind == "add" || layer.kind == "residual_add") {
    return 3;
  }
  if (layer.kind == "max_pool2d") {
    return 4;
  }
  if (layer.kind == "global_avg_pool") {
    return 5;
  }
  if (layer.kind == "matmul") {
    return 6;
  }
  if (layer.kind == "transpose" || layer.kind == "reshape") {
    return 7;
  }
  TVM_FFI_THROW(ValueError) << "BuildTyphoonGraph does not support replay kind `" << layer.kind
                            << "`";
  return 0;
}

bool IsPackedFFISymbol(const std::string& symbol) {
  static constexpr const char* kPackedFFIPrefix = "__tvm_ffi_";
  return symbol.rfind(kPackedFFIPrefix, 0) == 0;
}

Stmt ReplayWholeGraphBeginStmt(int64_t graph_id) {
  ffi::Array<PrimExpr> args{StringImm("TVMTyphoonReplayWholeGraphBegin"), Int32Const(graph_id)};
  return Evaluate(Call(DataType::Int(32), builtin::call_extern(), args));
}

Stmt ReplayCapturedLayerStmt(int64_t graph_id, int64_t layer_id, int32_t replay_kind_code) {
  ffi::Array<PrimExpr> args{StringImm("TVMTyphoonReplayCapturedLayer"), Int32Const(graph_id),
                            Int32Const(layer_id), Int32Const(replay_kind_code)};
  return Evaluate(Call(DataType::Int(32), builtin::call_extern(), args));
}

ffi::Array<Stmt> BuildReplayWholeGraphStmts(int64_t graph_id, const PrimFunc& final_func,
                                            const std::vector<PlannedLayer>& layers) {
  ffi::Array<Stmt> stmts;
  std::vector<int64_t> final_layer_ids{layers.back().layer_id};
  bool is_packed_ffi = IsPackedFFISymbol(GlobalSymbol(final_func));
  if (is_packed_ffi) {
    stmts.push_back(CapturePackedArgsStmt(graph_id, final_layer_ids, final_func));
  } else {
    stmts.push_back(CaptureCallStmt(graph_id, final_layer_ids, final_func));
  }
  stmts.push_back(ReplayWholeGraphBeginStmt(graph_id));
  for (const PlannedLayer& layer : layers) {
    stmts.push_back(
        ReplayCapturedLayerStmt(graph_id, layer.layer_id, CanonicalReplayKindCode(layer)));
  }
  stmts.push_back(SubmitGraphStmt(graph_id));
  stmts.push_back(WaitGraphStmt(graph_id));
  if (is_packed_ffi) {
    stmts.push_back(Evaluate(ret(Integer(0))));
  }
  return stmts;
}

bool MatchesPlannedSymbol(const std::string& actual_symbol, const std::string& planned_symbol) {
  if (actual_symbol == planned_symbol) {
    return true;
  }
  static constexpr const char* kPackedFFIPrefix = "__tvm_ffi_";
  if (actual_symbol.rfind(kPackedFFIPrefix, 0) == 0) {
    return actual_symbol.substr(std::strlen(kPackedFFIPrefix)) == planned_symbol;
  }
  return false;
}

std::vector<int64_t> OffsetTaskIds(const std::vector<int64_t>& task_ids, int64_t offset) {
  std::vector<int64_t> rewritten;
  rewritten.reserve(task_ids.size());
  for (int64_t task_id : task_ids) {
    rewritten.push_back(task_id + offset);
  }
  return rewritten;
}

void AppendGraphPrelude(ffi::Array<Stmt>* stmts, int64_t graph_id,
                        const std::vector<const RegionSpec*>& regions);

int64_t ComputeConvOutputExtent(int64_t input_extent, int64_t kernel_extent, int64_t stride,
                                int64_t pad) {
  return (input_extent + 2 * pad - kernel_extent) / stride + 1;
}

std::pair<int64_t, int64_t> InferConvStrideAndPad(int64_t input_extent, int64_t output_extent,
                                                  int64_t kernel_extent) {
  for (int64_t stride = 1; stride <= 4; ++stride) {
    for (int64_t pad = 0; pad <= kernel_extent; ++pad) {
      int64_t numer = input_extent + 2 * pad - kernel_extent;
      if (numer < 0) {
        continue;
      }
      if (ComputeConvOutputExtent(input_extent, kernel_extent, stride, pad) == output_extent) {
        return {stride, pad};
      }
    }
  }
  TVM_FFI_THROW(ValueError) << "BuildTyphoonGraph could not infer canonical conv stride/pad";
  return {1, 0};
}

Conv2DConfig MakeConv2DConfig(const Buffer& input, const Buffer& weight, const Buffer& output) {
  Conv2DConfig config;
  config.input_shape = ExpectConstantShape(input, input->name.c_str());
  config.weight_shape = ExpectConstantShape(weight, weight->name.c_str());
  config.output_shape = ExpectConstantShape(output, output->name.c_str());
  TVM_FFI_CHECK_EQ(config.input_shape.size(), 4U, ValueError)
      << "BuildTyphoonGraph expects conv2d input buffers in NCHW";
  TVM_FFI_CHECK_EQ(config.weight_shape.size(), 4U, ValueError)
      << "BuildTyphoonGraph expects conv2d weight buffers in OIHW";
  TVM_FFI_CHECK_EQ(config.output_shape.size(), 4U, ValueError)
      << "BuildTyphoonGraph expects conv2d output buffers in NCHW";
  TVM_FFI_CHECK_EQ(config.input_shape[0], 1, ValueError)
      << "BuildTyphoonGraph currently supports batch=1 conv2d only";
  TVM_FFI_CHECK_EQ(config.output_shape[0], 1, ValueError)
      << "BuildTyphoonGraph currently supports batch=1 conv2d only";

  config.in_channels = config.input_shape[1];
  config.in_h = config.input_shape[2];
  config.in_w = config.input_shape[3];
  config.out_channels = config.weight_shape[0];
  config.kernel_h = config.weight_shape[2];
  config.kernel_w = config.weight_shape[3];
  config.out_h = config.output_shape[2];
  config.out_w = config.output_shape[3];
  TVM_FFI_CHECK_EQ(config.weight_shape[1], config.in_channels, ValueError)
      << "BuildTyphoonGraph expects conv2d weight/input channel agreement";
  TVM_FFI_CHECK_EQ(config.output_shape[1], config.out_channels, ValueError)
      << "BuildTyphoonGraph expects conv2d weight/output channel agreement";

  std::tie(config.stride_h, config.pad_h) =
      InferConvStrideAndPad(config.in_h, config.out_h, config.kernel_h);
  std::tie(config.stride_w, config.pad_w) =
      InferConvStrideAndPad(config.in_w, config.out_w, config.kernel_w);
  config.patch_size = config.in_channels * config.kernel_h * config.kernel_w;
  return config;
}

int64_t NCHWChannelOffsetBytes(int64_t channel, int64_t h_offset, int64_t height, int64_t width) {
  return ((channel * height) + h_offset) * width * 4;
}

int64_t NCHWChannelPlaneBytes(int64_t height, int64_t width) { return height * width * 4; }

int64_t ComputeConvOutputTileHeight(const Conv2DConfig& config, const RegionSpec& act0,
                                    const RegionSpec& col0) {
  int64_t best = 0;
  for (int64_t out_h_tile = 1; out_h_tile <= config.out_h; ++out_h_tile) {
    int64_t worst_input_h = (out_h_tile - 1) * config.stride_h + config.kernel_h;
    int64_t input_tile_bytes = config.in_channels * worst_input_h * config.in_w * 4;
    int64_t col_tile_bytes = out_h_tile * config.out_w * config.patch_size * 4;
    if (input_tile_bytes <= act0.size && col_tile_bytes <= col0.size) {
      best = out_h_tile;
    }
  }
  TVM_FFI_CHECK_GT(best, 0, ValueError)
      << "BuildTyphoonGraph could not fit canonical conv tile in SRAM";
  return best;
}

PrimFunc BuildCanonicalConv2DBody(PrimFunc func) {
  const auto& regions = GetCanonicalRegions();
  const RegionSpec& act0 = FindRegion(regions, "act0");
  const RegionSpec& act1 = FindRegion(regions, "act1");
  const RegionSpec& wgt0 = FindRegion(regions, "wgt0");
  const RegionSpec& col0 = FindRegion(regions, "col0");
  const RegionSpec& aux0 = FindRegion(regions, "aux0");

  auto it = func->buffer_map.begin();
  Buffer input = (*it).second;
  Buffer weight = (*std::next(it)).second;
  Buffer output = (*std::next(it, 2)).second;
  Conv2DConfig config = MakeConv2DConfig(input, weight, output);

  int64_t out_h_tile = ComputeConvOutputTileHeight(config, act0, col0);
  int64_t max_weight_channels = wgt0.size / (config.patch_size * 4);
  TVM_FFI_CHECK_GT(max_weight_channels, 0, ValueError)
      << "BuildTyphoonGraph could not fit conv weights in canonical SRAM region";

  const int64_t graph_id = 0;
  ffi::Array<Stmt> stmts;
  int64_t next_task_id = 0;
  std::vector<int64_t> carry_deps;
  AppendGraphPrelude(&stmts, graph_id, {&act0, &col0, &wgt0, &aux0, &act1});

  for (int64_t out_h_start = 0; out_h_start < config.out_h; out_h_start += out_h_tile) {
    int64_t current_out_h_tile = std::min<int64_t>(out_h_tile, config.out_h - out_h_start);
    int64_t raw_input_h_start = out_h_start * config.stride_h - config.pad_h;
    int64_t raw_input_h_end =
        (out_h_start + current_out_h_tile - 1) * config.stride_h - config.pad_h + config.kernel_h;
    int64_t input_h_start = std::max<int64_t>(0, raw_input_h_start);
    int64_t input_h_end = std::min<int64_t>(config.in_h, raw_input_h_end);
    int64_t input_h_tile = input_h_end - input_h_start;
    TVM_FFI_CHECK_GT(input_h_tile, 0, ValueError)
        << "BuildTyphoonGraph generated an empty conv input tile";

    int64_t last_input_dma = -1;
    for (int64_t ic = 0; ic < config.in_channels; ++ic) {
      int64_t dma_input = next_task_id++;
      std::vector<int64_t> deps = last_input_dma >= 0
                                      ? std::vector<int64_t>{last_input_dma}
                                      : carry_deps;
      stmts.push_back(DMATaskStmt(graph_id, dma_input, 0, input->data,
                                  NCHWChannelOffsetBytes(ic, input_h_start, config.in_h, config.in_w),
                                  act0.region_id, NCHWChannelPlaneBytes(input_h_tile, config.in_w), deps,
                                  ic * NCHWChannelPlaneBytes(input_h_tile, config.in_w)));
      last_input_dma = dma_input;
    }

    int64_t rows = current_out_h_tile * config.out_w;
    int64_t max_output_channels = aux0.size / (rows * 4);
    int64_t out_channel_chunk =
        std::min<int64_t>(config.out_channels, std::min(max_weight_channels, max_output_channels));
    TVM_FFI_CHECK_GT(out_channel_chunk, 0, ValueError)
        << "BuildTyphoonGraph could not fit conv output tile in canonical SRAM region";

    int64_t reshape = next_task_id++;
    stmts.push_back(ReshapeTaskStmt(
        graph_id, reshape, act0.region_id, col0.region_id, 1,
        {1, config.in_channels, input_h_tile, config.in_w, config.kernel_h, config.kernel_w,
         config.stride_h, config.stride_w, config.pad_h, config.pad_w, current_out_h_tile,
         config.out_w, input_h_start, 0, out_h_start, 0},
        rows * config.patch_size * 4, {last_input_dma}));

    std::vector<int64_t> chunk_deps = {reshape};
    for (int64_t oc_start = 0; oc_start < config.out_channels; oc_start += out_channel_chunk) {
      int64_t current_oc_chunk =
          std::min<int64_t>(out_channel_chunk, config.out_channels - oc_start);
      int64_t dma_weight = next_task_id++;
      int64_t matmul = next_task_id++;
      int64_t transpose = next_task_id++;
      stmts.push_back(
          DMATaskStmt(graph_id, dma_weight, 0, weight->data, oc_start * config.patch_size * 4,
                      wgt0.region_id, current_oc_chunk * config.patch_size * 4, chunk_deps));
      stmts.push_back(MatmulTaskStmt(graph_id, matmul, col0.region_id, wgt0.region_id, aux0.region_id,
                                     {rows, current_oc_chunk, config.patch_size}, 1,
                                     {reshape, dma_weight}));
      stmts.push_back(ReshapeTaskStmt(graph_id, transpose, aux0.region_id, act1.region_id, 2,
                                      {rows, current_oc_chunk}, rows * current_oc_chunk * 4,
                                      {matmul}));

      std::vector<int64_t> output_dma_ids;
      output_dma_ids.reserve(current_oc_chunk);
      for (int64_t oc = 0; oc < current_oc_chunk; ++oc) {
        int64_t dma_output = next_task_id++;
        stmts.push_back(DMATaskStmt(
            graph_id, dma_output, 1, output->data,
            NCHWChannelOffsetBytes(oc_start + oc, out_h_start, config.out_h, config.out_w),
            act1.region_id, NCHWChannelPlaneBytes(current_out_h_tile, config.out_w), {transpose},
            oc * NCHWChannelPlaneBytes(current_out_h_tile, config.out_w)));
        output_dma_ids.push_back(dma_output);
      }
      chunk_deps = std::move(output_dma_ids);
    }
    carry_deps = std::move(chunk_deps);
  }

  stmts.push_back(SubmitGraphStmt(graph_id));
  stmts.push_back(WaitGraphStmt(graph_id));
  auto* n = func.CopyOnWrite();
  n->body = SeqStmt(stmts);
  return func;
}

void AppendGraphPrelude(ffi::Array<Stmt>* stmts, int64_t graph_id,
                        const std::vector<const RegionSpec*>& regions) {
  for (const RegionSpec* region : regions) {
    stmts->push_back(RegionDeclStmt(graph_id, *region));
  }
}

PrimFunc BuildChunkedUnaryVectorBody(PrimFunc func, int64_t op_code) {
  auto it = func->buffer_map.begin();
  const auto& regions = GetCanonicalRegions();
  const RegionSpec& act0 = FindRegion(regions, "act0");
  const RegionSpec& act1 = FindRegion(regions, "act1");
  Buffer input = (*it).second;
  Buffer output = (*std::next(it)).second;
  int64_t total_bytes = BufferBytes(input);
  const int64_t graph_id = 0;
  ffi::Array<Stmt> stmts;
  int64_t next_task_id = 0;
  std::vector<int64_t> carry_deps;
  AppendGraphPrelude(&stmts, graph_id, {&act0, &act1});
  for (int64_t offset = 0; offset < total_bytes; offset += act0.size) {
    int64_t chunk_bytes = std::min<int64_t>(act0.size, total_bytes - offset);
    int64_t dma_in = next_task_id++;
    int64_t vector = next_task_id++;
    int64_t dma_out = next_task_id++;
    stmts.push_back(DMATaskStmt(graph_id, dma_in, 0, input->data, offset, act0.region_id,
                                chunk_bytes, carry_deps));
    stmts.push_back(VectorTaskStmt(graph_id, vector, op_code, act0.region_id, -1, act1.region_id,
                                   chunk_bytes / 4, {}, {dma_in}));
    stmts.push_back(DMATaskStmt(graph_id, dma_out, 1, output->data, offset, act1.region_id,
                                chunk_bytes, {vector}));
    carry_deps = {dma_out};
  }
  stmts.push_back(SubmitGraphStmt(graph_id));
  stmts.push_back(WaitGraphStmt(graph_id));
  auto* n = func.CopyOnWrite();
  n->body = SeqStmt(stmts);
  return func;
}

PrimFunc BuildChunkedAddBody(PrimFunc func) {
  const auto& regions = GetCanonicalRegions();
  const RegionSpec& act0 = FindRegion(regions, "act0");
  const RegionSpec& act1 = FindRegion(regions, "act1");
  const RegionSpec& residual = FindRegion(regions, "residual");
  auto it = func->buffer_map.begin();
  Buffer lhs = (*it).second;
  Buffer rhs = (*std::next(it)).second;
  Buffer output = (*std::next(it, 2)).second;
  int64_t lhs_bytes = BufferBytes(lhs);
  int64_t rhs_bytes = BufferBytes(rhs);
  const int64_t graph_id = 0;
  ffi::Array<Stmt> stmts;
  int64_t next_task_id = 0;
  std::vector<int64_t> carry_deps;
  AppendGraphPrelude(&stmts, graph_id, {&act0, &residual, &act1});
  bool broadcast = rhs_bytes < lhs_bytes;
  if (!broadcast) {
    for (int64_t offset = 0; offset < lhs_bytes; offset += act0.size) {
      int64_t chunk_bytes = std::min<int64_t>(act0.size, lhs_bytes - offset);
      int64_t dma_lhs = next_task_id++;
      int64_t dma_rhs = next_task_id++;
      int64_t vector = next_task_id++;
      int64_t dma_out = next_task_id++;
      stmts.push_back(
          DMATaskStmt(graph_id, dma_lhs, 0, lhs->data, offset, act0.region_id, chunk_bytes, carry_deps));
      stmts.push_back(DMATaskStmt(graph_id, dma_rhs, 0, rhs->data, offset, residual.region_id,
                                  chunk_bytes, carry_deps));
      stmts.push_back(VectorTaskStmt(graph_id, vector, 0, act0.region_id, residual.region_id,
                                     act1.region_id, chunk_bytes / 4, {}, {dma_lhs, dma_rhs}));
      stmts.push_back(DMATaskStmt(graph_id, dma_out, 1, output->data, offset, act1.region_id,
                                  chunk_bytes, {vector}));
      carry_deps = {dma_out};
    }
  } else {
    int64_t inner = rhs_bytes / 4;
    int64_t total_elems = lhs_bytes / 4;
    int64_t outer_total = total_elems / inner;
    int64_t outer_chunk = std::max<int64_t>(1, act0.size / rhs_bytes);
    for (int64_t outer_offset = 0; outer_offset < outer_total; outer_offset += outer_chunk) {
      int64_t outer = std::min<int64_t>(outer_chunk, outer_total - outer_offset);
      int64_t chunk_bytes = outer * rhs_bytes;
      int64_t global_offset = outer_offset * rhs_bytes;
      int64_t dma_lhs = next_task_id++;
      int64_t dma_rhs = next_task_id++;
      int64_t vector = next_task_id++;
      int64_t dma_out = next_task_id++;
      stmts.push_back(DMATaskStmt(graph_id, dma_lhs, 0, lhs->data, global_offset, act0.region_id,
                                  chunk_bytes, carry_deps));
      stmts.push_back(
          DMATaskStmt(graph_id, dma_rhs, 0, rhs->data, 0, residual.region_id, rhs_bytes, carry_deps));
      stmts.push_back(VectorTaskStmt(graph_id, vector, 0, act0.region_id, residual.region_id,
                                     act1.region_id, chunk_bytes / 4, {outer, inner},
                                     {dma_lhs, dma_rhs}));
      stmts.push_back(DMATaskStmt(graph_id, dma_out, 1, output->data, global_offset, act1.region_id,
                                  chunk_bytes, {vector}));
      carry_deps = {dma_out};
    }
  }
  stmts.push_back(SubmitGraphStmt(graph_id));
  stmts.push_back(WaitGraphStmt(graph_id));
  auto* n = func.CopyOnWrite();
  n->body = SeqStmt(stmts);
  return func;
}

PrimFunc BuildChunkedCopyBody(PrimFunc func) {
  const auto& regions = GetCanonicalRegions();
  const RegionSpec& act0 = FindRegion(regions, "act0");
  const RegionSpec& act1 = FindRegion(regions, "act1");
  auto it = func->buffer_map.begin();
  Buffer input = (*it).second;
  Buffer output = (*std::next(it)).second;
  int64_t total_bytes = BufferBytes(input);
  const int64_t graph_id = 0;
  ffi::Array<Stmt> stmts;
  int64_t next_task_id = 0;
  std::vector<int64_t> carry_deps;
  AppendGraphPrelude(&stmts, graph_id, {&act0, &act1});
  for (int64_t offset = 0; offset < total_bytes; offset += act0.size) {
    int64_t chunk_bytes = std::min<int64_t>(act0.size, total_bytes - offset);
    int64_t dma_in = next_task_id++;
    int64_t reshape = next_task_id++;
    int64_t dma_out = next_task_id++;
    stmts.push_back(DMATaskStmt(graph_id, dma_in, 0, input->data, offset, act0.region_id,
                                chunk_bytes, carry_deps));
    stmts.push_back(ReshapeTaskStmt(graph_id, reshape, act0.region_id, act1.region_id, 0, {},
                                    chunk_bytes, {dma_in}));
    stmts.push_back(DMATaskStmt(graph_id, dma_out, 1, output->data, offset, act1.region_id,
                                chunk_bytes, {reshape}));
    carry_deps = {dma_out};
  }
  stmts.push_back(SubmitGraphStmt(graph_id));
  stmts.push_back(WaitGraphStmt(graph_id));
  auto* n = func.CopyOnWrite();
  n->body = SeqStmt(stmts);
  return func;
}

PrimFunc BuildTransposeCopyBody(PrimFunc func) {
  // The downstream dense path consumes the RHS as [N, K] with layout_code=1, so
  // the canonical fc.weight buffer can be forwarded without a materialized transpose.
  return BuildChunkedCopyBody(std::move(func));
}

PrimFunc BuildMaxPoolBody(PrimFunc func) {
  const auto& regions = GetCanonicalRegions();
  const RegionSpec& act0 = FindRegion(regions, "act0");
  const RegionSpec& act1 = FindRegion(regions, "act1");
  auto it = func->buffer_map.begin();
  Buffer input = (*it).second;
  Buffer output = (*std::next(it)).second;
  std::vector<int64_t> in_shape = ExpectConstantShape(input, input->name.c_str());
  std::vector<int64_t> out_shape = ExpectConstantShape(output, output->name.c_str());
  int64_t in_h = in_shape[2];
  int64_t in_w = in_shape[3];
  int64_t out_h = out_shape[2];
  int64_t out_w = out_shape[3];
  int64_t c_total = in_shape[1];
  int64_t c_chunk = std::max<int64_t>(1, act0.size / (in_h * in_w * 4));
  c_chunk = std::min<int64_t>(c_chunk, c_total);
  const int64_t graph_id = 0;
  ffi::Array<Stmt> stmts;
  int64_t next_task_id = 0;
  std::vector<int64_t> carry_deps;
  AppendGraphPrelude(&stmts, graph_id, {&act0, &act1});
  for (int64_t c_offset = 0; c_offset < c_total; c_offset += c_chunk) {
    int64_t c = std::min<int64_t>(c_chunk, c_total - c_offset);
    int64_t input_offset = c_offset * in_h * in_w * 4;
    int64_t output_offset = c_offset * out_h * out_w * 4;
    int64_t input_bytes = c * in_h * in_w * 4;
    int64_t output_bytes = c * out_h * out_w * 4;
    int64_t dma_in = next_task_id++;
    int64_t vector = next_task_id++;
    int64_t dma_out = next_task_id++;
    stmts.push_back(
        DMATaskStmt(graph_id, dma_in, 0, input->data, input_offset, act0.region_id, input_bytes,
                    carry_deps));
    stmts.push_back(VectorTaskStmt(graph_id, vector, 2, act0.region_id, -1, act1.region_id,
                                   c * out_h * out_w,
                                   {1, c, in_h, in_w, 3, 3, 2, 2, 1, 1, out_h, out_w}, {dma_in}));
    stmts.push_back(DMATaskStmt(graph_id, dma_out, 1, output->data, output_offset, act1.region_id,
                                output_bytes, {vector}));
    carry_deps = {dma_out};
  }
  stmts.push_back(SubmitGraphStmt(graph_id));
  stmts.push_back(WaitGraphStmt(graph_id));
  auto* n = func.CopyOnWrite();
  n->body = SeqStmt(stmts);
  return func;
}

PrimFunc BuildGlobalAveragePoolBody(PrimFunc func) {
  const auto& regions = GetCanonicalRegions();
  const RegionSpec& act0 = FindRegion(regions, "act0");
  const RegionSpec& act1 = FindRegion(regions, "act1");
  auto it = func->buffer_map.begin();
  Buffer input = (*it).second;
  Buffer output = (*std::next(it)).second;
  std::vector<int64_t> in_shape = ExpectConstantShape(input, input->name.c_str());
  int64_t c_total = in_shape[1];
  int64_t in_h = in_shape[2];
  int64_t in_w = in_shape[3];
  int64_t c_chunk = std::max<int64_t>(1, act0.size / (in_h * in_w * 4));
  c_chunk = std::min<int64_t>(c_chunk, c_total);
  const int64_t graph_id = 0;
  ffi::Array<Stmt> stmts;
  int64_t next_task_id = 0;
  std::vector<int64_t> carry_deps;
  AppendGraphPrelude(&stmts, graph_id, {&act0, &act1});
  for (int64_t c_offset = 0; c_offset < c_total; c_offset += c_chunk) {
    int64_t c = std::min<int64_t>(c_chunk, c_total - c_offset);
    int64_t input_offset = c_offset * in_h * in_w * 4;
    int64_t output_offset = c_offset * 4;
    int64_t input_bytes = c * in_h * in_w * 4;
    int64_t output_bytes = c * 4;
    int64_t dma_in = next_task_id++;
    int64_t vector = next_task_id++;
    int64_t dma_out = next_task_id++;
    stmts.push_back(
        DMATaskStmt(graph_id, dma_in, 0, input->data, input_offset, act0.region_id, input_bytes,
                    carry_deps));
    stmts.push_back(VectorTaskStmt(graph_id, vector, 3, act0.region_id, -1, act1.region_id, c,
                                   {1, c, in_h, in_w}, {dma_in}));
    stmts.push_back(DMATaskStmt(graph_id, dma_out, 1, output->data, output_offset, act1.region_id,
                                output_bytes, {vector}));
    carry_deps = {dma_out};
  }
  stmts.push_back(SubmitGraphStmt(graph_id));
  stmts.push_back(WaitGraphStmt(graph_id));
  auto* n = func.CopyOnWrite();
  n->body = SeqStmt(stmts);
  return func;
}

PrimFunc BuildDenseMatmulBody(PrimFunc func) {
  const auto& regions = GetCanonicalRegions();
  const RegionSpec& act0 = FindRegion(regions, "act0");
  const RegionSpec& act1 = FindRegion(regions, "act1");
  const RegionSpec& wgt0 = FindRegion(regions, "wgt0");
  auto it = func->buffer_map.begin();
  Buffer input = (*it).second;
  Buffer weight = (*std::next(it)).second;
  Buffer output = (*std::next(it, 2)).second;
  std::vector<int64_t> input_shape = ExpectConstantShape(input, input->name.c_str());
  std::vector<int64_t> output_shape = ExpectConstantShape(output, output->name.c_str());
  int64_t k = input_shape[1];
  int64_t n_total = output_shape[1];
  int64_t n_chunk = std::max<int64_t>(1, wgt0.size / (k * 4));
  n_chunk = std::min<int64_t>(n_chunk, n_total);
  const int64_t graph_id = 0;
  ffi::Array<Stmt> stmts;
  int64_t next_task_id = 0;
  std::vector<int64_t> carry_deps;
  AppendGraphPrelude(&stmts, graph_id, {&act0, &wgt0, &act1});
  for (int64_t n_offset = 0; n_offset < n_total; n_offset += n_chunk) {
    int64_t n = std::min<int64_t>(n_chunk, n_total - n_offset);
    int64_t weight_offset = n_offset * k * 4;
    int64_t weight_bytes = n * k * 4;
    int64_t output_bytes = n * 4;
    int64_t dma_input = next_task_id++;
    int64_t dma_weight = next_task_id++;
    int64_t matmul = next_task_id++;
    int64_t dma_out = next_task_id++;
    stmts.push_back(
        DMATaskStmt(graph_id, dma_input, 0, input->data, 0, act0.region_id, k * 4, carry_deps));
    stmts.push_back(DMATaskStmt(graph_id, dma_weight, 0, weight->data, weight_offset,
                                wgt0.region_id, weight_bytes, carry_deps));
    stmts.push_back(MatmulTaskStmt(graph_id, matmul, act0.region_id, wgt0.region_id, act1.region_id,
                                   {1, n, k}, 1, {dma_input, dma_weight}));
    stmts.push_back(DMATaskStmt(graph_id, dma_out, 1, output->data, n_offset * 4, act1.region_id,
                                output_bytes, {matmul}));
    carry_deps = {dma_out};
  }
  stmts.push_back(SubmitGraphStmt(graph_id));
  stmts.push_back(WaitGraphStmt(graph_id));
  auto* n = func.CopyOnWrite();
  n->body = SeqStmt(stmts);
  return func;
}

std::optional<PrimFunc> MaybeBuildCanonicalFunction(PrimFunc func) {
  std::string symbol = GlobalSymbol(func);
  if (symbol.empty()) {
    return std::nullopt;
  }
  if (symbol.rfind("conv2d", 0) == 0) {
    return BuildCanonicalConv2DBody(std::move(func));
  }
  if (symbol.rfind("relu", 0) == 0) {
    return BuildChunkedUnaryVectorBody(std::move(func), 1);
  }
  if (symbol == "max_pool2d") {
    return BuildMaxPoolBody(std::move(func));
  }
  if (symbol == "mean") {
    return BuildGlobalAveragePoolBody(std::move(func));
  }
  if (symbol == "matmul") {
    return BuildDenseMatmulBody(std::move(func));
  }
  if (symbol == "transpose") {
    return BuildTransposeCopyBody(std::move(func));
  }
  if (symbol.rfind("reshape", 0) == 0) {
    return BuildChunkedCopyBody(std::move(func));
  }
  if (symbol.rfind("add", 0) == 0) {
    return BuildChunkedAddBody(std::move(func));
  }
  return std::nullopt;
}

Stmt ReshapeTaskStmt(int64_t graph_id, int64_t task_id, int64_t in_region_id, int64_t out_region_id,
                     int64_t transform_code, const std::vector<int64_t>& metadata, int64_t elem_count,
                     const std::vector<int64_t>& deps) {
  ffi::Array<PrimExpr> args{Int32Const(graph_id), Int32Const(task_id), Int32Const(in_region_id),
                            Int32Const(out_region_id), Int64Const(elem_count), Int32Const(transform_code),
                            Int32Const(metadata.size())};
  for (int64_t value : metadata) {
    args.push_back(Int64Const(value));
  }
  args.push_back(Int32Const(deps.size()));
  for (int64_t dep : deps) {
    args.push_back(Int32Const(dep));
  }
  return Evaluate(Call(DataType::Int(32), Op::Get("tirx.typhoon.task_reshape"), args));
}

Stmt SubmitGraphStmt(int64_t graph_id) {
  return Evaluate(
      Call(DataType::Int(32), Op::Get("tirx.typhoon.submit_graph"), {Int32Const(graph_id)}));
}

std::vector<int64_t> BuildIm2ColMetadata(const std::vector<int64_t>& input_shape,
                                         const std::vector<int64_t>& weight_shape) {
  TVM_FFI_CHECK_EQ(input_shape.size(), 4U, ValueError)
      << "BuildTyphoonGraph expects NCHW input shape";
  TVM_FFI_CHECK_EQ(weight_shape.size(), 4U, ValueError)
      << "BuildTyphoonGraph expects OIHW weight shape";
  int64_t input_channels = input_shape[1];
  int64_t kernel_h = weight_shape[2];
  int64_t kernel_w = weight_shape[3];

  // Emit a single tile-sized im2col task that fits the fixed SRAM plan.
  return {1, input_channels, 15, 15, kernel_h, kernel_w, 2, 2, 3, 3, 8, 8};
}

int64_t ComputeIm2ColBytes(const std::vector<int64_t>& metadata) {
  TVM_FFI_CHECK_EQ(metadata.size(), 12U, ValueError)
      << "BuildTyphoonGraph expects 12 im2col metadata values";
  return metadata[0] * metadata[1] * metadata[4] * metadata[5] * metadata[10] * metadata[11] * 4;
}

bool HasTyphoonPrimFunc(const IRModule& mod) {
  for (const auto& [gvar, base_func] : mod->functions) {
    const auto* func = base_func.as<PrimFuncNode>();
    if (func == nullptr) {
      continue;
    }
    auto target = func->GetAttr<Target>(tvm::attr::kTarget);
    if (target.defined() && target.value()->kind->name == "typhoon") {
      return true;
    }
  }
  return false;
}

PrimFunc BuildGraphBody(PrimFunc func, int64_t graph_id, const std::string& resnet_plan_json,
                        const std::string& sram_plan_json) {
  std::vector<int64_t> input_shape = ExtractArray(resnet_plan_json, "input_shape");
  std::vector<int64_t> output_shape = ExtractArray(resnet_plan_json, "logical_output_shape");
  std::vector<int64_t> weight_shape = ExtractArray(resnet_plan_json, "weight_shape");
  std::vector<int64_t> tile = ExtractArray(sram_plan_json, "matmul_tile");
  std::vector<RegionSpec> regions = ExtractRegions(sram_plan_json);

  MatchedBuffers buffers = MatchBuffers(func, input_shape, weight_shape, output_shape);
  const RegionSpec& act0 = FindRegion(regions, "act0");
  const RegionSpec& act1 = FindRegion(regions, "act1");
  const RegionSpec& wgt0 = FindRegion(regions, "wgt0");
  const RegionSpec& col0 = FindRegion(regions, "col0");
  const RegionSpec& aux0 = FindRegion(regions, "aux0");

  std::vector<int64_t> im2col_metadata = BuildIm2ColMetadata(input_shape, weight_shape);
  int64_t im2col_bytes = ComputeIm2ColBytes(im2col_metadata);
  int64_t matmul_output_elems = tile[0] * tile[1];
  int64_t matmul_output_bytes = matmul_output_elems * 4;

  ffi::Array<Stmt> stmts;
  for (const RegionSpec& region : regions) {
    stmts.push_back(RegionDeclStmt(graph_id, region));
  }
  stmts.push_back(DMATaskStmt(graph_id, 0, 0, buffers.input->data, act0.region_id,
                              0, input_shape[1] * 15 * 15 * 4, {}));
  stmts.push_back(DMATaskStmt(graph_id, 1, 0, buffers.weight->data, 0, wgt0.region_id,
                              tile[1] * tile[2] * 4, {}));
  stmts.push_back(ReshapeTaskStmt(graph_id, 2, act0.region_id, col0.region_id, 1, im2col_metadata,
                                  im2col_bytes, {0}));
  stmts.push_back(MatmulTaskStmt(graph_id, 3, col0.region_id, wgt0.region_id, aux0.region_id,
                                 tile, 0, {1, 2}));
  stmts.push_back(VectorTaskStmt(graph_id, 4, 1, aux0.region_id, -1, act1.region_id,
                                 matmul_output_elems, {}, {3}));
  stmts.push_back(
      DMATaskStmt(graph_id, 5, 1, buffers.output->data, 0, act1.region_id, matmul_output_bytes, {4}));
  stmts.push_back(SubmitGraphStmt(graph_id));
  stmts.push_back(WaitGraphStmt(graph_id));

  auto* n = func.CopyOnWrite();
  n->body = SeqStmt(stmts);
  return func;
}

}  // namespace

namespace transform {

Pass BuildTyphoonGraph() {
  auto pass_func = [](IRModule mod, PassContext ctx) {
    if (!HasTyphoonPrimFunc(mod)) {
      return mod;
    }

    auto resnet_plan = mod->GetAttr<ffi::String>("typhoon_resnet18_plan");
    auto sram_plan = mod->GetAttr<ffi::String>("typhoon_sram_plan");
    if (!resnet_plan.has_value() && !sram_plan.has_value()) {
      return mod;
    }
    TVM_FFI_CHECK(resnet_plan.has_value() && sram_plan.has_value(), ValueError)
        << "BuildTyphoonGraph requires both `typhoon_resnet18_plan` and `typhoon_sram_plan`";

    IRModule updated = mod;
    auto* write_ptr = updated.CopyOnWrite();
    int64_t graph_id = 0;
    bool single_function = mod->functions.size() == 1;
    std::vector<PlannedLayer> planned_layers;
    if (!single_function) {
      planned_layers = ParsePlannedLayers(resnet_plan.value());
    }
    std::unordered_set<std::string> planned_symbols;
    std::unordered_map<std::string, std::vector<int64_t>> planned_layer_ids_by_symbol;
    for (const PlannedLayer& layer : planned_layers) {
      planned_symbols.insert(layer.symbol);
      planned_layer_ids_by_symbol[layer.symbol].push_back(layer.layer_id);
    }
    std::string final_symbol = planned_layers.empty() ? "" : planned_layers.back().symbol;

    std::unordered_map<std::string, std::pair<GlobalVar, PrimFunc>> typhoon_funcs_by_symbol;
    std::unordered_map<std::string, LayerSegmentTemplate> layer_templates;
    for (const auto& [gvar, base_func] : mod->functions) {
      const auto* func = base_func.as<PrimFuncNode>();
      if (func == nullptr) {
        continue;
      }
      PrimFunc prim_func = ffi::GetRef<PrimFunc>(func);
      auto target = prim_func->GetAttr<Target>(tvm::attr::kTarget);
      if (!target.defined() || target.value()->kind->name != "typhoon") {
        continue;
      }
      if (single_function) {
        write_ptr->Update(gvar, BuildGraphBody(std::move(prim_func), graph_id++, resnet_plan.value(),
                                               sram_plan.value()));
        continue;
      }
      std::string symbol = GlobalSymbol(prim_func);
      if (symbol.empty()) {
        continue;
      }
      typhoon_funcs_by_symbol[symbol] = {gvar, prim_func};
      if (planned_symbols.count(symbol) != 0) {
        auto rewritten = MaybeBuildCanonicalFunction(prim_func);
        TVM_FFI_CHECK(rewritten.has_value(), ValueError)
            << "BuildTyphoonGraph could not build canonical segment for `" << symbol << "`";
        layer_templates.emplace(symbol, MakeLayerSegmentTemplate(rewritten.value()));
      }
    }

    if (!single_function && !planned_layers.empty()) {
      for (const PlannedLayer& layer : planned_layers) {
        TVM_FFI_CHECK(typhoon_funcs_by_symbol.count(layer.symbol) != 0, ValueError)
            << "BuildTyphoonGraph plan references missing symbol `" << layer.symbol << "`";
        TVM_FFI_CHECK(layer_templates.count(layer.symbol) != 0, ValueError)
            << "BuildTyphoonGraph plan references untemplated symbol `" << layer.symbol << "`";
      }

      for (const auto& [symbol, entry] : typhoon_funcs_by_symbol) {
        if (planned_symbols.count(symbol) == 0) {
          continue;
        }
        PrimFunc rewritten = entry.second;
        auto* n = rewritten.CopyOnWrite();
        ffi::Array<Stmt> stmts{
            CaptureCallStmt(graph_id, planned_layer_ids_by_symbol.at(symbol), entry.second)};
        if (symbol == final_symbol) {
          for (const Stmt& stmt : BuildPlannedWholeGraphStmts(graph_id, planned_layers, layer_templates)) {
            stmts.push_back(stmt);
          }
        }
        n->body = stmts.size() == 1 ? stmts[0] : SeqStmt(stmts);
        write_ptr->Update(entry.first, rewritten);
      }
      return updated;
    }
    return updated;
  };
  return CreateModulePass(pass_func, 0, "tirx.BuildTyphoonGraph", {});
}

Pass CompactTyphoonWholeGraph() {
  auto pass_func = [](IRModule mod, PassContext ctx) {
    auto resnet_plan = mod->GetAttr<ffi::String>("typhoon_resnet18_plan");
    if (!resnet_plan.has_value()) {
      return mod;
    }

    std::vector<PlannedLayer> planned_layers = ParsePlannedLayers(resnet_plan.value());
    TVM_FFI_CHECK(!planned_layers.empty(), ValueError)
        << "CompactTyphoonWholeGraph requires planned layers";
    std::string final_symbol = planned_layers.back().symbol;

    IRModule updated = mod;
    for (const auto& [gvar, base_func] : mod->functions) {
      const auto* func = base_func.as<PrimFuncNode>();
      if (func == nullptr) {
        continue;
      }
      PrimFunc prim_func = ffi::GetRef<PrimFunc>(func);
      auto target = prim_func->GetAttr<Target>(tvm::attr::kTarget);
      if (!target.defined() || target.value()->kind->name != "typhoon") {
        continue;
      }
      if (!MatchesPlannedSymbol(GlobalSymbol(prim_func), final_symbol)) {
        continue;
      }
      auto* n = prim_func.CopyOnWrite();
      n->body = SeqStmt(BuildReplayWholeGraphStmts(0, prim_func, planned_layers));
      updated->Add(gvar, prim_func, true);
      break;
    }
    return updated;
  };
  return CreateModulePass(pass_func, 0, "tirx.CompactTyphoonWholeGraph", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tirx.transform.BuildTyphoonGraph", BuildTyphoonGraph);
  refl::GlobalDef().def("tirx.transform.CompactTyphoonWholeGraph", CompactTyphoonWholeGraph);
}

}  // namespace transform

}  // namespace tirx
}  // namespace tvm
