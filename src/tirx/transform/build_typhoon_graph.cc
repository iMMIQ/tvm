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
#include <tvm/tirx/expr.h>
#include <tvm/tirx/function.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/transform.h>

#include <optional>
#include <regex>
#include <string>
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
                 int64_t sram_region_id, int64_t bytes, const std::vector<int64_t>& deps) {
  ffi::Array<PrimExpr> args{Int32Const(graph_id), Int32Const(task_id), Int32Const(direction),
                            global_handle, Int64Const(0), Int32Const(sram_region_id),
                            Int64Const(bytes), Int32Const(deps.size())};
  for (int64_t dep : deps) {
    args.push_back(Int32Const(dep));
  }
  return Evaluate(Call(DataType::Int(32), Op::Get("tirx.typhoon.task_dma"), args));
}

Stmt MatmulTaskStmt(int64_t graph_id, int64_t task_id, int64_t a_region_id, int64_t b_region_id,
                    int64_t c_region_id, const std::vector<int64_t>& tile,
                    const std::vector<int64_t>& deps) {
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
                            Int32Const(0),
                            Int32Const(deps.size())};
  for (int64_t dep : deps) {
    args.push_back(Int32Const(dep));
  }
  return Evaluate(Call(DataType::Int(32), Op::Get("tirx.typhoon.task_matmul"), args));
}

Stmt VectorTaskStmt(int64_t graph_id, int64_t task_id, int64_t in_region_id, int64_t out_region_id,
                    int64_t elem_count, const std::vector<int64_t>& deps) {
  ffi::Array<PrimExpr> args{Int32Const(graph_id), Int32Const(task_id), Int32Const(1),
                            Int32Const(in_region_id), Int32Const(-1), Int32Const(out_region_id),
                            Int64Const(elem_count), Int32Const(2), Int32Const(0),
                            Int32Const(deps.size())};
  for (int64_t dep : deps) {
    args.push_back(Int32Const(dep));
  }
  return Evaluate(Call(DataType::Int(32), Op::Get("tirx.typhoon.task_vector"), args));
}

Stmt ReshapeTaskStmt(int64_t graph_id, int64_t task_id, int64_t in_region_id, int64_t out_region_id,
                     const std::vector<int64_t>& metadata, int64_t elem_count,
                     const std::vector<int64_t>& deps) {
  ffi::Array<PrimExpr> args{Int32Const(graph_id), Int32Const(task_id), Int32Const(in_region_id),
                            Int32Const(out_region_id), Int64Const(elem_count), Int32Const(1),
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
                              input_shape[1] * 15 * 15 * 4, {}));
  stmts.push_back(DMATaskStmt(graph_id, 1, 0, buffers.weight->data, wgt0.region_id,
                              tile[1] * tile[2] * 4, {}));
  stmts.push_back(ReshapeTaskStmt(graph_id, 2, act0.region_id, col0.region_id, im2col_metadata,
                                  im2col_bytes, {0}));
  stmts.push_back(MatmulTaskStmt(graph_id, 3, col0.region_id, wgt0.region_id, aux0.region_id,
                                 tile, {1, 2}));
  stmts.push_back(VectorTaskStmt(graph_id, 4, aux0.region_id, act1.region_id, matmul_output_elems,
                                 {3}));
  stmts.push_back(
      DMATaskStmt(graph_id, 5, 1, buffers.output->data, act1.region_id, matmul_output_bytes, {4}));
  stmts.push_back(SubmitGraphStmt(graph_id));

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
      write_ptr->Update(gvar, BuildGraphBody(std::move(prim_func), graph_id++, resnet_plan.value(),
                                             sram_plan.value()));
    }
    return updated;
  };
  return CreateModulePass(pass_func, 0, "tirx.BuildTyphoonGraph", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tirx.transform.BuildTyphoonGraph", BuildTyphoonGraph);
}

}  // namespace transform

}  // namespace tirx
}  // namespace tvm
