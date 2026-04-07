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
      if (auto rewritten = MaybeBuildCanonicalFunction(prim_func)) {
        write_ptr->Update(gvar, std::move(rewritten.value()));
      }
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
