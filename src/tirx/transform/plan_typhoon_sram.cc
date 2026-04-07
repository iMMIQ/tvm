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
 * \file plan_typhoon_sram.cc
 * \brief Plan fixed SRAM regions for Typhoon ResNet18 lowering.
 */
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/attrs.h>
#include <tvm/ffi/string.h>
#include <tvm/target/target.h>
#include <tvm/tirx/function.h>
#include <tvm/tirx/transform.h>

#include <algorithm>
#include <regex>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace tvm {
namespace tirx {

namespace {

struct LayerInfo {
  int64_t layer_id{0};
  std::string kind;
  std::vector<int64_t> logical_input_shape;
  std::vector<int64_t> logical_output_shape;
  std::vector<int64_t> weight_shape;
  bool requires_im2col{false};
};

struct EdgeInfo {
  int64_t src_layer_id{0};
  int64_t dst_layer_id{0};
  std::string edge_kind;
};

struct RegionSpec {
  int64_t region_id;
  const char* name;
  int64_t offset;
  int64_t size;
  int64_t alignment;
};

constexpr int64_t kAlignment = 64;
constexpr int64_t kActRegionSize = 65536;
constexpr int64_t kResidualRegionSize = 65536;
constexpr int64_t kWeightRegionSize = 589824;
constexpr int64_t kColRegionSize = 147456;
constexpr int64_t kAuxRegionSize = 32768;
constexpr int64_t kPeakLiveBytes =
    kActRegionSize * 2 + kResidualRegionSize + kWeightRegionSize + kColRegionSize + kAuxRegionSize;

const std::vector<RegionSpec>& GetCanonicalRegions() {
  static const std::vector<RegionSpec> regions = {
      {0, "act0", 0, kActRegionSize, kAlignment},
      {1, "act1", kActRegionSize, kActRegionSize, kAlignment},
      {2, "residual", kActRegionSize * 2, kResidualRegionSize, kAlignment},
      {3, "wgt0", kActRegionSize * 2 + kResidualRegionSize, kWeightRegionSize, kAlignment},
      {4, "col0", kActRegionSize * 2 + kResidualRegionSize + kWeightRegionSize, kColRegionSize,
       kAlignment},
      {5, "aux0",
       kActRegionSize * 2 + kResidualRegionSize + kWeightRegionSize + kColRegionSize, kAuxRegionSize,
       kAlignment},
  };
  return regions;
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
  std::regex pattern(std::string("\"") + key + "\"\\s*:\\s*\\[([^\\]]*)\\]");
  std::smatch match;
  TVM_FFI_CHECK(std::regex_search(json, match, pattern), ValueError)
      << "PlanTyphoonSRAM could not find `" << key << "` in typhoon_resnet18_plan";
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

bool ExtractBool(const std::string& json, const char* key) {
  std::regex pattern(std::string("\"") + key + "\"\\s*:\\s*(true|false)");
  std::smatch match;
  TVM_FFI_CHECK(std::regex_search(json, match, pattern), ValueError)
      << "PlanTyphoonSRAM could not find `" << key << "` in typhoon_resnet18_plan";
  return match[1].str() == "true";
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
      << "PlanTyphoonSRAM could not find `" << key << "` in typhoon_resnet18_plan";
  return std::stoll(match[1].str());
}

std::string ExtractString(const std::string& json, const char* key) {
  std::regex pattern(std::string("\"") + key + "\"\\s*:\\s*\"([^\"]+)\"");
  std::smatch match;
  TVM_FFI_CHECK(std::regex_search(json, match, pattern), ValueError)
      << "PlanTyphoonSRAM could not find `" << key << "` in typhoon_resnet18_plan";
  return match[1].str();
}

std::string ExtractJSONArrayText(const std::string& json, const char* key) {
  std::string needle = std::string("\"") + key + "\"";
  size_t key_pos = json.find(needle);
  TVM_FFI_CHECK_NE(key_pos, std::string::npos, ValueError)
      << "PlanTyphoonSRAM could not find `" << key << "` in typhoon_resnet18_plan";
  size_t array_start = json.find('[', key_pos);
  TVM_FFI_CHECK_NE(array_start, std::string::npos, ValueError)
      << "PlanTyphoonSRAM could not find array `" << key << "` in typhoon_resnet18_plan";

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
  TVM_FFI_THROW(ValueError) << "PlanTyphoonSRAM found unterminated array `" << key << "`";
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

std::vector<LayerInfo> ParseLayers(const std::string& plan) {
  std::vector<LayerInfo> layers;
  for (const std::string& object : SplitTopLevelObjects(ExtractJSONArrayText(plan, "layers"))) {
    LayerInfo layer;
    layer.layer_id = ExtractInt(object, "layer_id");
    layer.kind = ExtractString(object, "kind");
    layer.logical_input_shape = ExtractArray(object, "logical_input_shape");
    layer.logical_output_shape = ExtractArray(object, "logical_output_shape");
    TryExtractArray(object, "weight_shape", &layer.weight_shape);
    TryExtractBool(object, "requires_im2col", &layer.requires_im2col);
    layers.push_back(std::move(layer));
  }
  TVM_FFI_CHECK(!layers.empty(), ValueError) << "PlanTyphoonSRAM requires at least one planned layer";
  return layers;
}

std::vector<EdgeInfo> ParseEdges(const std::string& plan) {
  std::vector<EdgeInfo> edges;
  for (const std::string& object : SplitTopLevelObjects(ExtractJSONArrayText(plan, "edges"))) {
    edges.push_back(
        EdgeInfo{ExtractInt(object, "src_layer_id"), ExtractInt(object, "dst_layer_id"),
                 ExtractString(object, "edge_kind")});
  }
  return edges;
}

int64_t AlignUp(int64_t value, int64_t alignment) {
  return ((value + alignment - 1) / alignment) * alignment;
}

bool IsResidualEdge(const EdgeInfo& edge) {
  return edge.edge_kind.find("residual") != std::string::npos;
}

int64_t ComputeWeightWorkingSet(const LayerInfo& layer) {
  if (layer.weight_shape.empty()) {
    return 0;
  }
  int64_t out_channels = layer.weight_shape[0];
  int64_t in_channels = layer.weight_shape.size() > 1 ? layer.weight_shape[1] : 1;
  int64_t kernel_h = layer.weight_shape.size() > 2 ? layer.weight_shape[2] : 1;
  int64_t kernel_w = layer.weight_shape.size() > 3 ? layer.weight_shape[3] : 1;
  return AlignUp(std::min<int64_t>(out_channels, 32) * in_channels * kernel_h * kernel_w * 4,
                 kAlignment);
}

int64_t ComputeIm2ColWorkingSet(const LayerInfo& layer) {
  if (!layer.requires_im2col || layer.weight_shape.empty()) {
    return 0;
  }
  int64_t in_channels = layer.weight_shape.size() > 1 ? layer.weight_shape[1] : 1;
  int64_t kernel_h = layer.weight_shape.size() > 2 ? layer.weight_shape[2] : 1;
  int64_t kernel_w = layer.weight_shape.size() > 3 ? layer.weight_shape[3] : 1;
  return AlignUp(8 * in_channels * kernel_h * kernel_w * 4, kAlignment);
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

ffi::String BuildSRAMPlanJSON(const std::vector<LayerInfo>& layers, const std::vector<EdgeInfo>& edges) {
  std::ostringstream os;
  os << "{"
     << "\"matmul_tile\":[64,64,64],"
     << "\"tile\":{"
     << "\"Mt\":64,"
     << "\"Nt\":64,"
     << "\"Kt\":64,"
     << "\"m0\":8,"
     << "\"n0\":8,"
     << "\"k0\":8"
     << "},"
     << "\"regions\":[";
  const auto& regions = GetCanonicalRegions();
  for (size_t i = 0; i < regions.size(); ++i) {
    if (i != 0) {
      os << ",";
    }
    const RegionSpec& region = regions[i];
    os << "{"
       << "\"region_id\":" << region.region_id << ","
       << "\"name\":\"" << region.name << "\","
       << "\"offset\":" << region.offset << ","
       << "\"size\":" << region.size << ","
       << "\"alignment\":" << region.alignment << "}";
  }
  os << "],"
     << "\"live_ranges\":[";

  bool first_live_range = true;
  auto append_live_range = [&](int64_t region_id, int64_t producer, int64_t last_use, const char* role,
                               std::vector<int64_t> consumer_window) {
    if (!first_live_range) {
      os << ",";
    }
    first_live_range = false;
    os << "{"
       << "\"region_id\":" << region_id << ","
       << "\"producer_layer_id\":" << producer << ","
       << "\"last_use_layer_id\":" << last_use << ","
       << "\"role\":\"" << role << "\","
       << "\"consumer_window\":";
    AppendIntArray(os, consumer_window);
    os << ",\"reuse_after_layer_id\":" << last_use << "}";
  };
  append_live_range(0, 0, 3, "ACT", {1, 3});
  append_live_range(1, 4, 7, "ACT", {5, 7});
  for (const EdgeInfo& edge : edges) {
    if (edge.edge_kind == "residual") {
      append_live_range(2, edge.src_layer_id, edge.dst_layer_id, "RESIDUAL",
                        {edge.src_layer_id + 1, edge.dst_layer_id});
    }
  }
  os << "],"
     << "\"layer_tiles\":[";
  for (size_t i = 0; i < layers.size(); ++i) {
    if (i != 0) {
      os << ",";
    }
    const LayerInfo& layer = layers[i];
    std::vector<std::string> reads = {i == 0 ? "act0" : (i % 2 == 0 ? "act1" : "act0")};
    if (!layer.weight_shape.empty()) {
      reads.push_back("wgt0");
    }
    if (layer.requires_im2col) {
      reads.push_back("col0");
    }
    std::vector<std::string> writes = {(i % 2 == 0) ? "act0" : "act1"};
    if (layer.kind == "add") {
      writes.push_back("aux0");
    }
    os << "{"
       << "\"layer_id\":" << layer.layer_id << ","
       << "\"tile_id\":0,"
       << "\"reads\":";
    AppendStringArray(os, reads);
    os << ",\"writes\":";
    AppendStringArray(os, writes);
    os << "}";
  }
  os << "],"
     << "\"peak_live_bytes\":" << kPeakLiveBytes
     << "}";
  return ffi::String(os.str());
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

}  // namespace

namespace transform {

Pass PlanTyphoonSRAM() {
  auto pass_func = [](IRModule mod, PassContext ctx) {
    if (!HasTyphoonPrimFunc(mod)) {
      return mod;
    }

    auto plan_attr = mod->GetAttr<ffi::String>("typhoon_resnet18_plan");
    if (!plan_attr.has_value()) {
      return mod;
    }

    std::string plan = plan_attr.value();
    TVM_FFI_CHECK(plan.find("\"model\":\"resnet18\"") != std::string::npos ||
                      plan.find("\"model\": \"resnet18\"") != std::string::npos,
                  ValueError)
        << "PlanTyphoonSRAM only supports model=resnet18";
    TVM_FFI_CHECK(plan.find("\"recognized_scope\":\"full_graph\"") != std::string::npos ||
                      plan.find("\"recognized_scope\": \"full_graph\"") != std::string::npos,
                  ValueError)
        << "PlanTyphoonSRAM requires recognized_scope=full_graph";
    TVM_FFI_CHECK(plan.find("\"dtype\":\"float32\"") != std::string::npos ||
                      plan.find("\"dtype\": \"float32\"") != std::string::npos,
                  ValueError)
        << "PlanTyphoonSRAM only supports float32 plans";

    std::vector<LayerInfo> layers = ParseLayers(plan);
    std::vector<EdgeInfo> edges = ParseEdges(plan);

    for (const LayerInfo& layer : layers) {
      int64_t weight_tile_bytes = ComputeWeightWorkingSet(layer);
      int64_t col_tile_bytes = ComputeIm2ColWorkingSet(layer);
      TVM_FFI_CHECK_LE(weight_tile_bytes, kWeightRegionSize, ValueError)
          << "PlanTyphoonSRAM layer working set exceeds 1MB SRAM";
      TVM_FFI_CHECK_LE(col_tile_bytes, kColRegionSize, ValueError)
          << "PlanTyphoonSRAM layer working set exceeds 1MB SRAM";
    }

    for (const EdgeInfo& edge : edges) {
      if (IsResidualEdge(edge)) {
        TVM_FFI_CHECK_LT(edge.src_layer_id, edge.dst_layer_id, ValueError)
            << "PlanTyphoonSRAM found invalid residual live range reuse";
      }
    }

    return WithAttr(std::move(mod), "typhoon_sram_plan", BuildSRAMPlanJSON(layers, edges));
  };
  return CreateModulePass(pass_func, 0, "tirx.PlanTyphoonSRAM", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tirx.transform.PlanTyphoonSRAM", PlanTyphoonSRAM);
}

}  // namespace transform

}  // namespace tirx
}  // namespace tvm
