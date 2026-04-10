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
#include <unordered_map>
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
  std::vector<int64_t> secondary_input_shape;
  bool requires_im2col{false};
};

struct EdgeInfo {
  int64_t src_layer_id{0};
  int64_t dst_layer_id{0};
  std::string edge_kind;
};

struct StorageObject {
  int64_t storage_id{0};
  int64_t layer_id{0};
  std::string role;
  std::string region_class;
  int64_t size{0};
  int64_t alignment{0};
  int64_t lifetime_start{0};
  int64_t lifetime_end{0};
};

struct PlacementRegion {
  int64_t region_id{0};
  std::string name;
  std::string region_class;
  int64_t offset{0};
  int64_t size{0};
  int64_t alignment{0};
};

struct StoragePlacement {
  int64_t storage_id{0};
  int64_t region_id{0};
  int64_t offset{0};
};

struct LayerOperandBinding {
  int64_t layer_id{0};
  int64_t input_storage_id{-1};
  int64_t output_storage_id{-1};
  int64_t weight_storage_id{-1};
  int64_t secondary_input_storage_id{-1};
  std::vector<int64_t> workspace_storage_ids;
};

struct StoragePlanState {
  std::vector<StorageObject> objects;
  std::vector<LayerOperandBinding> operand_bindings;
  std::unordered_map<int64_t, int64_t> output_storage_by_layer_id;
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
    TryExtractArray(object, "secondary_input_shape", &layer.secondary_input_shape);
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

int64_t CanonicalRoleSize(const std::string& role) {
  if (role == "input" || role == "output" || role == "secondary_input") {
    return kActRegionSize;
  }
  if (role == "weight") {
    return kWeightRegionSize;
  }
  if (role == "im2col_workspace") {
    return kColRegionSize;
  }
  if (role == "accumulator_workspace") {
    return kAuxRegionSize;
  }
  TVM_FFI_THROW(ValueError) << "PlanTyphoonSRAM does not recognize storage role `" << role << "`";
  return 0;
}

std::string RegionClassForRole(const std::string& role) {
  if (role == "input" || role == "output" || role == "secondary_input") {
    return "activation";
  }
  if (role == "weight") {
    return "weight";
  }
  if (role == "im2col_workspace") {
    return "im2col_workspace";
  }
  if (role == "accumulator_workspace") {
    return "accumulator_workspace";
  }
  TVM_FFI_THROW(ValueError) << "PlanTyphoonSRAM does not recognize storage role `" << role << "`";
  return "";
}

bool MatchesShape(const std::vector<int64_t>& lhs, const std::vector<int64_t>& rhs) { return lhs == rhs; }

std::string InferOutputRole(
    const LayerInfo& layer, const std::unordered_map<int64_t, std::vector<EdgeInfo>>& outgoing_edges_by_src,
    const std::unordered_map<int64_t, LayerInfo>& layers_by_id) {
  auto outgoing_it = outgoing_edges_by_src.find(layer.layer_id);
  if (outgoing_it == outgoing_edges_by_src.end()) {
    return "output";
  }
  for (const EdgeInfo& edge : outgoing_it->second) {
    auto consumer_it = layers_by_id.find(edge.dst_layer_id);
    if (consumer_it == layers_by_id.end()) {
      continue;
    }
    const LayerInfo& consumer = consumer_it->second;
    if (!consumer.weight_shape.empty() &&
        MatchesShape(layer.logical_output_shape, consumer.weight_shape)) {
      return "weight";
    }
  }
  return "output";
}

StoragePlanState BuildStoragePlanState(const std::vector<LayerInfo>& layers,
                                       const std::vector<EdgeInfo>& edges) {
  StoragePlanState state;
  auto add_object = [&](int64_t layer_id, const std::string& role, int64_t lifetime_start,
                        int64_t lifetime_end) -> int64_t {
    int64_t storage_id = static_cast<int64_t>(state.objects.size());
    state.objects.push_back(StorageObject{storage_id, layer_id, role, RegionClassForRole(role),
                                          CanonicalRoleSize(role), kAlignment, lifetime_start,
                                          lifetime_end});
    return storage_id;
  };

  std::unordered_map<int64_t, LayerInfo> layers_by_id;
  std::unordered_map<int64_t, std::vector<EdgeInfo>> incoming_edges_by_dst;
  std::unordered_map<int64_t, std::vector<EdgeInfo>> outgoing_edges_by_src;
  for (const LayerInfo& layer : layers) {
    layers_by_id.emplace(layer.layer_id, layer);
  }
  for (const EdgeInfo& edge : edges) {
    incoming_edges_by_dst[edge.dst_layer_id].push_back(edge);
    outgoing_edges_by_src[edge.src_layer_id].push_back(edge);
  }

  for (const LayerInfo& layer : layers) {
    int64_t lifetime_end = layer.layer_id;
    auto outgoing_it = outgoing_edges_by_src.find(layer.layer_id);
    if (outgoing_it != outgoing_edges_by_src.end()) {
      for (const EdgeInfo& edge : outgoing_it->second) {
        lifetime_end = std::max<int64_t>(lifetime_end, edge.dst_layer_id);
      }
    }
    state.output_storage_by_layer_id[layer.layer_id] =
        add_object(layer.layer_id, InferOutputRole(layer, outgoing_edges_by_src, layers_by_id),
                   layer.layer_id, lifetime_end);
  }

  auto try_bind_internal_input = [&](const LayerInfo& layer, const std::vector<int64_t>& shape,
                                     bool prefer_residual) -> int64_t {
    auto incoming_it = incoming_edges_by_dst.find(layer.layer_id);
    if (incoming_it == incoming_edges_by_dst.end()) {
      return -1;
    }
    auto matches_preference = [&](const EdgeInfo& edge) {
      bool is_residual = IsResidualEdge(edge);
      return prefer_residual ? is_residual : !is_residual;
    };
    for (int pass = 0; pass < 2; ++pass) {
      for (const EdgeInfo& edge : incoming_it->second) {
        if ((pass == 0 && !matches_preference(edge)) || (pass == 1 && matches_preference(edge))) {
          continue;
        }
        auto producer_it = layers_by_id.find(edge.src_layer_id);
        if (producer_it == layers_by_id.end()) {
          continue;
        }
        if (!MatchesShape(producer_it->second.logical_output_shape, shape)) {
          continue;
        }
        auto storage_it = state.output_storage_by_layer_id.find(edge.src_layer_id);
        if (storage_it != state.output_storage_by_layer_id.end()) {
          return storage_it->second;
        }
      }
    }
    return -1;
  };

  auto try_bind_internal_weight = [&](const LayerInfo& layer) -> int64_t {
    if (layer.weight_shape.empty()) {
      return -1;
    }
    auto incoming_it = incoming_edges_by_dst.find(layer.layer_id);
    if (incoming_it == incoming_edges_by_dst.end()) {
      return -1;
    }
    for (const EdgeInfo& edge : incoming_it->second) {
      auto producer_it = layers_by_id.find(edge.src_layer_id);
      if (producer_it == layers_by_id.end()) {
        continue;
      }
      if (!MatchesShape(producer_it->second.logical_output_shape, layer.weight_shape)) {
        continue;
      }
      auto storage_it = state.output_storage_by_layer_id.find(edge.src_layer_id);
      if (storage_it != state.output_storage_by_layer_id.end()) {
        return storage_it->second;
      }
    }
    return -1;
  };

  for (const LayerInfo& layer : layers) {
    LayerOperandBinding binding;
    binding.layer_id = layer.layer_id;
    binding.output_storage_id = state.output_storage_by_layer_id.at(layer.layer_id);

    int64_t input_storage_id = try_bind_internal_input(layer, layer.logical_input_shape, false);
    if (input_storage_id < 0) {
      input_storage_id = add_object(layer.layer_id, "input", layer.layer_id, layer.layer_id);
    }
    binding.input_storage_id = input_storage_id;

    if (layer.kind == "add" || layer.kind == "bias_add" || layer.kind == "residual_add") {
      int64_t secondary_storage_id = -1;
      if (!layer.secondary_input_shape.empty()) {
        secondary_storage_id =
            try_bind_internal_input(layer, layer.secondary_input_shape, true);
      }
      if (secondary_storage_id < 0) {
        secondary_storage_id =
            add_object(layer.layer_id, "secondary_input", layer.layer_id, layer.layer_id);
      }
      binding.secondary_input_storage_id = secondary_storage_id;
    }
    if (!layer.weight_shape.empty()) {
      int64_t weight_storage_id = try_bind_internal_weight(layer);
      if (weight_storage_id < 0) {
        weight_storage_id = add_object(layer.layer_id, "weight", layer.layer_id, layer.layer_id);
      }
      binding.weight_storage_id = weight_storage_id;
    }
    if (layer.requires_im2col) {
      binding.workspace_storage_ids.push_back(
          add_object(layer.layer_id, "im2col_workspace", layer.layer_id, layer.layer_id));
      binding.workspace_storage_ids.push_back(
          add_object(layer.layer_id, "accumulator_workspace", layer.layer_id, layer.layer_id));
    }
    state.operand_bindings.push_back(std::move(binding));
  }
  return state;
}

std::vector<StoragePlacement> BuildStoragePlacements(const std::vector<StorageObject>& objects,
                                                     std::vector<PlacementRegion>* regions) {
  struct ActiveAssignment {
    int64_t lifetime_end{0};
    int64_t region_id{0};
  };
  struct RegionState {
    PlacementRegion region;
    std::vector<ActiveAssignment> active;
  };

  std::unordered_map<std::string, std::vector<RegionState>> states_by_class;
  std::vector<StoragePlacement> placements;
  int64_t next_region_id = 0;

  std::vector<StorageObject> sorted = objects;
  std::sort(sorted.begin(), sorted.end(),
            [](const StorageObject& lhs, const StorageObject& rhs) {
              if (lhs.lifetime_start != rhs.lifetime_start) {
                return lhs.lifetime_start < rhs.lifetime_start;
              }
              if (lhs.lifetime_end != rhs.lifetime_end) {
                return lhs.lifetime_end < rhs.lifetime_end;
              }
              return lhs.storage_id < rhs.storage_id;
            });

  for (const StorageObject& object : sorted) {
    auto& states = states_by_class[object.region_class];
    RegionState* chosen = nullptr;
    for (auto& state : states) {
      state.active.erase(std::remove_if(state.active.begin(), state.active.end(),
                                        [&](const ActiveAssignment& active) {
                                          return active.lifetime_end < object.lifetime_start;
                                        }),
                         state.active.end());
      if (state.active.empty()) {
        chosen = &state;
        break;
      }
    }
    if (chosen == nullptr) {
      PlacementRegion region;
      region.region_id = next_region_id++;
      region.name = object.region_class + "_slot" + std::to_string(states.size());
      region.region_class = object.region_class;
      region.offset = 0;
      region.size = object.size;
      region.alignment = object.alignment;
      states.push_back(RegionState{region, {}});
      chosen = &states.back();
    }
    chosen->region.size = std::max(chosen->region.size, object.size);
    chosen->region.alignment = std::max(chosen->region.alignment, object.alignment);
    chosen->active.push_back(ActiveAssignment{object.lifetime_end, chosen->region.region_id});
    placements.push_back(StoragePlacement{object.storage_id, chosen->region.region_id, 0});
  }

  int64_t next_offset = 0;
  std::vector<std::string> region_classes;
  region_classes.reserve(states_by_class.size());
  for (const auto& [region_class, states] : states_by_class) {
    region_classes.push_back(region_class);
  }
  std::sort(region_classes.begin(), region_classes.end());
  for (const std::string& region_class : region_classes) {
    auto& states = states_by_class.at(region_class);
    for (auto& state : states) {
      state.region.offset = AlignUp(next_offset, state.region.alignment);
      next_offset = state.region.offset + state.region.size;
      regions->push_back(state.region);
    }
  }
  std::sort(regions->begin(), regions->end(),
            [](const PlacementRegion& lhs, const PlacementRegion& rhs) {
              return lhs.region_id < rhs.region_id;
            });
  std::sort(placements.begin(), placements.end(),
            [](const StoragePlacement& lhs, const StoragePlacement& rhs) {
              return lhs.storage_id < rhs.storage_id;
            });
  return placements;
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
  StoragePlanState storage_plan = BuildStoragePlanState(layers, edges);
  const std::vector<StorageObject>& storage_objects = storage_plan.objects;
  std::vector<PlacementRegion> placement_regions;
  std::vector<StoragePlacement> placements = BuildStoragePlacements(storage_objects, &placement_regions);
  const std::vector<LayerOperandBinding>& operand_bindings = storage_plan.operand_bindings;
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
     << "\"storage_objects\":[";
  for (size_t i = 0; i < storage_objects.size(); ++i) {
    if (i != 0) {
      os << ",";
    }
    const StorageObject& object = storage_objects[i];
    os << "{"
       << "\"storage_id\":" << object.storage_id << ","
       << "\"layer_id\":" << object.layer_id << ","
       << "\"role\":\"" << object.role << "\","
       << "\"region_class\":\"" << object.region_class << "\","
       << "\"size\":" << object.size << ","
       << "\"alignment\":" << object.alignment << ","
       << "\"lifetime_start\":" << object.lifetime_start << ","
       << "\"lifetime_end\":" << object.lifetime_end << "}";
  }
  os << "],"
     << "\"placement_regions\":[";
  for (size_t i = 0; i < placement_regions.size(); ++i) {
    if (i != 0) {
      os << ",";
    }
    const PlacementRegion& region = placement_regions[i];
    os << "{"
       << "\"region_id\":" << region.region_id << ","
       << "\"name\":\"" << region.name << "\","
       << "\"region_class\":\"" << region.region_class << "\","
       << "\"offset\":" << region.offset << ","
       << "\"size\":" << region.size << ","
       << "\"alignment\":" << region.alignment << "}";
  }
  os << "],"
     << "\"placements\":[";
  for (size_t i = 0; i < placements.size(); ++i) {
    if (i != 0) {
      os << ",";
    }
    const StoragePlacement& placement = placements[i];
    os << "{"
       << "\"storage_id\":" << placement.storage_id << ","
       << "\"region_id\":" << placement.region_id << ","
       << "\"offset\":" << placement.offset << "}";
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
     << "\"layer_operands\":[";
  for (size_t i = 0; i < operand_bindings.size(); ++i) {
    if (i != 0) {
      os << ",";
    }
    const LayerOperandBinding& binding = operand_bindings[i];
    os << "{"
       << "\"layer_id\":" << binding.layer_id << ","
       << "\"input_storage_id\":" << binding.input_storage_id << ","
       << "\"output_storage_id\":" << binding.output_storage_id;
    if (binding.weight_storage_id >= 0) {
      os << ",\"weight_storage_id\":" << binding.weight_storage_id;
    }
    if (binding.secondary_input_storage_id >= 0) {
      os << ",\"secondary_input_storage_id\":" << binding.secondary_input_storage_id;
    }
    os << ",\"workspace_storage_ids\":";
    AppendIntArray(os, binding.workspace_storage_ids);
    os << "}";
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
