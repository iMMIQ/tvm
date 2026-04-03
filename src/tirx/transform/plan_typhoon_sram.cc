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

#include <regex>
#include <sstream>
#include <string>
#include <vector>

namespace tvm {
namespace tirx {

namespace {

std::string GetRequiredModuleAttr(const IRModule& mod, const char* key) {
  auto attr = mod->GetAttr<ffi::String>(key);
  TVM_FFI_CHECK(attr.has_value(), ValueError)
      << "PlanTyphoonSRAM requires module attr `" << key << "`";
  return attr.value();
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
      << "PlanTyphoonSRAM could not find `" << key << "` in typhoon_resnet18_plan";
  return ParseIntList(match[1].str());
}

bool ExtractBool(const std::string& json, const char* key) {
  std::regex pattern(std::string("\"") + key + "\"\\s*:\\s*(true|false)");
  std::smatch match;
  TVM_FFI_CHECK(std::regex_search(json, match, pattern), ValueError)
      << "PlanTyphoonSRAM could not find `" << key << "` in typhoon_resnet18_plan";
  return match[1].str() == "true";
}

ffi::String BuildSRAMPlanJSON() {
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
     << "\"regions\":["
     << "{\"region_id\":0,\"name\":\"act0\",\"offset\":0,\"size\":262144,\"alignment\":64},"
     << "{\"region_id\":1,\"name\":\"act1\",\"offset\":262144,\"size\":262144,\"alignment\":64},"
     << "{\"region_id\":2,\"name\":\"wgt0\",\"offset\":524288,\"size\":65536,\"alignment\":64},"
     << "{\"region_id\":3,\"name\":\"wgt1\",\"offset\":589824,\"size\":65536,\"alignment\":64},"
     << "{\"region_id\":4,\"name\":\"col0\",\"offset\":655360,\"size\":131072,\"alignment\":64},"
     << "{\"region_id\":5,\"name\":\"aux0\",\"offset\":786432,\"size\":65536,\"alignment\":64}"
     << "],"
     << "\"layer_tiles\":[{"
     << "\"layer_id\":0,"
     << "\"tile_id\":0,"
     << "\"reads\":[\"act0\",\"wgt0\"],"
     << "\"writes\":[\"col0\",\"aux0\"]"
     << "}]"
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

    std::string plan = GetRequiredModuleAttr(mod, "typhoon_resnet18_plan");
    TVM_FFI_CHECK(plan.find("\"model\":\"resnet18\"") != std::string::npos ||
                      plan.find("\"model\": \"resnet18\"") != std::string::npos,
                  ValueError)
        << "PlanTyphoonSRAM only supports model=resnet18";
    TVM_FFI_CHECK(plan.find("\"dtype\":\"float32\"") != std::string::npos ||
                      plan.find("\"dtype\": \"float32\"") != std::string::npos,
                  ValueError)
        << "PlanTyphoonSRAM only supports float32 plans";

    std::vector<int64_t> input_shape = ExtractArray(plan, "logical_input_shape");
    std::vector<int64_t> output_shape = ExtractArray(plan, "logical_output_shape");
    std::vector<int64_t> weight_shape = ExtractArray(plan, "weight_shape");
    bool requires_im2col = ExtractBool(plan, "requires_im2col");
    int64_t input_channels = input_shape.size() > 1 ? input_shape[1] : 0;
    int64_t output_tile_bytes = 64 * 64 * 4;
    int64_t weight_tile_bytes = 64 * input_channels * weight_shape[2] * weight_shape[3] * 4;
    int64_t col_tile_bytes =
        requires_im2col ? 64 * input_channels * weight_shape[2] * weight_shape[3] * 4 : 0;
    int64_t aux_tile_bytes = output_tile_bytes;

    TVM_FFI_CHECK_LE(weight_tile_bytes, 65536, ValueError)
        << "PlanTyphoonSRAM layer working set exceeds 1MB SRAM";
    TVM_FFI_CHECK_LE(col_tile_bytes, 131072, ValueError)
        << "PlanTyphoonSRAM layer working set exceeds 1MB SRAM";
    TVM_FFI_CHECK_LE(aux_tile_bytes, 65536, ValueError)
        << "PlanTyphoonSRAM layer working set exceeds 1MB SRAM";
    TVM_FFI_CHECK_LE(output_tile_bytes, 262144, ValueError)
        << "PlanTyphoonSRAM layer working set exceeds 1MB SRAM";

    return WithAttr(std::move(mod), "typhoon_sram_plan", BuildSRAMPlanJSON());
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
