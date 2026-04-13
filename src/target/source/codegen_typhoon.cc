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
 * \file codegen_typhoon.cc
 * \brief Typhoon host-executable codegen entrypoint.
 */
#include "codegen_c_host.h"

#include <algorithm>
#include <unordered_set>

#include <tvm/ffi/reflection/registry.h>

namespace tvm {
namespace codegen {

namespace {

class CodeGenTyphoonHost final : public CodeGenCHost {
 public:
  void GenerateForwardFunctionDeclarations(ffi::String global_symbol,
                                           const ffi::Array<Type>& arg_types,
                                           const Type& ret_type) final {
    if (EmitTyphoonRuntimeDeclaration(global_symbol)) {
      return;
    }
    CodeGenCHost::GenerateForwardFunctionDeclarations(global_symbol, arg_types, ret_type);
  }

 private:
  bool EmitTyphoonRuntimeDeclaration(ffi::String global_symbol) {
    if (!emitted_symbols_.insert(global_symbol).second) {
      return true;
    }

    const char* signature = nullptr;
    if (global_symbol == "TVMTyphoonGraphBegin") {
      signature = "int32_t TVMTyphoonGraphBegin(int32_t)";
    } else if (global_symbol == "TVMTyphoonDeclareRegion") {
      signature = "int32_t TVMTyphoonDeclareRegion(int32_t, int32_t, int64_t, int64_t, int64_t, "
                  "int32_t, const char*)";
    } else if (global_symbol == "TVMTyphoonAddDMATask") {
      signature = "int32_t TVMTyphoonAddDMATask(int32_t, int32_t, int32_t, void*, int64_t, "
                  "int32_t, int64_t, int64_t, int32_t, void*)";
    } else if (global_symbol == "TVMTyphoonAddBatchedDMATask") {
      signature =
          "int32_t TVMTyphoonAddBatchedDMATask(int32_t, int32_t, int32_t, void*, int64_t, "
          "int64_t, int32_t, int64_t, int64_t, int64_t, int64_t, int32_t, void*)";
    } else if (global_symbol == "TVMTyphoonAddMatmulTask") {
      signature = "int32_t TVMTyphoonAddMatmulTask(int32_t, int32_t, int32_t, int32_t, int32_t, "
                  "int64_t, int64_t, int64_t, int32_t, int32_t, int32_t, void*)";
    } else if (global_symbol == "TVMTyphoonAddVectorTask") {
      signature = "int32_t TVMTyphoonAddVectorTask(int32_t, int32_t, int32_t, int32_t, int32_t, "
                  "int32_t, int64_t, int32_t, int32_t, void*, int32_t, void*)";
    } else if (global_symbol == "TVMTyphoonAddReshapeTask") {
      signature = "int32_t TVMTyphoonAddReshapeTask(int32_t, int32_t, int32_t, int32_t, int64_t, "
                  "int32_t, int32_t, void*, int32_t, void*)";
    } else if (global_symbol == "TVMTyphoonCaptureCall") {
      signature = "int32_t TVMTyphoonCaptureCall(int32_t, int32_t, void*, void*, void*)";
    } else if (global_symbol == "TVMTyphoonCaptureCallAt") {
      signature = "int32_t TVMTyphoonCaptureCallAt(int32_t, int32_t, int32_t, void*, void*, "
                  "void*)";
    } else if (global_symbol == "TVMTyphoonCaptureCallPlanned") {
      signature = "int32_t TVMTyphoonCaptureCallPlanned(int32_t, const char*, int32_t, void*, "
                  "void*, void*)";
    } else if (global_symbol == "TVMTyphoonCapturePackedArgs") {
      signature = "int32_t TVMTyphoonCapturePackedArgs(int32_t, void*, int32_t)";
    } else if (global_symbol == "TVMTyphoonCapturePackedArgsAt") {
      signature = "int32_t TVMTyphoonCapturePackedArgsAt(int32_t, int32_t, void*, int32_t)";
    } else if (global_symbol == "TVMTyphoonCapturePackedArgsPlanned") {
      signature = "int32_t TVMTyphoonCapturePackedArgsPlanned(int32_t, const char*, void*, "
                  "int32_t)";
    } else if (global_symbol == "TVMTyphoonGetCapturedHandle") {
      signature = "void* TVMTyphoonGetCapturedHandle(int32_t, int32_t, int32_t)";
    } else if (global_symbol == "TVMTyphoonSubmitGraph") {
      signature = "int32_t TVMTyphoonSubmitGraph(int32_t)";
    } else if (global_symbol == "TVMTyphoonWaitGraph") {
      signature = "int32_t TVMTyphoonWaitGraph(int32_t)";
    } else {
      emitted_symbols_.erase(global_symbol);
      return false;
    }

    PrintFuncPrefix(fwd_decl_stream);
    fwd_decl_stream << signature << ";\n";
    return true;
  }

  std::unordered_set<std::string> emitted_symbols_;
};

}  // namespace

ffi::Module BuildTyphoon(IRModule mod, Target target) {
  bool output_ssa = false;
  bool emit_asserts = true;
  bool emit_fwd_func_decl = true;

  std::unordered_set<std::string> devices;
  if (mod->GetAttr<ffi::Map<GlobalVar, ffi::String>>("device_contexts") != nullptr) {
    ffi::Map<GlobalVar, ffi::String> device_contexts =
        mod->GetAttr<ffi::Map<GlobalVar, ffi::String>>("device_contexts").value();
    for (auto const& context : device_contexts) {
      devices.insert(context.second.data());
    }
  }

  CodeGenTyphoonHost cg;
  cg.Init(output_ssa, emit_asserts, emit_fwd_func_decl, target->str(), devices);
  cg.SetConstantsByteAlignment(target->GetAttr<Integer>("constants-byte-alignment").value_or(16));

  auto is_aot_executor_fn = [](const PrimFunc& func) -> bool {
    return func->GetAttr<Bool>("runner_function", Bool(false)).value();
  };

  std::vector<std::pair<GlobalVar, PrimFunc>> funcs;
  for (auto [gvar, base_func] : mod->functions) {
    TVM_FFI_ICHECK(base_func->IsInstance<PrimFuncNode>())
        << "BuildTyphoon: Can only take PrimFunc";
    funcs.push_back({gvar, Downcast<PrimFunc>(base_func)});
  }

  auto sort_key = [&is_aot_executor_fn](const auto& kv) {
    return std::tuple{is_aot_executor_fn(kv.second), kv.first->name_hint};
  };
  std::sort(funcs.begin(), funcs.end(), [&sort_key](const auto& lhs, const auto& rhs) {
    return sort_key(lhs) < sort_key(rhs);
  });

  for (const auto& [gvar, prim_func] : funcs) {
    cg.DeclareFunction(gvar, prim_func);
  }
  for (const auto& [gvar, prim_func] : funcs) {
    cg.AddFunction(gvar, prim_func, emit_fwd_func_decl);
  }

  return CSourceModuleCreate(cg.Finish(), "c", cg.GetFunctionNames());
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("target.build.typhoon", BuildTyphoon);
}

}  // namespace codegen
}  // namespace tvm
