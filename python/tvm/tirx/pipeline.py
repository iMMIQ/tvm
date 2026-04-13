# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# pylint: disable=invalid-name
"""The TIR backend compilation pipeline."""

import tvm
from tvm import tirx


def _is_typhoon_primfunc(func):
    if not isinstance(func, tvm.tirx.PrimFunc):
        return False
    attrs = func.attrs
    if not attrs or "target" not in attrs:
        return False
    return attrs["target"].kind.name == "typhoon"


def _typhoon_primfunc_items(mod):
    return [(gvar, func) for gvar, func in mod.functions.items() if _is_typhoon_primfunc(func)]


def _module_has_typhoon_graph_ir(mod):
    for _, func in mod.functions.items():
        if not _is_typhoon_primfunc(func):
            continue
        body_text = str(func.body)
        if "tirx.typhoon." in body_text or "TVMTyphoon" in body_text:
            return True
    return False


def _run_typhoon_graph_build(mod):
    attrs = mod.attrs or {}
    if "typhoon_graph_plan" not in attrs:
        mod = tirx.transform.IdentifyTyphoonGraph()(mod)
        attrs = mod.attrs or {}
    if "typhoon_graph_plan" not in attrs:
        return None
    return tirx.transform.BuildTyphoonGraph()(mod)


@tvm.transform.module_pass(opt_level=0, name="tirx.MaybeBuildTyphoonGraph")
def _maybe_build_typhoon_graph(mod, _ctx):
    if _module_has_typhoon_graph_ir(mod):
        return mod

    typhoon_funcs = _typhoon_primfunc_items(mod)
    if not typhoon_funcs:
        return mod

    try:
        built = _run_typhoon_graph_build(mod)
    except ValueError:
        built = None
    if built is not None:
        return built

    candidates = []
    for gvar, func in typhoon_funcs:
        try:
            built = _run_typhoon_graph_build(tvm.IRModule({gvar: func}, attrs=mod.attrs))
        except ValueError:
            continue
        if built is not None:
            candidates.append((gvar, built))

    if len(candidates) != 1:
        return mod

    gvar, built = candidates[0]
    helper_name = f"{gvar.name_hint}_typhoon_graph"
    helper_gvar = tvm.ir.GlobalVar(helper_name)
    helper_func = built[gvar].with_attr("global_symbol", helper_name)
    merged = tvm.IRModule(
        {**dict(mod.functions), helper_gvar: helper_func},
        attrs=mod.attrs,
    )
    for key in ("typhoon_graph_plan",):
        if built.attrs and key in built.attrs:
            merged = merged.with_attr(key, built.attrs[key])
    return merged


def finalize_host_passes():  # pylint: disable=unused-argument
    """The default finalization passes for TIR backend."""
    host_pass_list = [
        tirx.transform.LowerTVMBuiltin(),
        tirx.transform.LowerCustomDatatypes(),
        _maybe_build_typhoon_graph,
        tirx.transform.VerifyTyphoonGraph(),
        tirx.transform.LowerTyphoonTaskDeps(),
        tirx.transform.LowerTyphoonSubmitGraph(),
        tirx.transform.LowerIntrin(),
    ]
    return tvm.ir.transform.Sequential(host_pass_list)


def finalize_device_passes():  # pylint: disable=unused-argument
    """The default finalization passes for TIR backend."""
    device_pass_list = [
        tirx.transform.LowerWarpMemory(),
        tirx.transform.Simplify(),
        tirx.transform.LowerCustomDatatypes(),
        tirx.transform.LowerIntrin(),
    ]
    return tvm.ir.transform.Sequential(device_pass_list)


# global map of pre-built pipelines
PIPELINE_MAP = {}


def get_tir_pipeline(name: str | None = None, **kwargs) -> tvm.transform.Pass:
    """Get pre-build pipeline by name

    Parameters
    ----------
    name : Optional[str]
        Name of the pipeline
    """
    if name == "default":
        # for now, defualt to s_tir pipeline
        name = "s_tir"
    if name not in PIPELINE_MAP:
        raise ValueError(
            f"Unknown pre-built pipeline {name},candidates are {list(PIPELINE_MAP.keys())}"
        )
    return PIPELINE_MAP[name](**kwargs)


def get_default_tir_pipeline(
    target: tvm.target.Target,  # pylint: disable=unused-argument
) -> tvm.transform.Pass:
    """Get the default TIR pipeline for the given target."""
    if target.kind.name == "opencl" and "adreno" in target.keys:
        return get_tir_pipeline("adreno")
    else:
        return get_tir_pipeline("s_tir")
