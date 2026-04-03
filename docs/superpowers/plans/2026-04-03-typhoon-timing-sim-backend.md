# Typhoon Timing-Simulation Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the first-stage `typhoon` backend for TVM: a single-core, task-DAG, timing-simulation backend with explicit SRAM regions, deterministic scheduling, runtime validation, and a temporary empirical task-level cost model.

**Architecture:** The compiler side introduces a new `typhoon` target, Typhoon-specific TIR graph-building ops, Python helper APIs, and target-specific intrinsic lowering into a host-executable runtime ABI. The runtime side implements a graph builder, validator, scheduler, SRAM model, trace collector, and empirical latency estimator; generated C code calls this simulator instead of real hardware.

**Tech Stack:** C++ (`target`, `tirx`, `runtime`), Python (`tvm.tirx` helpers and tests), TVM TIR/TIRx lowering, host C codegen, `pytest`

---

## Implementation Notes

- Follow `@superpowers:test-driven-development` for every task: write the smallest failing test first, then add the minimal code to pass it.
- Use `@superpowers:verification-before-completion` before claiming any task is done.
- Do not add Relax/Relay lowering or TVMScript parser syntax in this phase.
- Keep Typhoon SRAM references explicit via `region_id`; do not add generic `alloc`-style APIs.
- Keep the runtime deterministic: the scheduler tie-break order is `task_id`, then creation order.
- Canonical Typhoon TIR op names are `tirx.typhoon.*`; register them with
  `TVM_TIR_REGISTER_OP("typhoon....")` so the generated op names stay consistent with the Python
  helper surface and LowerIntrin registration.

## ABI Contract

Use this exact phase-1 contract so the helper API, lowering, and runtime ABI all agree.

### Graph / Region Ops

- `region_decl(region_id, offset, size, alignment, preinitialized, tag)`
- `submit_graph(graph_id)`
- `wait_graph(graph_id)`

`preinitialized` is a boolean flag. It exists only to represent SRAM inputs that are already valid
at graph start.

### DMA Op

- `task_dma(task_id, direction, global_handle, global_byte_offset, sram_region_id, bytes, deps)`

Representation rules:

- `direction = 0` means `global_to_sram`
- `direction = 1` means `sram_to_global`
- `global_handle` is the lowered `Buffer.data`/pointer argument
- `global_byte_offset` is the byte offset inside that buffer
- `sram_region_id` is the SRAM endpoint
- `deps` is lowered as `num_deps, dep_0, ..., dep_n`

### Compute Ops

- `task_matmul(task_id, a_region_id, b_region_id, c_region_id, m, n, k, dtype_code, layout_code, deps)`
- `task_vector(task_id, op_code, in0_region_id, in1_region_id, out_region_id, elem_count, dtype_code, deps)`
- `task_reshape(task_id, in_region_id, out_region_id, elem_count, transform_code, deps)`

Use small integer enum codes for `dtype`, `layout`, `vector op`, and `reshape transform` in
phase 1.

### Lowered Runtime Calls

- `TVMTyphoonGraphBegin(int32_t graph_id)`
- `TVMTyphoonDeclareRegion(int32_t graph_id, int32_t region_id, int64_t offset, int64_t size, int64_t alignment, int32_t preinitialized, const char* tag)`
- `TVMTyphoonAddDMATask(int32_t graph_id, int32_t task_id, int32_t direction, void* global_handle, int64_t global_byte_offset, int32_t sram_region_id, int64_t bytes, int32_t num_deps, const int32_t* dep_ids)`
- `TVMTyphoonAddMatmulTask(int32_t graph_id, int32_t task_id, int32_t a_region_id, int32_t b_region_id, int32_t c_region_id, int64_t m, int64_t n, int64_t k, int32_t dtype_code, int32_t layout_code, int32_t num_deps, const int32_t* dep_ids)`
- `TVMTyphoonAddVectorTask(int32_t graph_id, int32_t task_id, int32_t op_code, int32_t in0_region_id, int32_t in1_region_id, int32_t out_region_id, int64_t elem_count, int32_t dtype_code, int32_t num_deps, const int32_t* dep_ids)`
- `TVMTyphoonAddReshapeTask(int32_t graph_id, int32_t task_id, int32_t in_region_id, int32_t out_region_id, int64_t elem_count, int32_t transform_code, int32_t num_deps, const int32_t* dep_ids)`
- `TVMTyphoonSubmitGraph(int32_t graph_id)`
- `TVMTyphoonWaitGraph(int32_t graph_id)`

## File Structure

### Compiler and Build Integration

- Modify: `CMakeLists.txt`
  Add `src/runtime/typhoon/*.cc` to the runtime source glob so `tvm_runtime` links the simulator.
- Modify: `src/target/target_kind.cc`
  Register `typhoon` target attributes such as `sram_size`, future-facing `num_cores`, and debug flags.
- Modify: `src/target/source/codegen_c_host.h`
  Expose the reusable host-C build entry used by `target.build.c` so `target.build.typhoon` can delegate cleanly.
- Modify: `src/target/source/codegen_c_host.cc`
  Keep the existing `c` target behavior unchanged while making the shared builder callable from Typhoon codegen.
- Create: `src/target/source/codegen_typhoon.cc`
  Register `target.build.typhoon`; reuse host C codegen and ensure forward declarations are emitted for Typhoon runtime entrypoints.
- Create: `src/target/source/intrin_rule_typhoon.cc`
  Register `typhoon.FLowerIntrinsic` rules that lower Typhoon graph ops into `call_extern` runtime ABI calls.

### TIR/TIRx Surface

- Create: `src/tirx/op/typhoon.cc`
  Register `tirx.typhoon.*` graph-building ops and their effect metadata.
- Create: `src/tirx/transform/verify_typhoon_graph.cc`
  Implement compile-time structural validation for Typhoon graphs before LowerIntrin.
- Create: `python/tvm/tirx/typhoon.py`
  Provide Python helpers for `region_decl`, `task_dma`, `task_matmul`, `task_vector`, `task_reshape`, `submit_graph`, and `wait_graph`.
- Modify: `python/tvm/tirx/__init__.py`
  Export the Typhoon helper module from the public `tvm.tirx` package.
- Modify: `python/tvm/tirx/transform/transform.py`
  Expose `VerifyTyphoonGraph()` to Python.
- Modify: `python/tvm/tirx/pipeline.py`
  Add a Typhoon-specific default/finalization path that runs compile-time validation before LowerIntrin.

### Runtime Simulator

- Create: `src/runtime/typhoon/typhoon_types.h`
  Shared structs/enums for regions, task descriptors, trace records, global endpoints, and scheduler state.
- Create: `src/runtime/typhoon/typhoon_graph.h`
  Internal graph container and validation entrypoints.
- Create: `src/runtime/typhoon/typhoon_graph.cc`
  Graph builder implementation, SRAM region table, initialization tracking, and compile/runtime consistency checks.
- Create: `src/runtime/typhoon/typhoon_cost_model.h`
  Cost model interface and defaults.
- Create: `src/runtime/typhoon/typhoon_cost_model.cc`
  Development-only empirical latency estimator for DMA, matmul, vector, and reshape.
- Create: `src/runtime/typhoon/typhoon_scheduler.h`
  Scheduler and timeline simulation interfaces.
- Create: `src/runtime/typhoon/typhoon_scheduler.cc`
  Deterministic single-core scheduler, resource occupancy tracking, and trace generation.
- Create: `src/runtime/typhoon/typhoon_runtime.cc`
  Exported C ABI functions used by lowered Typhoon code plus optional test/debug accessors for last trace retrieval.

### Tests

- Create: `tests/python/target/test_target_typhoon.py`
  Target registration and build-hook smoke tests.
- Create: `tests/python/tirx-base/test_tir_typhoon_ops.py`
  Python helper/API and graph-op construction tests.
- Create: `tests/python/tirx-transform/test_tir_transform_typhoon_lower_intrin.py`
  LowerIntrin coverage for Typhoon graph ops.
- Create: `tests/python/tirx-transform/test_tir_transform_verify_typhoon_graph.py`
  Compiler-side validation coverage for Typhoon graph structure and SRAM rules.
- Create: `tests/python/runtime/test_runtime_typhoon.py`
  Runtime validation, deterministic scheduling, trace fields, and latency-model unit coverage.
- Create: `tests/python/codegen/test_target_codegen_typhoon.py`
  End-to-end Typhoon build/execute tests, including invalid SRAM usage and overlap scenarios.

## Task 1: Register the `typhoon` target and build hook

**Files:**
- Modify: `src/target/target_kind.cc`
- Modify: `src/target/source/codegen_c_host.h`
- Modify: `src/target/source/codegen_c_host.cc`
- Create: `src/target/source/codegen_typhoon.cc`
- Test: `tests/python/target/test_target_typhoon.py`

- [ ] **Step 1: Write the failing target registration test**

```python
import tvm


def test_target_typhoon_registration():
    tgt = tvm.target.Target("typhoon -sram_size=1048576")
    assert tgt.kind.name == "typhoon"
    assert int(tgt.attrs["sram_size"]) == 1048576
    assert tvm.get_global_func("target.build.typhoon", allow_missing=True) is not None
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/python/target/test_target_typhoon.py::test_target_typhoon_registration -v`
Expected: FAIL because `typhoon` target kind and/or `target.build.typhoon` are not registered.

- [ ] **Step 3: Implement the target registration and build delegate**

```cc
TVM_REGISTER_TARGET_KIND("typhoon", kDLExtDev)
    .add_attr_option<int64_t>("sram_size", refl::DefaultValue(1048576))
    .add_attr_option<int64_t>("num_cores", refl::DefaultValue(1))
    .set_default_keys({"typhoon", "ext_dev"});

ffi::Module BuildTyphoon(IRModule mod, Target target) {
  return BuildCHost(mod, target);
}
```

Expose `BuildCHost(IRModule, Target)` from `codegen_c_host.h` instead of keeping it file-local.

- [ ] **Step 4: Rebuild the C++ targets**

Run: `cmake --build build -j8 --target tvm_runtime tvm`
Expected: build succeeds and `codegen_typhoon.cc` links without duplicate-symbol issues.

- [ ] **Step 5: Run the target test to verify it passes**

Run: `python -m pytest tests/python/target/test_target_typhoon.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/target/target_kind.cc src/target/source/codegen_c_host.h src/target/source/codegen_c_host.cc src/target/source/codegen_typhoon.cc tests/python/target/test_target_typhoon.py
git commit -m "feat: register typhoon target"
```

## Task 2: Add Typhoon graph ops and Python helper APIs

**Files:**
- Create: `src/tirx/op/typhoon.cc`
- Create: `python/tvm/tirx/typhoon.py`
- Modify: `python/tvm/tirx/__init__.py`
- Test: `tests/python/tirx-base/test_tir_typhoon_ops.py`

- [ ] **Step 1: Write the failing helper/API test**

```python
import tvm


def test_typhoon_region_and_task_builders():
    region = tvm.tirx.typhoon.region_decl(0, 0, 4096, 64, 0, "A")
    task = tvm.tirx.typhoon.task_matmul(1, 0, 1, 2, 64, 64, 64, 1, 0, [])
    assert region.value.op.name == "tirx.typhoon.region_decl"
    assert task.value.op.name == "tirx.typhoon.task_matmul"
```

- [ ] **Step 2: Run the helper test to verify it fails**

Run: `python -m pytest tests/python/tirx-base/test_tir_typhoon_ops.py::test_typhoon_region_and_task_builders -v`
Expected: FAIL because `tvm.tirx.typhoon` and the Typhoon ops do not exist.

- [ ] **Step 3: Register Typhoon ops and implement Python wrappers**

```cc
TVM_TIR_REGISTER_OP("typhoon.region_decl")
    .set_num_inputs(6)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque));
```

```python
def region_decl(region_id, offset, size, alignment, preinitialized, tag, span=None):
    call = tvm.tirx.call_intrin(
        "int32",
        "tirx.typhoon.region_decl",
        region_id,
        offset,
        size,
        alignment,
        preinitialized,
        tag,
        span=span,
    )
    return tvm.tirx.Evaluate(call)
```

Do the same for `task_dma`, `task_matmul`, `task_vector`, `task_reshape`, `submit_graph`, and
`wait_graph`. Keep task operands explicit; SRAM operands are `region_id`, not raw offsets.
Use `TVM_TIR_REGISTER_OP("typhoon....")` consistently for all Typhoon ops so the canonical public
names remain `tirx.typhoon.*`.

- [ ] **Step 4: Rebuild after adding the new TIR op registration**

Run: `cmake --build build -j8 --target tvm_runtime tvm`
Expected: build succeeds and the new op-registration TU is linked.

- [ ] **Step 5: Run the helper test to verify it passes**

Run: `python -m pytest tests/python/tirx-base/test_tir_typhoon_ops.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/tirx/op/typhoon.cc python/tvm/tirx/typhoon.py python/tvm/tirx/__init__.py tests/python/tirx-base/test_tir_typhoon_ops.py
git commit -m "feat: add typhoon tirx graph helpers"
```

## Task 3: Lower Typhoon graph ops to the runtime ABI

**Files:**
- Create: `src/target/source/intrin_rule_typhoon.cc`
- Modify: `python/tvm/tirx/typhoon.py`
- Test: `tests/python/tirx-transform/test_tir_transform_typhoon_lower_intrin.py`

- [ ] **Step 1: Write the failing LowerIntrin test**

```python
import tvm


def test_lower_typhoon_region_decl_to_runtime_call():
    stmt = tvm.tirx.typhoon.region_decl(0, 0, 4096, 64, 0, "A")
    func = tvm.tirx.PrimFunc([], stmt).with_attr("target", tvm.target.Target("typhoon"))
    mod = tvm.IRModule.from_expr(func)
    lowered = tvm.tirx.transform.LowerIntrin()(mod)["main"].body
    assert "TVMTyphoonDeclareRegion" in str(lowered)
```

- [ ] **Step 2: Run the lowering test to verify it fails**

Run: `python -m pytest tests/python/tirx-transform/test_tir_transform_typhoon_lower_intrin.py::test_lower_typhoon_region_decl_to_runtime_call -v`
Expected: FAIL because `typhoon.FLowerIntrinsic` is not defined.

- [ ] **Step 3: Implement target-specific lowering rules**

```cc
TVM_REGISTER_OP("tirx.typhoon.region_decl").set_attr<FLowerIntrinsic>(
    "typhoon.FLowerIntrinsic", LowerToTyphoonExternCall("TVMTyphoonDeclareRegion"));
```

Lower each Typhoon graph op to a stable runtime ABI call:

- `TVMTyphoonGraphBegin`
- `TVMTyphoonDeclareRegion`
- `TVMTyphoonAddDMATask`
- `TVMTyphoonAddMatmulTask`
- `TVMTyphoonAddVectorTask`
- `TVMTyphoonAddReshapeTask`
- `TVMTyphoonSubmitGraph`
- `TVMTyphoonWaitGraph`

Use flattened arguments first; do not introduce a complicated serialization format in this phase.
Use the ABI Contract above exactly. For dependency lists, lower flat dependency IDs into a small
stack array and pass `(num_deps, dep_ids_ptr)` to the runtime call. For DMA, lower the global
endpoint as `(void* global_handle, int64_t global_byte_offset)` plus direction and `sram_region_id`.
Lower `tirx.typhoon.submit_graph(graph_id)` to `TVMTyphoonGraphBegin(graph_id)` immediately
followed by `TVMTyphoonSubmitGraph(graph_id)`, and do not emit `GraphBegin` anywhere else.
Implement `LowerToTyphoonExternCall` locally inside `src/target/source/intrin_rule_typhoon.cc` as
a small helper/lambda that rewrites the Typhoon op into `tirx::Call(..., tirx::builtin::call_extern(), ...)`
with the right runtime C symbol name and flattened arguments.

- [ ] **Step 4: Rebuild and run the lowering test**

Run: `cmake --build build -j8 --target tvm_runtime tvm && python -m pytest tests/python/tirx-transform/test_tir_transform_typhoon_lower_intrin.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/target/source/intrin_rule_typhoon.cc python/tvm/tirx/typhoon.py tests/python/tirx-transform/test_tir_transform_typhoon_lower_intrin.py
git commit -m "feat: lower typhoon graph ops to runtime abi"
```

## Task 4: Add compiler-side Typhoon graph validation

**Files:**
- Create: `src/tirx/transform/verify_typhoon_graph.cc`
- Modify: `python/tvm/tirx/transform/transform.py`
- Modify: `python/tvm/tirx/pipeline.py`
- Test: `tests/python/tirx-transform/test_tir_transform_verify_typhoon_graph.py`

- [ ] **Step 1: Write the failing compile-time validation tests**

```python
import pytest
import tvm


def test_verify_typhoon_graph_rejects_overlapping_regions():
    with pytest.raises(tvm.error.TVMError, match="overlap"):
        tvm.tirx.transform.VerifyTyphoonGraph()(make_invalid_typhoon_mod(overlap=True))


def test_verify_typhoon_graph_rejects_cycles():
    with pytest.raises(tvm.error.TVMError, match="cycle"):
        tvm.tirx.transform.VerifyTyphoonGraph()(make_invalid_typhoon_mod(cycle=True))


def test_verify_typhoon_graph_rejects_read_before_init():
    with pytest.raises(tvm.error.TVMError, match="initialized"):
        tvm.tirx.transform.VerifyTyphoonGraph()(make_invalid_typhoon_mod(read_before_init=True))


def test_verify_typhoon_graph_accepts_preinitialized_region_reads():
    tvm.tirx.transform.VerifyTyphoonGraph()(make_valid_typhoon_mod(preinitialized_input=True))
```

- [ ] **Step 2: Run the compile-time validation tests to verify they fail**

Run: `python -m pytest tests/python/tirx-transform/test_tir_transform_verify_typhoon_graph.py -v`
Expected: FAIL because `VerifyTyphoonGraph` does not exist.

- [ ] **Step 3: Implement the validation pass and pipeline hook**

```cc
Pass VerifyTyphoonGraph() {
  return CreateModulePass(pass_func, 0, "tirx.VerifyTyphoonGraph", {});
}
```

The pass must check at compile time:

- region bounds, alignment, and overlap
- duplicate `region_id` and duplicate `task_id`
- unknown dependency references
- dependency cycles
- compute ops only touching `region_id` operands
- task inputs logically initialized before use
- `global_to_sram DMA` marks destination regions initialized
- compute task outputs mark destination regions initialized
- `preinitialized` region usage rules
- write/write hazards where two tasks write the same region without an ordering path

Wire it into the Typhoon default compilation path before `LowerIntrin`.

- [ ] **Step 4: Run the compile-time validation tests to verify they pass**

Run: `python -m pytest tests/python/tirx-transform/test_tir_transform_verify_typhoon_graph.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/tirx/transform/verify_typhoon_graph.cc python/tvm/tirx/transform/transform.py python/tvm/tirx/pipeline.py tests/python/tirx-transform/test_tir_transform_verify_typhoon_graph.py
git commit -m "feat: add typhoon compile-time validation"
```

## Task 5: Implement graph building and validation in the Typhoon runtime

**Files:**
- Modify: `CMakeLists.txt`
- Create: `src/runtime/typhoon/typhoon_types.h`
- Create: `src/runtime/typhoon/typhoon_graph.h`
- Create: `src/runtime/typhoon/typhoon_graph.cc`
- Create: `src/runtime/typhoon/typhoon_runtime.cc`
- Test: `tests/python/runtime/test_runtime_typhoon.py`

- [ ] **Step 1: Write the failing runtime validation tests**

```python
import pytest
import tvm


def test_typhoon_rejects_overlapping_regions():
    with pytest.raises(tvm.error.TVMError, match="overlap"):
        run_invalid_typhoon_graph(overlap=True)


def test_typhoon_rejects_compute_from_global_endpoint():
    with pytest.raises(tvm.error.TVMError, match="SRAM"):
        run_invalid_typhoon_graph(compute_reads_global=True)


def test_typhoon_rejects_unknown_task_dependency():
    with pytest.raises(tvm.error.TVMError, match="unknown task"):
        run_invalid_typhoon_graph(unknown_dep=True)


def test_typhoon_rejects_duplicate_task_ids():
    with pytest.raises(tvm.error.TVMError, match="duplicate task_id"):
        run_invalid_typhoon_graph(duplicate_task_id=True)


def test_typhoon_accepts_preinitialized_region_reads():
    run_valid_typhoon_graph(preinitialized_input=True)


def test_typhoon_rejects_dma_bytes_larger_than_region():
    with pytest.raises(tvm.error.TVMError, match="out-of-bounds|size mismatch"):
        run_invalid_typhoon_graph(dma_bytes_too_large=True)


def test_typhoon_rejects_vector_output_larger_than_region():
    with pytest.raises(tvm.error.TVMError, match="out-of-bounds|size mismatch"):
        run_invalid_typhoon_graph(vector_output_too_large=True)
```

- [ ] **Step 2: Run the runtime validation tests to verify they fail**

Run: `python -m pytest tests/python/runtime/test_runtime_typhoon.py -k "overlap or global_endpoint" -v`
Expected: FAIL because no Typhoon runtime ABI exists.

- [ ] **Step 3: Implement graph structs, region tables, and validation**

```cc
struct TyphoonRegion {
  int32_t region_id;
  int64_t offset;
  int64_t size;
  int64_t alignment;
  std::string tag;
  bool initialized{false};
};
```

Implement validation rules from the spec:

- total region bounds must fit in 1MB
- region alignment must hold
- regions must not overlap
- duplicate `region_id` and duplicate `task_id` must be rejected
- dependency IDs must reference known tasks
- cycles in the dependency graph must be rejected before scheduling
- compute tasks must read/write only `region_id` operands
- only DMA may bridge global endpoints and SRAM
- region reads without prior initialization must be rejected unless the region is declared
  `preinitialized`
- two tasks that write the same output `region_id` must have an ordering path between them, or the
  graph is invalid
- runtime out-of-bounds SRAM access must be rejected
- output size mismatches must be rejected, including `DMA bytes > region.size` and
  `elem_count * dtype_size > region.size`
- initialization tracking rules must be enforced conservatively

Define “global endpoint” concretely as `PrimFunc argument buffer handle + byte offset`.

- [ ] **Step 4: Add the exported runtime ABI entrypoints and build wiring**

```cc
extern "C" int TVMTyphoonDeclareRegion(...);
extern "C" int TVMTyphoonAddMatmulTask(...);
extern "C" int TVMTyphoonSubmitGraph(...);
```

Also update `CMakeLists.txt` so `src/runtime/typhoon/*.cc` is part of `RUNTIME_SRCS`.

- [ ] **Step 5: Rebuild and run the validation tests**

Run: `cmake --build build -j8 --target tvm_runtime tvm && python -m pytest tests/python/runtime/test_runtime_typhoon.py -k "overlap or global_endpoint or unknown_task_dependency or duplicate_task_ids or preinitialized_region_reads or dma_bytes_larger_than_region or vector_output_larger_than_region" -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add CMakeLists.txt src/runtime/typhoon/typhoon_types.h src/runtime/typhoon/typhoon_graph.h src/runtime/typhoon/typhoon_graph.cc src/runtime/typhoon/typhoon_runtime.cc tests/python/runtime/test_runtime_typhoon.py
git commit -m "feat: add typhoon graph validation runtime"
```

## Task 6: Add deterministic scheduling, empirical latency estimation, and trace output

**Files:**
- Create: `src/runtime/typhoon/typhoon_cost_model.h`
- Create: `src/runtime/typhoon/typhoon_cost_model.cc`
- Create: `src/runtime/typhoon/typhoon_scheduler.h`
- Create: `src/runtime/typhoon/typhoon_scheduler.cc`
- Modify: `src/runtime/typhoon/typhoon_runtime.cc`
- Test: `tests/python/runtime/test_runtime_typhoon.py`

- [ ] **Step 1: Write the failing scheduler and trace tests**

```python
def test_typhoon_scheduler_is_deterministic():
    trace_a = run_same_ready_graph_twice()
    trace_b = run_same_ready_graph_twice()
    assert trace_a == trace_b


def test_typhoon_trace_has_required_fields():
    trace = run_and_fetch_typhoon_trace()
    assert {
        "task_id",
        "kind",
        "resource",
        "start_time",
        "end_time",
        "sram_bytes_read",
        "sram_bytes_written",
    } <= set(trace[0])
```

- [ ] **Step 2: Run the scheduler tests to verify they fail**

Run: `python -m pytest tests/python/runtime/test_runtime_typhoon.py -k "deterministic or trace_has_required_fields" -v`
Expected: FAIL because the runtime does not yet simulate execution or export trace data.

- [ ] **Step 3: Implement the scheduler and temporary cost model**

```cc
int64_t EstimateLatency(const TyphoonTask& task, const TyphoonHWConfig& hw) {
  switch (task.kind) {
    case TyphoonTaskKind::kDMA:
      return hw.dma_launch_cycles + DivRoundUp(task.bytes, hw.dma_bytes_per_cycle);
    case TyphoonTaskKind::kMatmul:
      return hw.matmul_launch_cycles + EstimateMatmulCycles(task);
    ...
  }
}
```

Implement:

- one DMA, one matrix, one vector, one reshape resource
- deterministic ready-queue ordering by `task_id`, then creation order
- task `start_time` / `end_time`
- SRAM read/write byte accounting per task
- trace records emitted after every task completion
- trace records must include `sram_bytes_read` and `sram_bytes_written`
- graph/task metadata inconsistencies rejected before execution starts
- runtime-time write/write hazards surfaced as errors if a malformed graph somehow bypasses earlier
  validation

- [ ] **Step 4: Add a trace retrieval hook for tests**

```cc
TVM_REGISTER_GLOBAL("runtime.typhoon_get_last_trace_json")
    .set_body_typed([]() { return SerializeLastTraceToJSON(); });
```

Keep this test/debug hook minimal and clearly marked as development support.

- [ ] **Step 5: Rebuild and run the scheduler tests**

Run: `cmake --build build -j8 --target tvm_runtime tvm && python -m pytest tests/python/runtime/test_runtime_typhoon.py -k "deterministic or trace_has_required_fields" -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/runtime/typhoon/typhoon_cost_model.h src/runtime/typhoon/typhoon_cost_model.cc src/runtime/typhoon/typhoon_scheduler.h src/runtime/typhoon/typhoon_scheduler.cc src/runtime/typhoon/typhoon_runtime.cc tests/python/runtime/test_runtime_typhoon.py
git commit -m "feat: add typhoon scheduler and trace support"
```

## Task 7: Add end-to-end Typhoon build and execution coverage

**Files:**
- Test: `tests/python/codegen/test_target_codegen_typhoon.py`
- Modify: `tests/python/runtime/test_runtime_typhoon.py`

- [ ] **Step 1: Write the failing end-to-end tests**

```python
def test_typhoon_dma_matmul_dma_roundtrip():
    mod = build_typhoon_roundtrip_graph()
    out = run_module(mod)
    np.testing.assert_allclose(out, ref)


def test_typhoon_dma_vector_overlap_trace():
    mod = build_typhoon_overlap_graph()
    run_module(mod)
    trace = get_trace()
    assert has_overlap(trace, "dma", "vector")
```

- [ ] **Step 2: Run the end-to-end tests to verify they fail**

Run: `python -m pytest tests/python/codegen/test_target_codegen_typhoon.py -v`
Expected: FAIL because one or more runtime ABI paths are still incomplete.

- [ ] **Step 3: Fill in the remaining integration glue**

```python
func = tvm.compile(prim_func, target="typhoon")
func(a_nd, b_nd, c_nd)
```

Complete any missing integration needed for:

- `DMA -> matmul -> DMA`
- `DMA + vector` overlap
- mixed `reshape/vector/matmul`
- invalid SRAM usage raising a clear error

If generated C needs additional forward declarations, keep the fix in the Typhoon lowering/codegen
layer rather than special-casing tests.

- [ ] **Step 4: Run the focused Typhoon test suite**

Run: `python -m pytest tests/python/target/test_target_typhoon.py tests/python/tirx-base/test_tir_typhoon_ops.py tests/python/tirx-transform/test_tir_transform_typhoon_lower_intrin.py tests/python/runtime/test_runtime_typhoon.py tests/python/codegen/test_target_codegen_typhoon.py -v`
Expected: PASS

- [ ] **Step 5: Run a final rebuild and one broad regression slice**

Run: `cmake --build build -j8 --target tvm_runtime tvm && python -m pytest tests/python/codegen/test_target_codegen_c_host.py tests/python/tirx-transform/test_tir_transform_lower_intrin.py -v`
Expected: PASS, showing Typhoon work did not regress generic host-C or LowerIntrin behavior.

- [ ] **Step 6: Commit**

```bash
git add tests/python/codegen/test_target_codegen_typhoon.py tests/python/runtime/test_runtime_typhoon.py
git commit -m "test: add typhoon end-to-end coverage"
```

## Exit Criteria

The implementation is ready for handoff only when all of the following are true:

- `tvm.target.Target("typhoon ...")` parses and builds
- Typhoon graph helpers exist in `tvm.tirx.typhoon`
- Typhoon ops lower into a stable runtime ABI
- runtime validation catches invalid region and access patterns
- scheduler is deterministic
- trace output contains at least `task_id`, `kind`, `resource`, `start_time`, `end_time`,
  `sram_bytes_read`, and `sram_bytes_written`
- end-to-end examples for DMA/matmul/vector/reshape pass
- no Relax/Relay lowering or multi-core behavior was added
