# Typhoon ResNet18 Fixed-Shape Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Compile fixed-shape `resnet18.onnx` (`1x3x224x224`, `float32`, `NCHW`) into Typhoon task-DAG IR and run it through the existing single-core timing simulator.

**Architecture:** Reuse the existing ONNX -> Relax -> generic lowering pipeline, then add a Typhoon-specific post-lowering path that recognizes the fixed ResNet18 patterns, plans explicit SRAM regions and compile-time tiles, emits `tirx.typhoon.*` graph IR, and extends the existing Typhoon reshape/vector task semantics to cover `im2col` and pooling. Keep the scheduler resource model and overall runtime ABI shape unchanged.

**Tech Stack:** Python (`tvm.relax`, `tvm.tirx`), C++ (`relax`, `tirx`, `runtime`, `target`), TVM Relax/TIR lowering, Typhoon runtime simulator, `pytest`

---

## Shared Metadata Contract

The three new Typhoon compiler passes must communicate through explicit module attributes so they
can be implemented and tested independently.

- `typhoon_resnet18_plan`
  Produced by the pattern-identification pass.
  Carries a serialized plan describing the recognized fixed-shape ResNet18 layer/block structure,
  the canonicalized per-layer operator kind, and the logical tensor/layout requirements needed by
  later passes.
- `typhoon_sram_plan`
  Produced by the SRAM planner.
  Carries the fixed reusable region-pool assignment, compile-time tile choices, and per-layer/per-
  tile region reuse schedule.
- `typhoon_graph_plan`
  Optional internal attribute produced by the graph builder if needed for debugging or test
  assertions, but the authoritative emitted result is the resulting `tirx.typhoon.*` IR itself.

Use one concrete serialized format consistently across these passes, such as JSON carried in a
module attribute. Do not invent a different schema in each task.

## File Structure

### Existing Files To Modify

- Modify: `python/tvm/tirx/typhoon.py`
  Extend helper surface for richer `task_vector` and `task_reshape` payload metadata while preserving the existing Typhoon task kinds.
- Modify: `python/tvm/tirx/transform/transform.py`
  Export any new Typhoon transform passes to Python.
- Modify: `python/tvm/tirx/pipeline.py`
  Insert the new Typhoon automatic mapping pass into the finalization path before `VerifyTyphoonGraph`.
- Modify: `src/tirx/op/typhoon.cc`
  Update Typhoon op registration metadata if reshape/vector payload arity changes.
- Modify: `src/tirx/transform/verify_typhoon_graph.cc`
  Validate the richer reshape/vector payloads, supported transform/vector op codes, and fixed-shape ResNet18 constraints.
- Modify: `src/tirx/transform/lower_typhoon_task_deps.cc`
  Keep dependency lowering aligned if task argument layouts change.
- Modify: `src/runtime/typhoon/typhoon_types.h`
  Extend task descriptors with the extra metadata required for `im2col`, pooling, and fixed-shape tile execution.
- Modify: `src/runtime/typhoon/typhoon_graph.h`
  Update builder signatures and shared graph interfaces for the richer task payloads.
- Modify: `src/runtime/typhoon/typhoon_graph.cc`
  Parse, store, and validate richer task metadata.
- Modify: `src/runtime/typhoon/typhoon_cost_model.h`
  Add a common fixed launch-noise term and expose tile-sensitive latency configuration.
- Modify: `src/runtime/typhoon/typhoon_cost_model.cc`
  Implement `latency = common_fixed_noise + workload_term`.
- Modify: `src/runtime/typhoon/typhoon_runtime.cc`
  Execute `task_reshape(im2col)` and pooling-capable `task_vector` semantics.
- Modify: `src/runtime/typhoon/typhoon_scheduler.cc`
  If needed, update SRAM byte accounting for the richer reshape/vector semantics while keeping the same 4-resource scheduler model.

### New Files To Create

- Create: `src/tirx/transform/identify_typhoon_resnet18.cc`
  Recognize the fixed-shape ResNet18 canonical lowered patterns and build an internal layer plan.
- Create: `src/tirx/transform/plan_typhoon_sram.cc`
  Plan static region allocation and reuse for `1MB` SRAM using the fixed `64x64x64` F32 tile contract.
- Create: `src/tirx/transform/build_typhoon_graph.cc`
  Emit `region_decl`, `task_dma`, `task_matmul`, `task_vector`, `task_reshape`, `submit_graph`, and `wait_graph`.
- Create: `tests/python/tirx-transform/test_tir_transform_identify_typhoon_resnet18.py`
  Cover ResNet18 pattern recognition and early scope gating.
- Create: `tests/python/tirx-transform/test_tir_transform_plan_typhoon_sram.py`
  Cover fixed-tile SRAM planning and region reuse.
- Create: `tests/python/tirx-transform/test_tir_transform_build_typhoon_graph.py`
  Cover graph emission for key ResNet18 subgraphs and layout conversions.
- Create: `tests/python/relax/test_relax_typhoon_resnet18.py`
  End-to-end fixed-shape ResNet18 compile checks, model-path override behavior, and emitted Typhoon runtime ABI presence.

### Existing Tests To Extend

- Modify: `tests/python/runtime/test_runtime_typhoon.py`
  Add runtime coverage for `im2col` reshape semantics, pooling vector ops, and common-noise cost-model behavior.
- Modify: `tests/python/tirx-transform/test_tir_transform_verify_typhoon_graph.py`
  Add compile-time validation coverage for the richer reshape/vector payloads.
- Modify: `tests/python/codegen/test_target_codegen_typhoon.py`
  Add focused coverage for tile-size-sensitive latency and extended reshape/vector lowering semantics.

## Task 1: Add Fixed-Shape ResNet18 Scope Gate And Pattern Recognition

**Files:**
- Create: `src/tirx/transform/identify_typhoon_resnet18.cc`
- Modify: `python/tvm/tirx/transform/transform.py`
- Test: `tests/python/tirx-transform/test_tir_transform_identify_typhoon_resnet18.py`

- [ ] **Step 1: Write the failing scope-gate and pattern tests**

```python
def test_typhoon_resnet18_rejects_non_resnet18_graph():
    mod = build_non_resnet18_tir_module()
    with pytest.raises(ValueError, match="ResNet18"):
        tvm.tirx.transform.IdentifyTyphoonResNet18()(mod)


def test_typhoon_resnet18_accepts_fixed_shape_resnet18_conv_stem():
    mod = build_resnet18_stem_tir_module()
    out = tvm.tirx.transform.IdentifyTyphoonResNet18()(mod)
    assert "typhoon_resnet18_plan" in out.attrs
```

- [ ] **Step 2: Run the new test slice to verify it fails**

Run: `env PYTHONPATH=/home/ayd/code/tvm/python TVM_LIBRARY_PATH=/home/ayd/code/tvm/build python -m pytest -p no:tvm.testing.plugin tests/python/tirx-transform/test_tir_transform_identify_typhoon_resnet18.py -v`
Expected: FAIL because the transform does not exist yet.

- [ ] **Step 3: Implement the Python transform export**

Expose a new transform entrypoint:

```python
def IdentifyTyphoonResNet18():
    return _ffi_api.IdentifyTyphoonResNet18()  # type: ignore
```

- [ ] **Step 4: Implement the C++ recognition pass**

The pass should:

- only activate for `target.kind.name == "typhoon"`
- inspect the post-generic-lowering TIR form
- own any Typhoon-specific canonicalization needed to reach the matchable TIR shape
- reject unsupported structure early
- require input shape `1 x 3 x 224 x 224`
- store the recognized result in a stable module attribute named `typhoon_resnet18_plan`

- [ ] **Step 5: Run the scope-gate test slice to verify it passes**

Run: `env PYTHONPATH=/home/ayd/code/tvm/python TVM_LIBRARY_PATH=/home/ayd/code/tvm/build python -m pytest -p no:tvm.testing.plugin tests/python/tirx-transform/test_tir_transform_identify_typhoon_resnet18.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/tirx/transform/identify_typhoon_resnet18.cc python/tvm/tirx/transform/transform.py tests/python/tirx-transform/test_tir_transform_identify_typhoon_resnet18.py
git commit -m "feat: identify typhoon resnet18 patterns"
```

## Task 2: Add Fixed-Tile SRAM Planning

**Files:**
- Create: `src/tirx/transform/plan_typhoon_sram.cc`
- Modify: `python/tvm/tirx/transform/transform.py`
- Test: `tests/python/tirx-transform/test_tir_transform_plan_typhoon_sram.py`

- [ ] **Step 1: Write the failing SRAM planning tests**

```python
def test_typhoon_sram_plan_uses_fixed_f32_tiles():
    mod = build_resnet18_plan_module()
    out = tvm.tirx.transform.PlanTyphoonSRAM()(mod)
    plan = out.attrs["typhoon_sram_plan"]
    assert plan["matmul_tile"] == [64, 64, 64]


def test_typhoon_sram_plan_rejects_tile_overflow():
    mod = build_artificial_large_layer_plan()
    with pytest.raises(ValueError, match="1MB|SRAM"):
        tvm.tirx.transform.PlanTyphoonSRAM()(mod)
```

- [ ] **Step 2: Run the SRAM planning tests to verify they fail**

Run: `env PYTHONPATH=/home/ayd/code/tvm/python TVM_LIBRARY_PATH=/home/ayd/code/tvm/build python -m pytest -p no:tvm.testing.plugin tests/python/tirx-transform/test_tir_transform_plan_typhoon_sram.py -v`
Expected: FAIL because the planner does not exist yet.

- [ ] **Step 3: Implement the planner**

The planner must:

- assume `float32` only
- fix `m0 = n0 = k0 = 8`
- fix `Mt = Nt = Kt = 64`
- consume `typhoon_resnet18_plan`
- assign a fixed reusable `ACT/WGT/COL/AUX` region pool
- support fixed double-buffering for selected activation/weight paths
- reject any plan that exceeds `1MB`
- store the result in a stable module attribute named `typhoon_sram_plan`

- [ ] **Step 4: Export the planner to Python**

Expose a new transform entrypoint:

```python
def PlanTyphoonSRAM():
    return _ffi_api.PlanTyphoonSRAM()  # type: ignore
```

- [ ] **Step 5: Run the SRAM planning tests to verify they pass**

Run: `env PYTHONPATH=/home/ayd/code/tvm/python TVM_LIBRARY_PATH=/home/ayd/code/tvm/build python -m pytest -p no:tvm.testing.plugin tests/python/tirx-transform/test_tir_transform_plan_typhoon_sram.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/tirx/transform/plan_typhoon_sram.cc python/tvm/tirx/transform/transform.py tests/python/tirx-transform/test_tir_transform_plan_typhoon_sram.py
git commit -m "feat: plan typhoon sram for resnet18"
```

## Task 3: Extend Typhoon Task Payloads For `im2col` And Pooling

**Files:**
- Modify: `python/tvm/tirx/typhoon.py`
- Modify: `src/tirx/op/typhoon.cc`
- Modify: `src/tirx/transform/verify_typhoon_graph.cc`
- Modify: `src/runtime/typhoon/typhoon_types.h`
- Modify: `src/runtime/typhoon/typhoon_graph.h`
- Modify: `src/runtime/typhoon/typhoon_graph.cc`
- Modify: `tests/python/tirx-transform/test_tir_transform_verify_typhoon_graph.py`
- Modify: `tests/python/runtime/test_runtime_typhoon.py`

- [ ] **Step 1: Write the failing validation/runtime tests for richer reshape/vector payloads**

```python
def test_verify_typhoon_graph_accepts_im2col_reshape_payload():
    mod = build_im2col_graph_module()
    tvm.tirx.transform.VerifyTyphoonGraph()(mod)


def test_runtime_typhoon_executes_vector_maxpool():
    trace, output = run_maxpool_graph()
    assert output.shape == (1, 64, 56, 56)
```

- [ ] **Step 2: Run the targeted tests to verify they fail**

Run: `env PYTHONPATH=/home/ayd/code/tvm/python TVM_LIBRARY_PATH=/home/ayd/code/tvm/build python -m pytest -p no:tvm.testing.plugin tests/python/tirx-transform/test_tir_transform_verify_typhoon_graph.py tests/python/runtime/test_runtime_typhoon.py -k 'im2col or maxpool or global_average_pool' -v`
Expected: FAIL because the extra task payload metadata is not supported yet.

- [ ] **Step 3: Extend the helper and op registration surfaces**

Add explicit payload arguments for:

- `task_reshape(..., transform_code, extra_shape_metadata, deps)`
- `task_vector(..., op_code, ..., extra_window_metadata, deps)`

Keep the existing task kinds and overall ABI shape.

- [ ] **Step 4: Extend compile-time and runtime graph validation**

Validation must check:

- supported `im2col` transform codes and parameter arity
- supported pooling vector op codes and parameter arity
- shape compatibility for pooling/im2col metadata

- [ ] **Step 5: Extend runtime task descriptors and graph builder parsing**

The runtime graph structures must carry the added payload metadata without introducing new task
kinds.

- [ ] **Step 6: Extend runtime execution semantics**

Implement:

- `task_reshape(im2col)` using shape-unit-native duplication/reorder semantics
- `task_vector(maxpool)`
- `task_vector(global_avg_pool)`

- [ ] **Step 7: Run the targeted tests to verify they pass**

Run: `env PYTHONPATH=/home/ayd/code/tvm/python TVM_LIBRARY_PATH=/home/ayd/code/tvm/build python -m pytest -p no:tvm.testing.plugin tests/python/tirx-transform/test_tir_transform_verify_typhoon_graph.py tests/python/runtime/test_runtime_typhoon.py -k 'im2col or maxpool or global_average_pool' -v`
Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add python/tvm/tirx/typhoon.py src/tirx/op/typhoon.cc src/tirx/transform/verify_typhoon_graph.cc src/runtime/typhoon/typhoon_types.h src/runtime/typhoon/typhoon_graph.h src/runtime/typhoon/typhoon_graph.cc src/runtime/typhoon/typhoon_runtime.cc tests/python/tirx-transform/test_tir_transform_verify_typhoon_graph.py tests/python/runtime/test_runtime_typhoon.py
git commit -m "feat: extend typhoon reshape and vector payloads"
```

## Task 4: Build Typhoon Graph Emission Pass

**Files:**
- Create: `src/tirx/transform/build_typhoon_graph.cc`
- Modify: `python/tvm/tirx/transform/transform.py`
- Modify: `python/tvm/tirx/pipeline.py`
- Test: `tests/python/tirx-transform/test_tir_transform_build_typhoon_graph.py`

- [ ] **Step 1: Write the failing graph-emission tests**

```python
def test_build_typhoon_graph_emits_region_decl_and_tasks():
    mod = build_planned_resnet18_conv_block()
    out = tvm.tirx.transform.BuildTyphoonGraph()(mod)
    text = out.script()
    assert "tirx.typhoon.region_decl" in text
    assert "tirx.typhoon.task_matmul" in text
    assert "tirx.typhoon.submit_graph" in text
```

- [ ] **Step 2: Run the graph-emission tests to verify they fail**

Run: `env PYTHONPATH=/home/ayd/code/tvm/python TVM_LIBRARY_PATH=/home/ayd/code/tvm/build python -m pytest -p no:tvm.testing.plugin tests/python/tirx-transform/test_tir_transform_build_typhoon_graph.py -v`
Expected: FAIL because the emission pass does not exist yet.

- [ ] **Step 3: Implement the graph-emission pass**

The pass must:

- consume `typhoon_resnet18_plan` and `typhoon_sram_plan`
- emit fixed reusable `region_decl` statements for the planned region pool
- emit `task_dma`, `task_matmul`, `task_vector`, `task_reshape`
- emit one graph submission per generated function

- [ ] **Step 4: Wire the new pass into the Typhoon pipeline**

Insert it after generic lowering and before:

- `VerifyTyphoonGraph`
- `LowerTyphoonTaskDeps`
- `LowerTyphoonSubmitGraph`
- `LowerIntrin`

- [ ] **Step 5: Run the graph-emission tests to verify they pass**

Run: `env PYTHONPATH=/home/ayd/code/tvm/python TVM_LIBRARY_PATH=/home/ayd/code/tvm/build python -m pytest -p no:tvm.testing.plugin tests/python/tirx-transform/test_tir_transform_build_typhoon_graph.py tests/python/tirx-transform/test_tir_transform_typhoon_lower_intrin.py tests/python/tirx-transform/test_tir_transform_verify_typhoon_graph.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/tirx/transform/build_typhoon_graph.cc python/tvm/tirx/transform/transform.py python/tvm/tirx/pipeline.py tests/python/tirx-transform/test_tir_transform_build_typhoon_graph.py
git commit -m "feat: emit typhoon graph for resnet18"
```

## Task 5: Update The Temporary Cost Model For Tile Penalties

**Files:**
- Modify: `src/runtime/typhoon/typhoon_cost_model.h`
- Modify: `src/runtime/typhoon/typhoon_cost_model.cc`
- Modify: `tests/python/runtime/test_runtime_typhoon.py`
- Modify: `tests/python/codegen/test_target_codegen_typhoon.py`

- [ ] **Step 1: Write the failing cost-model tests**

```python
def test_typhoon_cost_model_penalizes_smaller_tiles():
    big = estimate_matmul_latency(m=64, n=64, k=64)
    small = estimate_matmul_latency(m=32, n=32, k=32) * 8
    assert small > big
```

- [ ] **Step 2: Run the cost-model tests to verify they fail**

Run: `env PYTHONPATH=/home/ayd/code/tvm/python TVM_LIBRARY_PATH=/home/ayd/code/tvm/build python -m pytest -p no:tvm.testing.plugin tests/python/runtime/test_runtime_typhoon.py tests/python/codegen/test_target_codegen_typhoon.py -k 'cost_model or tile' -v`
Expected: FAIL because the common-noise latency term is not implemented yet.

- [ ] **Step 3: Implement the common fixed-noise latency model**

The model must use:

- `latency = common_fixed_noise + workload_term`

Where:

- `common_fixed_noise` is identical across DMA/matmul/vector/reshape
- `workload_term` remains task-kind specific

- [ ] **Step 4: Run the cost-model tests to verify they pass**

Run: `env PYTHONPATH=/home/ayd/code/tvm/python TVM_LIBRARY_PATH=/home/ayd/code/tvm/build python -m pytest -p no:tvm.testing.plugin tests/python/runtime/test_runtime_typhoon.py tests/python/codegen/test_target_codegen_typhoon.py -k 'cost_model or tile' -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/runtime/typhoon/typhoon_cost_model.h src/runtime/typhoon/typhoon_cost_model.cc tests/python/runtime/test_runtime_typhoon.py tests/python/codegen/test_target_codegen_typhoon.py
git commit -m "feat: add tile-sensitive typhoon cost model"
```

## Task 6: Add End-To-End Relax ResNet18 Compilation Coverage

**Files:**
- Create: `tests/python/relax/test_relax_typhoon_resnet18.py`

- [ ] **Step 1: Write the failing end-to-end Relax test**

```python
def test_relax_resnet18_compiles_to_typhoon_graph(tmp_path):
    model_path = get_resnet18_model_path()
    ex = compile_resnet18_to_typhoon(model_path)
    source = ex.mod._collect_dso_modules()[0].inspect_source()
    assert "TVMTyphoonAddMatmulTask" in source
    assert "TVMTyphoonSubmitGraph" in source
```

- [ ] **Step 2: Run the end-to-end Relax test to verify it fails**

Run: `env PYTHONPATH=/home/ayd/code/tvm/python TVM_LIBRARY_PATH=/home/ayd/code/tvm/build python -m pytest -p no:tvm.testing.plugin tests/python/relax/test_relax_typhoon_resnet18.py -v`
Expected: FAIL because the compiled ResNet18 path still produces generic VM code instead of Typhoon graph tasks.

- [ ] **Step 3: Implement the model-path contract**

The test helper should:

- default to `~/model/resnet18.onnx`
- allow override via environment variable
- require input shape `1 x 3 x 224 x 224`

- [ ] **Step 4: Run the end-to-end Relax test to verify it passes**

Run: `env PYTHONPATH=/home/ayd/code/tvm/python TVM_LIBRARY_PATH=/home/ayd/code/tvm/build python -m pytest -p no:tvm.testing.plugin tests/python/relax/test_relax_typhoon_resnet18.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/python/relax/test_relax_typhoon_resnet18.py
git commit -m "test: add typhoon resnet18 relax coverage"
```

## Task 7: Add Full-Model Simulator Execution And Numerical Validation

**Files:**
- Modify: `tests/python/relax/test_relax_typhoon_resnet18.py`

- [ ] **Step 1: Extend the end-to-end Relax test with simulator execution**

```python
def test_relax_resnet18_runs_in_typhoon_simulator(tmp_path):
    model_path = get_resnet18_model_path()
    ex = compile_resnet18_to_typhoon(model_path)
    output = run_resnet18_in_typhoon_simulator(ex, input_shape=(1, 3, 224, 224))
    ref = run_reference_resnet18(model_path, input_shape=(1, 3, 224, 224))
    np.testing.assert_allclose(output, ref, rtol=1e-4, atol=1e-4)
```

- [ ] **Step 2: Run the simulator-execution test to verify it fails**

Run: `env PYTHONPATH=/home/ayd/code/tvm/python TVM_LIBRARY_PATH=/home/ayd/code/tvm/build python -m pytest -p no:tvm.testing.plugin tests/python/relax/test_relax_typhoon_resnet18.py -k 'runs_in_typhoon_simulator' -v`
Expected: FAIL until the full-model graph both compiles and executes correctly through the simulator.

- [ ] **Step 3: Implement the minimal harness and any missing fixes**

The test must:

- run the compiled model through the Typhoon simulator path
- collect the actual output tensor
- compare against a reference result
- verify the compiled source still contains `TVMTyphoon*` calls

- [ ] **Step 4: Run the simulator-execution test to verify it passes**

Run: `env PYTHONPATH=/home/ayd/code/tvm/python TVM_LIBRARY_PATH=/home/ayd/code/tvm/build python -m pytest -p no:tvm.testing.plugin tests/python/relax/test_relax_typhoon_resnet18.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/python/relax/test_relax_typhoon_resnet18.py
git commit -m "test: validate typhoon resnet18 simulator execution"
```

## Task 8: Run Full Focused Verification

**Files:**
- Test-only verification task

- [ ] **Step 1: Rebuild TVM after all task changes**

Run: `cmake --build build -j8 --target tvm_runtime tvm`
Expected: build succeeds

- [ ] **Step 2: Run the focused Typhoon and ResNet18 suite**

Run: `env PYTHONPATH=/home/ayd/code/tvm/python TVM_LIBRARY_PATH=/home/ayd/code/tvm/build python -m pytest -p no:tvm.testing.plugin tests/python/target/test_target_typhoon.py tests/python/tirx-base/test_tir_typhoon_ops.py tests/python/tirx-transform/test_tir_transform_typhoon_lower_intrin.py tests/python/tirx-transform/test_tir_transform_verify_typhoon_graph.py tests/python/tirx-transform/test_tir_transform_identify_typhoon_resnet18.py tests/python/tirx-transform/test_tir_transform_plan_typhoon_sram.py tests/python/tirx-transform/test_tir_transform_build_typhoon_graph.py tests/python/runtime/test_runtime_typhoon.py tests/python/codegen/test_target_codegen_typhoon.py tests/python/relax/test_relax_typhoon_resnet18.py -v`
Expected: PASS

- [ ] **Step 3: Run one broad regression slice**

Run: `env PYTHONPATH=/home/ayd/code/tvm/python TVM_LIBRARY_PATH=/home/ayd/code/tvm/build python -m pytest -p no:tvm.testing.plugin tests/python/codegen/test_target_codegen_c_host.py tests/python/tirx-transform/test_tir_transform_lower_intrin.py tests/python/contrib/test_hexagon/test_relax_integration.py -k 'not requires_hexagon' -v`
Expected: PASS or only expected skips

- [ ] **Step 4: Commit any final fixes**

```bash
git add -A
git commit -m "test: verify typhoon resnet18 support"
```
