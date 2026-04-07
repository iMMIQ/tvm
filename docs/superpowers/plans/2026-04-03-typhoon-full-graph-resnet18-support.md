# Typhoon Full-Graph ResNet18 Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Compile the full fixed-shape `resnet18.onnx` graph into Typhoon C ABI with no host fallback, execute the full graph in the Typhoon runtime simulator, and validate both runtime safety and elementwise numerical agreement with ONNXRuntime.

**Architecture:** Expand the current stem-only Typhoon flow into a full-graph pipeline tied to the canonical local `resnet18.onnx` artifact. The compiler will emit one full-graph Typhoon DAG using the existing four Typhoon task families, the SRAM planner will become whole-graph lifetime-aware, and the runtime will execute and validate every ResNet18 operator through Typhoon simulation only.

**Tech Stack:** Python (`tvm.relax`, `tvm.tirx`, `pytest`, `onnx`, `onnxruntime`), C++ (`tirx`, `runtime`, `target`), TVM Relax/TIR lowering, Typhoon runtime simulator

---

## File Structure

### Existing Files To Modify

- Modify: `src/tirx/transform/identify_typhoon_resnet18.cc`
  Expand recognition from stem-only to the canonical full fixed-shape ResNet18 graph and emit a full-graph `typhoon_resnet18_plan`.
- Modify: `src/tirx/transform/plan_typhoon_sram.cc`
  Replace layer-local SRAM sizing with whole-graph region assignment, live-range tracking, and reuse validation.
- Modify: `src/tirx/transform/build_typhoon_graph.cc`
  Emit a full-graph Typhoon task DAG for all ResNet18 layers instead of a stem helper only.
- Modify: `src/tirx/transform/verify_typhoon_graph.cc`
  Validate full-graph Typhoon DAG legality, expanded reshape/vector semantics, and cross-layer region use.
- Modify: `src/tirx/transform/lower_typhoon_task_deps.cc`
  Keep metadata/dependency lowering aligned with expanded full-graph task payloads.
- Modify: `python/tvm/tirx/pipeline.py`
  Keep Typhoon graph construction wired for the full-graph path and no-fallback expectations.
- Modify: `python/tvm/tirx/build.py`
  Preserve early Typhoon graph build so full-graph recognition happens before generic packed-API lowering.
- Modify: `src/runtime/typhoon/typhoon_types.h`
  Extend task metadata fields as needed for full-graph reshape/vector/dense execution and richer trace data.
- Modify: `src/runtime/typhoon/typhoon_graph.h`
  Extend graph builder/runtime interfaces for full-graph metadata and diagnostics.
- Modify: `src/runtime/typhoon/typhoon_graph.cc`
  Validate whole-graph region lifetimes, task footprints, and runtime safety constraints.
- Modify: `src/runtime/typhoon/typhoon_runtime.cc`
  Execute all full-graph ResNet18 task semantics and keep runtime safety checks active.
- Modify: `src/runtime/typhoon/typhoon_scheduler.cc`
  Preserve the 4-resource scheduler while making trace/hazard reporting useful for whole-graph execution.
- Modify: `src/runtime/typhoon/typhoon_cost_model.h`
  Keep whole-graph latency and SRAM traffic accounting aligned with full-graph tasks.
- Modify: `src/runtime/typhoon/typhoon_cost_model.cc`
  Keep latency and byte estimators correct for expanded full-graph task usage.
- Modify: `src/target/source/codegen_c_host.cc`
  Keep host-side C codegen compatible with Typhoon helper metadata payloads.
- Modify: `tests/python/tirx-transform/test_tir_transform_identify_typhoon_resnet18.py`
  Replace stem-only assertions with full-graph recognition assertions.
- Modify: `tests/python/tirx-transform/test_tir_transform_plan_typhoon_sram.py`
  Add whole-graph SRAM legality and reuse tests.
- Modify: `tests/python/tirx-transform/test_tir_transform_build_typhoon_graph.py`
  Replace local helper expectations with full-graph task-DAG expectations.
- Modify: `tests/python/tirx-transform/test_tir_transform_verify_typhoon_graph.py`
  Add full-graph legality and hazard-shape validation.
- Modify: `tests/python/runtime/test_runtime_typhoon.py`
  Add full-graph runtime checks, richer trace checks, and failure-path coverage.
- Modify: `tests/python/codegen/test_target_codegen_typhoon.py`
  Assert actual full-graph Typhoon C ABI emission.
- Modify: `tests/python/relax/test_relax_typhoon_resnet18.py`
  Enforce no host fallback, full-graph Typhoon ABI presence, runtime safety, and ONNXRuntime elementwise comparison.

### New Files To Create

- Create: `tests/python/tirx-transform/test_tir_transform_typhoon_full_graph_plan.py`
  Focused tests for the canonical full-graph metadata schema and `recognized_scope = "full_graph"`.
- Create: `tests/python/tirx-transform/test_tir_transform_typhoon_full_graph_sram.py`
  Focused tests for live-range-based SRAM reuse and residual preservation.
- Create: `tests/python/relax/test_relax_typhoon_resnet18_onnxruntime.py`
  Dedicated ONNXRuntime-backed full-model numerical comparison tests and clearer missing-dependency handling.

## Environment Notes

- The supported model contract is the canonical local artifact at `~/model/resnet18.onnx` or `TYPHOON_RESNET18_ONNX_PATH`.
- The canonical ONNX graph contains `Conv`, `Relu`, `Identity`, `Add`, `MaxPool`, `GlobalAveragePool`, `Flatten`, and `Gemm`, with batch norm already folded away.
- End-to-end numerical validation requires `onnxruntime`. If it is missing, the ONNXRuntime-backed tests must fail with a clear message instead of silently weakening coverage.
- Use deterministic input generation with a fixed NumPy random seed.

## Shared Test Scaffolding

Create shared helpers before feature-specific tests start to grow:

- Create: `tests/python/tirx-transform/typhoon_resnet18_test_utils.py`
  Canonical full-graph transform fixtures such as `build_canonical_resnet18_tir_module()` and `build_full_graph_plan_module()`.
- Create: `tests/python/runtime/typhoon_test_utils.py`
  Runtime graph fixture builders such as `build_runtime_graph_with_*` and `get_trace_json()`.
- Create: `tests/python/relax/typhoon_resnet18_test_utils.py`
  Relax/ONNXRuntime helpers such as canonical model loading, seeded feeds, `assert_elementwise_close_with_context(...)`, and `format_first_mismatch(...)`.

Every new helper symbol referenced by the later tasks should be implemented in one of these files rather than invented ad hoc in individual tests.

### Task 0: Add Shared Typhoon ResNet18 Test Scaffolding

**Files:**
- Create: `tests/python/tirx-transform/typhoon_resnet18_test_utils.py`
- Create: `tests/python/runtime/typhoon_test_utils.py`
- Create: `tests/python/relax/typhoon_resnet18_test_utils.py`

- [ ] **Step 1: Write the failing scaffolding smoke tests**

```python
def test_typhoon_resnet18_test_utils_loads_canonical_model():
    from tests.python.relax.typhoon_resnet18_test_utils import load_canonical_resnet18_model
    model = load_canonical_resnet18_model()
    assert model is not None


def test_runtime_typhoon_test_utils_exposes_trace_helper():
    from tests.python.runtime.typhoon_test_utils import get_trace_json
    assert callable(get_trace_json)
```

- [ ] **Step 2: Run the scaffolding slice to verify it fails**

Run: `env PYTHONPATH=/home/ayd/code/tvm/python TVM_LIBRARY_PATH=/home/ayd/code/tvm/build python -m pytest -p no:tvm.testing.plugin tests/python/relax tests/python/runtime -k "canonical_model or trace_helper" -v`
Expected: FAIL because the shared utility modules do not exist yet.

- [ ] **Step 3: Create the shared utility modules**

Add canonical helpers such as:

- `load_canonical_resnet18_model()`
- `build_canonical_resnet18_tir_module()`
- `build_full_graph_plan_module()`
- `build_runtime_graph_with_*`
- `get_trace_json()`
- `assert_elementwise_close_with_context(...)`
- `format_first_mismatch(...)`

- [ ] **Step 4: Re-run the scaffolding slice to verify it passes**

Run: `env PYTHONPATH=/home/ayd/code/tvm/python TVM_LIBRARY_PATH=/home/ayd/code/tvm/build python -m pytest -p no:tvm.testing.plugin tests/python/relax tests/python/runtime -k "canonical_model or trace_helper" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/python/tirx-transform/typhoon_resnet18_test_utils.py \
        tests/python/runtime/typhoon_test_utils.py \
        tests/python/relax/typhoon_resnet18_test_utils.py
git commit -m "test: add typhoon resnet18 shared scaffolding"
```

### Task 1: Lock In Canonical Full-Graph Recognition Contract

**Files:**
- Modify: `src/tirx/transform/identify_typhoon_resnet18.cc`
- Modify: `tests/python/tirx-transform/test_tir_transform_identify_typhoon_resnet18.py`
- Create: `tests/python/tirx-transform/test_tir_transform_typhoon_full_graph_plan.py`

- [ ] **Step 1: Write the failing full-graph recognition tests**

```python
def test_typhoon_resnet18_recognizes_full_graph_scope():
    mod = build_canonical_resnet18_tir_module()
    out = tvm.tirx.transform.IdentifyTyphoonResNet18()(mod)
    plan = json.loads(out.attrs["typhoon_resnet18_plan"])
    assert plan["recognized_scope"] == "full_graph"
    assert plan["model"] == "resnet18"
    assert len(plan["layers"]) >= 29
    assert [stage["stage_id"] for stage in plan["stages"]] == [0, 1, 2, 3]
    assert [block["block_id"] for block in plan["blocks"][:2]] == [0, 1]
    first = plan["layers"][0]
    assert {"layer_id", "kind", "logical_input_shape", "logical_output_shape"} <= first.keys()
    assert any("layout_transform" in layer for layer in plan["layers"])
    assert any(layer.get("requires_im2col", False) for layer in plan["layers"])
    assert any(edge.get("edge_kind") == "residual" for edge in plan["edges"])
    assert [layer["kind"] for layer in plan["layers"][-3:]] == ["global_avg_pool", "flatten", "dense"]


def test_typhoon_resnet18_rejects_graphs_with_explicit_batch_norm():
    mod = build_resnet18_with_batch_norm_tir_module()
    with pytest.raises(ValueError, match="BatchNormalization|canonical"):
        tvm.tirx.transform.IdentifyTyphoonResNet18()(mod)
```

- [ ] **Step 2: Run the recognition slice to verify it fails**

Run: `env PYTHONPATH=/home/ayd/code/tvm/python TVM_LIBRARY_PATH=/home/ayd/code/tvm/build python -m pytest -p no:tvm.testing.plugin tests/python/tirx-transform/test_tir_transform_identify_typhoon_resnet18.py tests/python/tirx-transform/test_tir_transform_typhoon_full_graph_plan.py -v`
Expected: FAIL because the current recognizer only emits `recognized_scope = "stem"` and does not reject unsupported canonical variants explicitly.

- [ ] **Step 3: Expand the plan schema in the recognizer**

Implement a full-graph schema that includes:

```json
{
  "model": "resnet18",
  "recognized_scope": "full_graph",
  "input_shape": [1, 3, 224, 224],
  "dtype": "float32",
  "stages": [...],
  "blocks": [...],
  "layers": [
    {
      "layer_id": 0,
      "kind": "conv2d",
      "logical_input_shape": [1, 3, 224, 224],
      "logical_output_shape": [1, 64, 112, 112],
      "requires_im2col": true,
      "layout_transform": {
        "input_layout": "NCHW",
        "output_layout": "zZ"
      }
    }
  ],
  "edges": [...]
}
```

- [ ] **Step 4: Implement full-graph recognition**

In `src/tirx/transform/identify_typhoon_resnet18.cc`, replace the stem-only matcher with recognition for the canonical fixed-shape full graph after Typhoon-targeted lowering. The recognizer must:

- require the canonical operator family only
- reject explicit batch norm and unsupported exporter variants early
- identify residual edges and head operators (`GlobalAveragePool`, `Flatten`, `Gemm`)
- emit `recognized_scope = "full_graph"`

- [ ] **Step 5: Re-run the recognition slice to verify it passes**

Run: `ninja -C /home/ayd/code/tvm/build tvm tvm_runtime`
Expected: PASS

- [ ] **Step 6: Re-run the recognition slice to verify it passes**

Run: `env PYTHONPATH=/home/ayd/code/tvm/python TVM_LIBRARY_PATH=/home/ayd/code/tvm/build python -m pytest -p no:tvm.testing.plugin tests/python/tirx-transform/test_tir_transform_identify_typhoon_resnet18.py tests/python/tirx-transform/test_tir_transform_typhoon_full_graph_plan.py -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add src/tirx/transform/identify_typhoon_resnet18.cc \
        tests/python/tirx-transform/test_tir_transform_identify_typhoon_resnet18.py \
        tests/python/tirx-transform/test_tir_transform_typhoon_full_graph_plan.py
git commit -m "feat: recognize full typhoon resnet18 graph"
```

### Task 2: Add Whole-Graph SRAM Planning And Residual Lifetimes

**Files:**
- Modify: `src/tirx/transform/plan_typhoon_sram.cc`
- Modify: `tests/python/tirx-transform/test_tir_transform_plan_typhoon_sram.py`
- Create: `tests/python/tirx-transform/test_tir_transform_typhoon_full_graph_sram.py`

- [ ] **Step 1: Write the failing whole-graph SRAM planning tests**

```python
def test_typhoon_sram_plan_tracks_full_graph_live_ranges():
    mod = build_full_graph_plan_module()
    out = tvm.tirx.transform.PlanTyphoonSRAM()(mod)
    plan = json.loads(out.attrs["typhoon_sram_plan"])
    assert any(region["name"] == "residual" for region in plan["regions"])
    assert all("offset" in region and "size" in region and "alignment" in region for region in plan["regions"])
    assert any(item["last_use_layer_id"] > item["producer_layer_id"] for item in plan["live_ranges"])
    assert any("consumer_window" in item and "reuse_after_layer_id" in item for item in plan["live_ranges"])
    assert plan["peak_live_bytes"] <= 1024 * 1024


def test_typhoon_sram_plan_regions_are_in_bounds_and_non_overlapping():
    mod = build_full_graph_plan_module()
    out = tvm.tirx.transform.PlanTyphoonSRAM()(mod)
    plan = json.loads(out.attrs["typhoon_sram_plan"])
    spans = []
    for region in plan["regions"]:
        assert region["offset"] >= 0
        assert region["size"] > 0
        assert region["offset"] + region["size"] <= 1024 * 1024
        spans.append((region["region_id"], region["offset"], region["offset"] + region["size"]))
    for i, lhs in enumerate(spans):
        for rhs in spans[i + 1:]:
            assert max(lhs[1], rhs[1]) >= min(lhs[2], rhs[2])


def test_typhoon_sram_plan_rejects_invalid_residual_reuse():
    mod = build_invalid_residual_plan_module()
    with pytest.raises(ValueError, match="residual|live range|reuse"):
        tvm.tirx.transform.PlanTyphoonSRAM()(mod)
```

- [ ] **Step 2: Run the SRAM planning slice to verify it fails**

Run: `env PYTHONPATH=/home/ayd/code/tvm/python TVM_LIBRARY_PATH=/home/ayd/code/tvm/build python -m pytest -p no:tvm.testing.plugin tests/python/tirx-transform/test_tir_transform_plan_typhoon_sram.py tests/python/tirx-transform/test_tir_transform_typhoon_full_graph_sram.py -v`
Expected: FAIL because the current SRAM planner is layer-local and stem-oriented.

- [ ] **Step 3: Extend the SRAM plan schema**

Update the planner to emit whole-graph metadata such as:

```json
{
  "regions": [...],
  "live_ranges": [
    {
      "region_id": 5,
      "producer_layer_id": 12,
      "last_use_layer_id": 15,
      "role": "RESIDUAL",
      "consumer_window": [13, 15],
      "reuse_after_layer_id": 15
    }
  ],
  "peak_live_bytes": 786432
}
```

- [ ] **Step 4: Implement whole-graph lifetime-aware planning**

In `src/tirx/transform/plan_typhoon_sram.cc`, plan:

- reusable `ACT/WGT/COL/AUX/OUT/RESIDUAL` pools
- live ranges for every produced value
- residual preservation until merge
- streamed `WGT` residency by layer/tile
- explicit rejection when the whole graph cannot fit within 1MB under the chosen schedule

- [ ] **Step 5: Re-run the SRAM planning slice to verify it passes**

Run: `ninja -C /home/ayd/code/tvm/build tvm tvm_runtime`
Expected: PASS

- [ ] **Step 6: Re-run the SRAM planning slice to verify it passes**

Run: `env PYTHONPATH=/home/ayd/code/tvm/python TVM_LIBRARY_PATH=/home/ayd/code/tvm/build python -m pytest -p no:tvm.testing.plugin tests/python/tirx-transform/test_tir_transform_plan_typhoon_sram.py tests/python/tirx-transform/test_tir_transform_typhoon_full_graph_sram.py -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add src/tirx/transform/plan_typhoon_sram.cc \
        tests/python/tirx-transform/test_tir_transform_plan_typhoon_sram.py \
        tests/python/tirx-transform/test_tir_transform_typhoon_full_graph_sram.py
git commit -m "feat: plan full-graph typhoon sram reuse"
```

### Task 3: Emit A Full-Graph Typhoon Task DAG

**Files:**
- Modify: `src/tirx/transform/build_typhoon_graph.cc`
- Modify: `src/tirx/transform/verify_typhoon_graph.cc`
- Modify: `src/tirx/transform/lower_typhoon_task_deps.cc`
- Modify: `tests/python/tirx-transform/test_tir_transform_build_typhoon_graph.py`
- Modify: `tests/python/tirx-transform/test_tir_transform_verify_typhoon_graph.py`

- [ ] **Step 1: Write the failing full-graph emission tests**

```python
def test_build_typhoon_graph_emits_full_graph_helper():
    mod = build_full_graph_planned_module()
    out = tvm.tirx.transform.BuildTyphoonGraph()(mod)
    text = out.script()
    assert text.count("_typhoon_graph") >= 1
    assert text.count("_typhoon_graph") <= 4
    assert text.count("TVMTyphoonAddMatmulTask") >= 20
    assert "TVMTyphoonAddVectorTask" in text
    assert "TVMTyphoonAddReshapeTask" in text


def test_build_typhoon_graph_uses_plan_as_authoritative_topology():
    mod = build_full_graph_planned_module_with_mismatched_tir_structure()
    with pytest.raises(ValueError, match="plan|topology|authoritative"):
        tvm.tirx.transform.BuildTyphoonGraph()(mod)


def test_verify_typhoon_graph_rejects_invalid_full_graph_region_reuse():
    mod = build_invalid_full_graph_typhoon_module()
    with pytest.raises(ValueError, match="region|reuse|hazard"):
        tvm.tirx.transform.VerifyTyphoonGraph()(mod)
```

- [ ] **Step 2: Run the graph emission slice to verify it fails**

Run: `env PYTHONPATH=/home/ayd/code/tvm/python TVM_LIBRARY_PATH=/home/ayd/code/tvm/build python -m pytest -p no:tvm.testing.plugin tests/python/tirx-transform/test_tir_transform_build_typhoon_graph.py tests/python/tirx-transform/test_tir_transform_verify_typhoon_graph.py -v`
Expected: FAIL because the current graph builder emits stem-only Typhoon DAGs.

- [ ] **Step 3: Extend Typhoon task payload coverage**

Expand builder/verification expectations so the full graph can be represented entirely through:

- `task_dma`
- `task_matmul`
- `task_vector` for `add`, `relu`, `maxpool`, `global_avg_pool`
- `task_reshape` for copy, `im2col`, flatten, transpose, layout conversion

- [ ] **Step 4: Implement full-graph DAG emission**

In `src/tirx/transform/build_typhoon_graph.cc`, emit a single full-graph helper or a very small number of large Typhoon helpers for the canonical full model. The emitted helper set must cover:

- all 20 conv-backed matmuls
- all residual adds
- all relus
- maxpool
- global average pool
- flatten / transpose / dense preparation
- final dense matmul and bias add

- [ ] **Step 5: Make the builder consume the plan as the only topology source**

Update `src/tirx/transform/build_typhoon_graph.cc` so it consumes `typhoon_resnet18_plan` as the authoritative graph description. It may use TIR only to bind buffers/params to planned layers, but it must not rediscover graph topology from TIR. Add explicit validation that mismatches between plan topology and available lowered functions fail clearly.

- [ ] **Step 6: Tighten graph verification and dependency lowering**

Update `src/tirx/transform/verify_typhoon_graph.cc` and `src/tirx/transform/lower_typhoon_task_deps.cc` so full-graph metadata, dependency payloads, and region use are validated and lowered consistently.

- [ ] **Step 7: Rebuild the C++ runtime and codegen artifacts**

Run: `ninja -C /home/ayd/code/tvm/build tvm tvm_runtime`
Expected: PASS

- [ ] **Step 8: Re-run the graph emission slice to verify it passes**

Run: `env PYTHONPATH=/home/ayd/code/tvm/python TVM_LIBRARY_PATH=/home/ayd/code/tvm/build python -m pytest -p no:tvm.testing.plugin tests/python/tirx-transform/test_tir_transform_build_typhoon_graph.py tests/python/tirx-transform/test_tir_transform_verify_typhoon_graph.py -v`
Expected: PASS

- [ ] **Step 9: Commit**

```bash
git add src/tirx/transform/build_typhoon_graph.cc \
        src/tirx/transform/verify_typhoon_graph.cc \
        src/tirx/transform/lower_typhoon_task_deps.cc \
        tests/python/tirx-transform/test_tir_transform_build_typhoon_graph.py \
        tests/python/tirx-transform/test_tir_transform_verify_typhoon_graph.py
git commit -m "feat: emit full-graph typhoon resnet18 dag"
```

### Task 4: Keep The Build Pipeline Full-Graph And No-Fallback

**Files:**
- Modify: `python/tvm/tirx/pipeline.py`
- Modify: `python/tvm/tirx/build.py`
- Modify: `src/target/source/codegen_c_host.cc`
- Modify: `tests/python/codegen/test_target_codegen_typhoon.py`

- [ ] **Step 1: Write the failing no-fallback C source tests**

```python
def test_typhoon_codegen_emits_full_graph_c_abi_for_resnet18():
    lib = build_canonical_resnet18_typhoon_module()
    src = lib.inspect_source()
    assert src.count("_typhoon_graph") >= 1
    assert src.count("_typhoon_graph") <= 4
    assert "__tvm_ffi_conv2d_typhoon_graph" not in src
    assert "TVMTyphoonSubmitGraph" in src
```

- [ ] **Step 2: Run the codegen slice to verify it fails**

Run: `env PYTHONPATH=/home/ayd/code/tvm/python TVM_LIBRARY_PATH=/home/ayd/code/tvm/build python -m pytest -p no:tvm.testing.plugin tests/python/codegen/test_target_codegen_typhoon.py -v`
Expected: FAIL because the current codegen still reflects local helper behavior and not a full-graph no-fallback helper.

- [ ] **Step 3: Keep Typhoon graphization in the correct phase**

Ensure the build path still runs Typhoon graph construction before generic packed-API lowering and now preserves the full-graph helper naming/retention policy.

- [ ] **Step 4: Keep C host codegen compatible with full-graph metadata payloads**

If full-graph helper lowering expands stack-alloca or metadata emission requirements, update `src/target/source/codegen_c_host.cc` so actual generated Typhoon C remains compilable and inspectable.

- [ ] **Step 5: Add actual C source assertions**

Update `tests/python/codegen/test_target_codegen_typhoon.py` to assert:

- full-graph helper symbol presence, allowing one helper or a very small number of large helpers
- `TVMTyphoonDeclareRegion`
- `TVMTyphoonAddDMATask`
- all four Typhoon task ABI families appear
- no unexpected host fallback helper names are used for canonical ResNet18 execution

- [ ] **Step 6: Rebuild if `codegen_c_host.cc` changed**

Run: `ninja -C /home/ayd/code/tvm/build tvm tvm_runtime`
Expected: PASS

- [ ] **Step 7: Re-run the codegen slice to verify it passes**

Run: `env PYTHONPATH=/home/ayd/code/tvm/python TVM_LIBRARY_PATH=/home/ayd/code/tvm/build python -m pytest -p no:tvm.testing.plugin tests/python/codegen/test_target_codegen_typhoon.py -v`
Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add python/tvm/tirx/pipeline.py \
        python/tvm/tirx/build.py \
        src/target/source/codegen_c_host.cc \
        tests/python/codegen/test_target_codegen_typhoon.py
git commit -m "feat: preserve full-graph typhoon build path"
```

### Task 5: Extend Runtime Execution For Full-Graph ResNet18

**Files:**
- Modify: `src/runtime/typhoon/typhoon_types.h`
- Modify: `src/runtime/typhoon/typhoon_graph.h`
- Modify: `src/runtime/typhoon/typhoon_graph.cc`
- Modify: `src/runtime/typhoon/typhoon_runtime.cc`
- Modify: `src/runtime/typhoon/typhoon_scheduler.cc`
- Modify: `src/runtime/typhoon/typhoon_cost_model.h`
- Modify: `src/runtime/typhoon/typhoon_cost_model.cc`
- Modify: `tests/python/runtime/test_runtime_typhoon.py`

- [ ] **Step 1: Write the failing runtime safety and execution tests**

```python
def test_runtime_typhoon_rejects_full_graph_live_range_violation():
    graph = build_runtime_graph_with_residual_region_clobber()
    with pytest.raises(RuntimeError, match="live range|hazard|region"):
        submit_and_wait(graph)


def test_runtime_typhoon_rejects_full_graph_out_of_bounds_access():
    graph = build_runtime_graph_with_oob_region_access()
    with pytest.raises(RuntimeError, match="out-of-bounds|size mismatch"):
        submit_and_wait(graph)


def test_runtime_typhoon_rejects_full_graph_uninitialized_read():
    graph = build_runtime_graph_with_uninitialized_read()
    with pytest.raises(RuntimeError, match="uninitialized|region"):
        submit_and_wait(graph)


def test_runtime_typhoon_rejects_full_graph_illegal_dependency():
    graph = build_runtime_graph_with_unknown_dependency()
    with pytest.raises(RuntimeError, match="dependency|unknown"):
        submit_and_wait(graph)


def test_runtime_typhoon_rejects_full_graph_duplicate_task_id():
    graph = build_runtime_graph_with_duplicate_task_ids()
    with pytest.raises(RuntimeError, match="duplicate task_id"):
        submit_and_wait(graph)


def test_runtime_typhoon_rejects_full_graph_concurrent_write_hazard():
    graph = build_runtime_graph_with_concurrent_region_writes()
    with pytest.raises(RuntimeError, match="write hazard|region"):
        submit_and_wait(graph)


def test_runtime_typhoon_executes_full_graph_task_families():
    graph, expected = build_reference_resnet18_fragment_graph()
    actual = submit_and_wait(graph)
    tvm.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4)
```

- [ ] **Step 2: Run the runtime slice to verify it fails**

Run: `env PYTHONPATH=/home/ayd/code/tvm/python TVM_LIBRARY_PATH=/home/ayd/code/tvm/build python -m pytest -p no:tvm.testing.plugin tests/python/runtime/test_runtime_typhoon.py -v`
Expected: FAIL because the current runtime and graph validation are not yet whole-graph-aware.

- [ ] **Step 3: Extend task and trace data structures**

Add fields needed for:

- full-graph region lifetime validation
- richer task metadata
- region-aware trace diagnostics

- [ ] **Step 4: Implement whole-graph runtime safety checks**

In `src/runtime/typhoon/typhoon_graph.cc` and `src/runtime/typhoon/typhoon_runtime.cc`, add validation/execution support for:

- overlapping region declarations
- out-of-bounds accesses
- undeclared region access
- uninitialized region reads
- illegal dependencies
- duplicate task ids
- concurrent write hazards
- full-graph region lifetime misuse
- residual preservation
- complete vector/reshape operator family coverage
- final dense execution

- [ ] **Step 5: Keep scheduler and cost model aligned**

Update `src/runtime/typhoon/typhoon_scheduler.cc` and `src/runtime/typhoon/typhoon_cost_model.cc` so trace entries and byte accounting remain accurate for whole-graph execution while preserving the 4-resource model.

- [ ] **Step 6: Rebuild the C++ runtime and codegen artifacts**

Run: `ninja -C /home/ayd/code/tvm/build tvm tvm_runtime`
Expected: PASS

- [ ] **Step 7: Re-run the runtime slice to verify it passes**

Run: `env PYTHONPATH=/home/ayd/code/tvm/python TVM_LIBRARY_PATH=/home/ayd/code/tvm/build python -m pytest -p no:tvm.testing.plugin tests/python/runtime/test_runtime_typhoon.py -v`
Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add src/runtime/typhoon/typhoon_types.h \
        src/runtime/typhoon/typhoon_graph.h \
        src/runtime/typhoon/typhoon_graph.cc \
        src/runtime/typhoon/typhoon_runtime.cc \
        src/runtime/typhoon/typhoon_scheduler.cc \
        src/runtime/typhoon/typhoon_cost_model.h \
        src/runtime/typhoon/typhoon_cost_model.cc \
        tests/python/runtime/test_runtime_typhoon.py
git commit -m "feat: execute full-graph typhoon runtime safely"
```

### Task 6: Add Full-Graph Relax And ONNXRuntime End-To-End Validation

**Files:**
- Modify: `tests/python/relax/test_relax_typhoon_resnet18.py`
- Create: `tests/python/relax/test_relax_typhoon_resnet18_onnxruntime.py`

- [ ] **Step 1: Write the failing end-to-end full-graph tests**

```python
def test_relax_resnet18_compiles_entire_graph_to_typhoon():
    exe, source = _build_typhoon_vmexec(_import_relax(_load_onnx_model(get_resnet18_model_path())))
    assert source.count("_typhoon_graph") >= 1
    assert source.count("_typhoon_graph") <= 4
    assert "__tvm_ffi_conv2d_typhoon_graph" not in source


def test_relax_resnet18_matches_onnxruntime_elementwise():
    model = _load_onnx_model(get_resnet18_model_path())
    mod = _import_relax(model)
    feed = _build_feed_dict(model)
    actual = _run_typhoon(mod, feed)
    expected = _run_onnxruntime(model, feed)
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4)


def test_relax_resnet18_fails_clearly_when_onnxruntime_is_missing(monkeypatch):
    with monkeypatch.context() as m:
        m.setitem(sys.modules, "onnxruntime", None)
        with pytest.raises(RuntimeError, match="onnxruntime"):
            _import_onnxruntime()
```

- [ ] **Step 2: Run the Relax end-to-end slice to verify it fails**

Run: `env PYTHONPATH=/home/ayd/code/tvm/python TVM_LIBRARY_PATH=/home/ayd/code/tvm/build python -m pytest -p no:tvm.testing.plugin tests/python/relax/test_relax_typhoon_resnet18.py tests/python/relax/test_relax_typhoon_resnet18_onnxruntime.py -v`
Expected: FAIL because the current path is not full-graph Typhoon and does not compare against ONNXRuntime.

- [ ] **Step 3: Add ONNXRuntime-backed reference execution**

In `tests/python/relax/test_relax_typhoon_resnet18_onnxruntime.py`, add:

- a clear hard failure message if `onnxruntime` is missing
- a dedicated test that exercises the missing-dependency failure path
- canonical seeded input generation
- direct ONNXRuntime inference for the local artifact

- [ ] **Step 4: Tighten the Typhoon Relax tests**

Update `tests/python/relax/test_relax_typhoon_resnet18.py` to assert:

- full-graph Typhoon helper presence in actual emitted C
- no host fallback for canonical ResNet18 execution
- runtime trace completes without safety violations
- plain end-to-end elementwise agreement with ONNXRuntime under `rtol=1e-4, atol=1e-4`

- [ ] **Step 5: Re-run the Relax end-to-end slice to verify it passes**

Run: `env PYTHONPATH=/home/ayd/code/tvm/python TVM_LIBRARY_PATH=/home/ayd/code/tvm/build python -m pytest -p no:tvm.testing.plugin tests/python/relax/test_relax_typhoon_resnet18.py tests/python/relax/test_relax_typhoon_resnet18_onnxruntime.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add tests/python/relax/test_relax_typhoon_resnet18.py \
        tests/python/relax/test_relax_typhoon_resnet18_onnxruntime.py
git commit -m "test: validate full-graph typhoon resnet18 against onnxruntime"
```

### Task 7: Add Artifact And Trace Diagnostics For Full-Graph Debugging

**Files:**
- Modify: `src/runtime/typhoon/typhoon_runtime.cc`
- Modify: `tests/python/runtime/test_runtime_typhoon.py`
- Modify: `tests/python/relax/test_relax_typhoon_resnet18.py`

- [ ] **Step 1: Write the failing diagnostics tests**

```python
def test_runtime_typhoon_trace_reports_region_context():
    trace = run_graph_and_get_trace()
    assert {"task_id", "kind", "resource", "start_time", "end_time"} <= trace[0].keys()
    assert "sram_bytes_read" in trace[0]
    assert "sram_bytes_written" in trace[0]
    assert "region_reads" in trace[0]
    assert "region_writes" in trace[0]


def test_relax_typhoon_reports_first_mismatch_context():
    actual = np.array([0.0, 1.0], dtype="float32")
    expected = np.array([0.0, 2.0], dtype="float32")
    message = format_first_mismatch(actual, expected, layer_hint="stage3_block1", task_hint=42)
    assert "first mismatch" in message.lower()
    assert "stage3_block1" in message
    assert "42" in message
    
    
def test_relax_typhoon_onnxruntime_failure_surfaces_context():
    with pytest.raises(AssertionError, match="first mismatch|layer|task"):
        assert_elementwise_close_with_context(
            np.array([0.0, 1.0], dtype="float32"),
            np.array([0.0, 2.0], dtype="float32"),
            rtol=1e-4,
            atol=1e-4,
            layer_hint="final_output",
            task_hint=999,
        )
```

- [ ] **Step 2: Run the diagnostics slice to verify it fails**

Run: `env PYTHONPATH=/home/ayd/code/tvm/python TVM_LIBRARY_PATH=/home/ayd/code/tvm/build python -m pytest -p no:tvm.testing.plugin tests/python/runtime/test_runtime_typhoon.py tests/python/relax/test_relax_typhoon_resnet18.py -v`
Expected: FAIL because the current trace schema is too small for full-graph debug use.

- [ ] **Step 3: Extend trace reporting**

Update runtime trace generation to include enough debugging context for whole-graph failures, including at least:

- `sram_bytes_read`
- `sram_bytes_written`
- `region_reads`
- `region_writes`
- task kind/resource/timing

Update the Relax-side comparison helpers so numerical failures can report:

- first mismatch index
- actual vs expected values
- layer hint
- task hint when available

- [ ] **Step 4: Rebuild the C++ runtime and codegen artifacts**

Run: `ninja -C /home/ayd/code/tvm/build tvm tvm_runtime`
Expected: PASS

- [ ] **Step 5: Re-run the diagnostics slice to verify it passes**

Run: `env PYTHONPATH=/home/ayd/code/tvm/python TVM_LIBRARY_PATH=/home/ayd/code/tvm/build python -m pytest -p no:tvm.testing.plugin tests/python/runtime/test_runtime_typhoon.py tests/python/relax/test_relax_typhoon_resnet18.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/runtime/typhoon/typhoon_runtime.cc \
        tests/python/runtime/test_runtime_typhoon.py \
        tests/python/relax/test_relax_typhoon_resnet18.py
git commit -m "feat: enrich full-graph typhoon trace diagnostics"
```

### Task 8: Run Full Verification Suite

**Files:**
- Test only; no intentional source edits

- [ ] **Step 1: Run focused transform coverage**

Run: `env PYTHONPATH=/home/ayd/code/tvm/python TVM_LIBRARY_PATH=/home/ayd/code/tvm/build python -m pytest -p no:tvm.testing.plugin tests/python/tirx-transform/test_tir_transform_identify_typhoon_resnet18.py tests/python/tirx-transform/test_tir_transform_typhoon_full_graph_plan.py tests/python/tirx-transform/test_tir_transform_plan_typhoon_sram.py tests/python/tirx-transform/test_tir_transform_typhoon_full_graph_sram.py tests/python/tirx-transform/test_tir_transform_build_typhoon_graph.py tests/python/tirx-transform/test_tir_transform_verify_typhoon_graph.py -v`
Expected: PASS

- [ ] **Step 2: Run codegen and runtime coverage**

Run: `env PYTHONPATH=/home/ayd/code/tvm/python TVM_LIBRARY_PATH=/home/ayd/code/tvm/build python -m pytest -p no:tvm.testing.plugin tests/python/codegen/test_target_codegen_typhoon.py tests/python/runtime/test_runtime_typhoon.py -v`
Expected: PASS

- [ ] **Step 3: Run end-to-end Relax and ONNXRuntime coverage**

Run: `env PYTHONPATH=/home/ayd/code/tvm/python TVM_LIBRARY_PATH=/home/ayd/code/tvm/build python -m pytest -p no:tvm.testing.plugin tests/python/relax/test_relax_typhoon_resnet18.py tests/python/relax/test_relax_typhoon_resnet18_onnxruntime.py -v`
Expected: PASS

- [ ] **Step 4: Export an actual full-graph Typhoon C file for inspection**

Run:

```bash
env PYTHONPATH=/home/ayd/code/tvm/python TVM_LIBRARY_PATH=/home/ayd/code/tvm/build python - <<'EOF'
import os
import onnx
import tvm
from tvm import relax
from tvm.relax.frontend.onnx import from_onnx
from tvm.relax import vm_build

model_path = os.environ.get('TYPHOON_RESNET18_ONNX_PATH', os.path.expanduser('~/model/resnet18.onnx'))
model = onnx.load(model_path)
mod = from_onnx(model, keep_params_in_input=True)
with tvm.target.Target({'kind': 'typhoon'}):
    mod = relax.get_pipeline('default')(mod)
builder = relax.ExecBuilder()
tir_mod = vm_build._vmcodegen(builder, mod, exec_mode='bytecode')
lib = tvm.tirx.build(tir_mod, target=tvm.target.Target({'kind': 'typhoon'}))
out_path = '/home/ayd/code/tvm/build/resnet18_full_graph_typhoon_codegen.c'
with open(out_path, 'w', encoding='utf-8') as f:
    f.write(lib.inspect_source())
print(out_path)
EOF
```

Expected: prints `/home/ayd/code/tvm/build/resnet18_full_graph_typhoon_codegen.c`

- [ ] **Step 5: Confirm no-fallback helper expectations in the exported C**

Run: `rg -n "resnet18_typhoon_graph|TVMTyphoonDeclareRegion|TVMTyphoonSubmitGraph|TVMTyphoonAddMatmulTask|TVMTyphoonAddVectorTask|TVMTyphoonAddReshapeTask" /home/ayd/code/tvm/build/resnet18_full_graph_typhoon_codegen.c`
Expected: all patterns are found

- [ ] **Step 6: Commit**

```bash
git commit --allow-empty -m "test: verify full-graph typhoon resnet18 support"
```
