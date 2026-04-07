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

import json

import pytest
import tvm

from tests.python.tirx_transform.typhoon_resnet18_test_utils import (
    build_full_graph_plan_module,
    build_invalid_residual_plan_module,
)


def test_typhoon_sram_plan_tracks_full_graph_live_ranges():
    mod = build_full_graph_plan_module()
    out = tvm.tirx.transform.PlanTyphoonSRAM()(mod)
    plan = json.loads(out.attrs["typhoon_sram_plan"])
    assert any(region["name"] == "residual" for region in plan["regions"])
    assert all(
        "offset" in region and "size" in region and "alignment" in region
        for region in plan["regions"]
    )
    assert any(item["last_use_layer_id"] > item["producer_layer_id"] for item in plan["live_ranges"])
    assert any(
        "consumer_window" in item and "reuse_after_layer_id" in item for item in plan["live_ranges"]
    )
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
        for rhs in spans[i + 1 :]:
            assert max(lhs[1], rhs[1]) >= min(lhs[2], rhs[2])


def test_typhoon_sram_plan_rejects_invalid_residual_reuse():
    mod = build_invalid_residual_plan_module()
    with pytest.raises(ValueError, match="residual|live range|reuse"):
        tvm.tirx.transform.PlanTyphoonSRAM()(mod)
