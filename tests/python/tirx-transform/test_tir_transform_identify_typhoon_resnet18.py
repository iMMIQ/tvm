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


def _make_body(input_buffer, weight_buffer, output_buffer, body_kind):
    if body_kind == "stem":
        oc = tvm.tirx.Var("oc", "int32")
        oh = tvm.tirx.Var("oh", "int32")
        ow = tvm.tirx.Var("ow", "int32")
        store = tvm.tirx.BufferStore(
            output_buffer,
            tvm.tirx.BufferLoad(input_buffer, [0, 0, oh * 2, ow * 2])
            + tvm.tirx.BufferLoad(weight_buffer, [oc, 0, 0, 0]),
            [0, oc, oh, ow],
        )
        return tvm.tirx.For(
            oc,
            tvm.tirx.const(0, "int32"),
            tvm.tirx.const(64, "int32"),
            tvm.tirx.ForKind.SERIAL,
            tvm.tirx.For(
                oh,
                tvm.tirx.const(0, "int32"),
                tvm.tirx.const(112, "int32"),
                tvm.tirx.ForKind.SERIAL,
                tvm.tirx.For(
                    ow,
                    tvm.tirx.const(0, "int32"),
                    tvm.tirx.const(112, "int32"),
                    tvm.tirx.ForKind.SERIAL,
                    store,
                ),
            ),
        )
    if body_kind == "padded_stem":
        pad_buffer = tvm.tirx.decl_buffer((1, 3, 230, 230), "float32", name="pad_temp")
        nn = tvm.tirx.Var("nn", "int32")
        cc = tvm.tirx.Var("cc", "int32")
        yy = tvm.tirx.Var("yy", "int32")
        xx = tvm.tirx.Var("xx", "int32")
        rc = tvm.tirx.Var("rc", "int32")
        ry = tvm.tirx.Var("ry", "int32")
        rx = tvm.tirx.Var("rx", "int32")
        pi = tvm.tirx.Var("pi", "int32")
        pj = tvm.tirx.Var("pj", "int32")
        pad_store = tvm.tirx.BufferStore(
            pad_buffer,
            tvm.tirx.if_then_else(
                tvm.tirx.all(pi >= 3, pi < 227, pj >= 3, pj < 227),
                tvm.tirx.BufferLoad(input_buffer, [0, cc, pi - 3, pj - 3]),
                tvm.tirx.const(0, "float32"),
            ),
            [0, cc, pi, pj],
        )
        conv_store = tvm.tirx.BufferStore(
            output_buffer,
            tvm.tirx.BufferLoad(output_buffer, [nn, yy, xx, cc])
            + tvm.tirx.BufferLoad(pad_buffer, [nn, rc, yy * 2 + ry, xx * 2 + rx])
            * tvm.tirx.BufferLoad(weight_buffer, [cc, rc, ry, rx]),
            [nn, yy, xx, cc],
        )
        return tvm.tirx.stmt_seq(
            tvm.tirx.For(
                cc,
                tvm.tirx.const(0, "int32"),
                tvm.tirx.const(3, "int32"),
                tvm.tirx.ForKind.SERIAL,
                tvm.tirx.For(
                    pi,
                    tvm.tirx.const(0, "int32"),
                    tvm.tirx.const(230, "int32"),
                    tvm.tirx.ForKind.SERIAL,
                    tvm.tirx.For(
                        pj,
                        tvm.tirx.const(0, "int32"),
                        tvm.tirx.const(230, "int32"),
                        tvm.tirx.ForKind.SERIAL,
                        pad_store,
                    ),
                ),
            ),
            tvm.tirx.For(
                nn,
                tvm.tirx.const(0, "int32"),
                tvm.tirx.const(1, "int32"),
                tvm.tirx.ForKind.SERIAL,
                tvm.tirx.For(
                    cc,
                    tvm.tirx.const(0, "int32"),
                    tvm.tirx.const(64, "int32"),
                    tvm.tirx.ForKind.SERIAL,
                    tvm.tirx.For(
                        yy,
                        tvm.tirx.const(0, "int32"),
                        tvm.tirx.const(112, "int32"),
                        tvm.tirx.ForKind.SERIAL,
                        tvm.tirx.For(
                            xx,
                            tvm.tirx.const(0, "int32"),
                            tvm.tirx.const(112, "int32"),
                            tvm.tirx.ForKind.SERIAL,
                            tvm.tirx.For(
                                rc,
                                tvm.tirx.const(0, "int32"),
                                tvm.tirx.const(3, "int32"),
                                tvm.tirx.ForKind.SERIAL,
                                tvm.tirx.For(
                                    ry,
                                    tvm.tirx.const(0, "int32"),
                                    tvm.tirx.const(7, "int32"),
                                    tvm.tirx.ForKind.SERIAL,
                                    tvm.tirx.For(
                                        rx,
                                        tvm.tirx.const(0, "int32"),
                                        tvm.tirx.const(7, "int32"),
                                        tvm.tirx.ForKind.SERIAL,
                                        conv_store,
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        )
    if body_kind == "trivial":
        return tvm.tirx.Evaluate(0)
    raise ValueError(f"Unsupported body_kind: {body_kind}")


def _make_func(
    input_shape,
    output_shape,
    weight_shape=(64, 3, 7, 7),
    target_kind="typhoon",
    body_kind="stem",
):
    input_buffer = tvm.tirx.decl_buffer(input_shape, "float32", name="input")
    weight_buffer = tvm.tirx.decl_buffer(weight_shape, "float32", name="weight")
    output_buffer = tvm.tirx.decl_buffer(output_shape, "float32", name="output")
    body = _make_body(input_buffer, weight_buffer, output_buffer, body_kind)
    return tvm.tirx.PrimFunc([input_buffer, weight_buffer, output_buffer], body).with_attr(
        "target", tvm.target.Target({"kind": target_kind})
    )


def _make_mod(
    input_shape,
    output_shape,
    weight_shape=(64, 3, 7, 7),
    target_kind="typhoon",
    body_kind="stem",
):
    return tvm.IRModule.from_expr(
        _make_func(input_shape, output_shape, weight_shape, target_kind, body_kind)
    )


def build_non_resnet18_tir_module():
    return _make_mod((1, 3, 128, 128), (1, 32, 64, 64))


def build_resnet18_stem_tir_module():
    return _make_mod((1, 3, 224, 224), (1, 64, 112, 112))


def build_non_typhoon_resnet18_stem_module():
    return _make_mod((1, 3, 224, 224), (1, 64, 112, 112), target_kind="llvm")


def build_trivial_body_resnet18_stem_module():
    return _make_mod((1, 3, 224, 224), (1, 64, 112, 112), body_kind="trivial")


def build_padded_resnet18_stem_module():
    return _make_mod((1, 3, 224, 224), (1, 64, 112, 112), body_kind="padded_stem")


def test_typhoon_resnet18_rejects_non_resnet18_graph():
    mod = build_non_resnet18_tir_module()
    with pytest.raises(ValueError, match="ResNet18"):
        tvm.tirx.transform.IdentifyTyphoonResNet18()(mod)


def test_typhoon_resnet18_ignores_non_typhoon_modules():
    mod = build_non_typhoon_resnet18_stem_module()
    out = tvm.tirx.transform.IdentifyTyphoonResNet18()(mod)
    assert "typhoon_resnet18_plan" not in out.attrs


def test_typhoon_resnet18_rejects_trivial_body_with_stem_shapes():
    mod = build_trivial_body_resnet18_stem_module()
    with pytest.raises(ValueError, match="ResNet18"):
        tvm.tirx.transform.IdentifyTyphoonResNet18()(mod)


def test_typhoon_resnet18_rejects_modules_with_multiple_typhoon_funcs():
    mod = tvm.IRModule(
        {
            "main": _make_func((1, 3, 224, 224), (1, 64, 112, 112)).with_attr(
                "global_symbol", "main"
            ),
            "aux": _make_func((1, 3, 224, 224), (1, 64, 112, 112)).with_attr(
                "global_symbol", "aux"
            ),
        }
    )
    with pytest.raises(ValueError, match="single|ResNet18"):
        tvm.tirx.transform.IdentifyTyphoonResNet18()(mod)


def test_typhoon_resnet18_accepts_fixed_shape_resnet18_conv_stem():
    mod = build_resnet18_stem_tir_module()
    out = tvm.tirx.transform.IdentifyTyphoonResNet18()(mod)
    assert "typhoon_resnet18_plan" in out.attrs

    plan = json.loads(out.attrs["typhoon_resnet18_plan"])
    assert plan["model"] == "resnet18"
    assert plan["recognized_scope"] == "stem"
    assert plan["input_shape"] == [1, 3, 224, 224]
    assert plan["dtype"] == "float32"
    assert plan["layers"][0]["op_name"] == "stem_conv"
    assert plan["layers"][0]["block_id"] == 0
    assert plan["layers"][0]["weight_shape"] == [64, 3, 7, 7]


def test_typhoon_resnet18_accepts_padded_stem_lowering_form():
    mod = build_padded_resnet18_stem_module()
    out = tvm.tirx.transform.IdentifyTyphoonResNet18()(mod)
    assert "typhoon_resnet18_plan" in out.attrs
