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

import tvm


def test_typhoon_region_and_task_builders():
    region = tvm.tirx.typhoon.region_decl(7, 0, 0, 4096, 64, 0, "A")
    task = tvm.tirx.typhoon.task_matmul(7, 1, 0, 1, 2, 64, 64, 64, 1, 0, [4, 9])

    assert isinstance(region, tvm.tirx.Evaluate)
    assert isinstance(task, tvm.tirx.Evaluate)
    assert region.value.op.name == "tirx.typhoon.region_decl"
    assert task.value.op.name == "tirx.typhoon.task_matmul"
    assert [arg.value for arg in region.value.args[:6]] == [7, 0, 0, 4096, 64, 0]
    assert region.value.args[6].value == "A"
    assert [arg.value for arg in task.value.args] == [7, 1, 0, 1, 2, 64, 64, 64, 1, 0, 2, 4, 9]
