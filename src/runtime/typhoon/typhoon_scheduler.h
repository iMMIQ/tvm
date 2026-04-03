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

#ifndef TVM_RUNTIME_TYPHOON_SCHEDULER_H_
#define TVM_RUNTIME_TYPHOON_SCHEDULER_H_

#include "typhoon_cost_model.h"
#include "typhoon_graph.h"

#include <vector>

namespace tvm {
namespace runtime {
namespace typhoon {

class TyphoonScheduler {
 public:
  explicit TyphoonScheduler(TyphoonHWConfig hw = {});

  std::vector<TyphoonTraceRecord> Run(const TyphoonGraphBuilder& graph) const;

 private:
  TyphoonHWConfig hw_;
};

std::string SerializeTraceToJSON(const std::vector<TyphoonTraceRecord>& trace);

}  // namespace typhoon
}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_TYPHOON_SCHEDULER_H_
