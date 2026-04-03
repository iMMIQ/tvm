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

#include "typhoon_scheduler.h"

#include <algorithm>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace tvm {
namespace runtime {
namespace typhoon {

namespace {

int ResourceIndex(TaskKind kind) {
  switch (kind) {
    case TaskKind::kDMA:
      return 0;
    case TaskKind::kMatmul:
      return 1;
    case TaskKind::kVector:
      return 2;
    case TaskKind::kReshape:
      return 3;
  }
  return 0;
}

const char* ResourceName(TaskKind kind) {
  switch (kind) {
    case TaskKind::kDMA:
      return "dma";
    case TaskKind::kMatmul:
      return "matrix";
    case TaskKind::kVector:
      return "vector";
    case TaskKind::kReshape:
      return "reshape";
  }
  return "unknown";
}

const char* KindName(TaskKind kind) {
  switch (kind) {
    case TaskKind::kDMA:
      return "dma";
    case TaskKind::kMatmul:
      return "matmul";
    case TaskKind::kVector:
      return "vector";
    case TaskKind::kReshape:
      return "reshape";
  }
  return "unknown";
}

struct RunningTask {
  int64_t end_time;
  size_t task_index;
  bool operator>(const RunningTask& other) const {
    if (end_time != other.end_time) {
      return end_time > other.end_time;
    }
    return task_index > other.task_index;
  }
};

}  // namespace

TyphoonScheduler::TyphoonScheduler(TyphoonHWConfig hw) : hw_(std::move(hw)) {}

std::vector<TyphoonTraceRecord> TyphoonScheduler::Run(const TyphoonGraphBuilder& graph) const {
  const auto& tasks = graph.tasks();
  std::unordered_map<int32_t, size_t> task_index = graph.task_index();
  std::vector<int> remaining_deps(tasks.size(), 0);
  std::vector<std::vector<size_t>> users(tasks.size());
  for (size_t i = 0; i < tasks.size(); ++i) {
    remaining_deps[i] = static_cast<int>(tasks[i].deps.size());
    for (int32_t dep_id : tasks[i].deps) {
      users.at(task_index.at(dep_id)).push_back(i);
    }
  }

  std::vector<std::vector<size_t>> ready(4);
  for (size_t i = 0; i < tasks.size(); ++i) {
    if (remaining_deps[i] == 0) {
      ready[ResourceIndex(tasks[i].kind)].push_back(i);
    }
  }

  auto ready_cmp = [&tasks](size_t lhs, size_t rhs) {
    if (tasks[lhs].task_id != tasks[rhs].task_id) {
      return tasks[lhs].task_id < tasks[rhs].task_id;
    }
    return lhs < rhs;
  };
  for (auto& queue : ready) {
    std::sort(queue.begin(), queue.end(), ready_cmp);
  }

  std::vector<int64_t> resource_free_at(4, 0);
  std::vector<int64_t> end_time(tasks.size(), -1);
  std::vector<TyphoonTraceRecord> trace;
  std::priority_queue<RunningTask, std::vector<RunningTask>, std::greater<RunningTask>> running;
  std::unordered_set<int32_t> active_writer_regions;
  size_t completed = 0;
  int64_t current_time = 0;

  auto start_ready_tasks = [&](int64_t now) {
    for (int resource = 0; resource < 4; ++resource) {
      if (resource_free_at[resource] > now || ready[resource].empty()) {
        continue;
      }
      size_t task_index_value = ready[resource].front();
      ready[resource].erase(ready[resource].begin());
      const auto& task = tasks[task_index_value];
      for (int32_t region_id : task.writes) {
        if (active_writer_regions.count(region_id)) {
          throw std::runtime_error("Typhoon runtime detected concurrent write hazard on region " +
                                   std::to_string(region_id));
        }
      }

      int64_t latency = EstimateLatency(task, hw_);
      int64_t finish_time = now + latency;
      resource_free_at[resource] = finish_time;
      for (int32_t region_id : task.writes) {
        active_writer_regions.insert(region_id);
      }
      running.push({finish_time, task_index_value});
      trace.push_back({task.task_id,
                       KindName(task.kind),
                       ResourceName(task.kind),
                       now,
                       finish_time,
                       EstimateSramBytesRead(task),
                       EstimateSramBytesWritten(task)});
    }
  };

  start_ready_tasks(current_time);
  while (completed < tasks.size()) {
    if (running.empty()) {
      throw std::runtime_error("Typhoon scheduler found no runnable tasks");
    }

    RunningTask event = running.top();
    running.pop();
    current_time = event.end_time;
    const auto& task = tasks[event.task_index];
    for (int32_t region_id : task.writes) {
      active_writer_regions.erase(region_id);
    }
    end_time[event.task_index] = current_time;
    ++completed;

    for (size_t user : users[event.task_index]) {
      --remaining_deps[user];
      if (remaining_deps[user] == 0) {
        auto& queue = ready[ResourceIndex(tasks[user].kind)];
        queue.push_back(user);
        std::sort(queue.begin(), queue.end(), ready_cmp);
      }
    }
    start_ready_tasks(current_time);
  }

  std::sort(trace.begin(), trace.end(), [](const TyphoonTraceRecord& lhs, const TyphoonTraceRecord& rhs) {
    if (lhs.start_time != rhs.start_time) {
      return lhs.start_time < rhs.start_time;
    }
    return lhs.task_id < rhs.task_id;
  });
  return trace;
}

std::string SerializeTraceToJSON(const std::vector<TyphoonTraceRecord>& trace) {
  std::ostringstream os;
  os << "[";
  for (size_t i = 0; i < trace.size(); ++i) {
    const auto& record = trace[i];
    if (i != 0) {
      os << ",";
    }
    os << "{"
       << "\"task_id\":" << record.task_id << ","
       << "\"kind\":\"" << record.kind << "\","
       << "\"resource\":\"" << record.resource << "\","
       << "\"start_time\":" << record.start_time << ","
       << "\"end_time\":" << record.end_time << ","
       << "\"sram_bytes_read\":" << record.sram_bytes_read << ","
       << "\"sram_bytes_written\":" << record.sram_bytes_written << "}";
  }
  os << "]";
  return os.str();
}

}  // namespace typhoon
}  // namespace runtime
}  // namespace tvm
