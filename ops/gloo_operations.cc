// Copyright 2019 Uber Technologies, Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

#include "gloo_operations.h"

#include "gloo/allgather.h"
#include "gloo/allgatherv.h"
#include "gloo/allreduce.h"
#include "gloo/broadcast.h"
#include "gloo/math.h"
#include "gloo/types.h"
#if GLOO_HAVE_TRANSPORT_SHARP
#include "gloo/sharp_allreduce.h"
#include "gloo/pipeallreduce.h"
#include "gloo/pipeallreduce-s.h"
#include "gloo/pipeallreduce-a.h"
#endif

#include "../common.h"
#include "../global_state.h"

#include <fstream>
#include <chrono>
using namespace std;

namespace horovod {
namespace common {

IGlooAlgorithms* GetAlgorithmsForType(DataType dtype,
                                      GlooContext* gloo_context) {
  switch (dtype) {
  case HOROVOD_UINT8:
    return new GlooAlgorithms<u_int8_t>(gloo_context);
  case HOROVOD_INT8:
    return new GlooAlgorithms<int8_t>(gloo_context);
  case HOROVOD_UINT16:
    return new GlooAlgorithms<u_int16_t>(gloo_context);
  case HOROVOD_INT16:
    return new GlooAlgorithms<int16_t>(gloo_context);
  case HOROVOD_INT32:
    return new GlooAlgorithms<int32_t>(gloo_context);
  case HOROVOD_INT64:
    return new GlooAlgorithms<int64_t>(gloo_context);
  case HOROVOD_FLOAT16:
    return new GlooAlgorithms<gloo::float16>(gloo_context);
  case HOROVOD_FLOAT32:
    return new GlooAlgorithms<float>(gloo_context);
  case HOROVOD_FLOAT64:
    return new GlooAlgorithms<double>(gloo_context);
  case HOROVOD_BOOL:
    return new GlooAlgorithms<bool>(gloo_context);
  default:
    throw std::logic_error("Type " + DataType_Name(dtype) +
                           " is not supported in Gloo mode.");
  }
}

template <typename T>
GlooAlgorithms<T>::GlooAlgorithms(GlooContext* gloo_context)
    : gloo_context_(gloo_context) {}

template <typename T>
void GlooAlgorithms<T>::Allreduce(void* buffer_data, int num_elements) {
  // if(gloo_context_->ctx->rank == 0){
  //   ofstream fout("data_num.txt", std::ios::app);
  //   fout<<num_elements<<std::endl;
  //   fout.close();
  // }
  
  
  // auto GLOO_SHARP_ALLREDUCE = getenv("GLOO_SHARP_ALLREDUCE");&&gloo_context_->GLOO_PIPE_ALLREDUCE==NULL
#if GLOO_HAVE_TRANSPORT_SHARP
  if(gloo_context_->GLOO_SHARP_ALLREDUCE==NULL && getenv("DOUBLE_KEY") == NULL){
#endif
    gloo::AllreduceOptions opts(gloo_context_->ctx);
    opts.setOutput<T>(static_cast<T*>(buffer_data), (size_t) num_elements);

    void (*func)(void*, const void*, const void*, size_t) = &::gloo::sum<T>;
    opts.setReduceFunction(gloo::AllreduceOptions::Func(func));
// #if GLOO_HAVE_TIMELINE
//     opts.setTensorNames(tensor_names);
// #endif
    gloo::allreduce(opts);
#if GLOO_HAVE_TRANSPORT_SHARP
  }else{
    
    if(gloo_context_->GLOO_PIPE_ALLREDUCE!=NULL){
        // char* gloo_transport = std::getenv("HOROVOD_GLOO_TRANSPORT");
    if(getenv("SHARP_GLEX") != NULL){
        gloo::SPipeAllreduceOptions opts(gloo_context_->sec_ctx,gloo_context_->third_ctx);
        opts.setOutput(static_cast<T*>(buffer_data), (size_t) num_elements);
         void (*func)(void*, const void*, const void*, size_t) = &::gloo::sum<T>;
        opts.setReduceFunction(gloo::AllreduceOptions::Func(func));
        gloo::spipe_allreduce(opts);
        }
    else if(getenv("SHARP_ALLREDUCE") != NULL){
        gloo::SPipeAllreduceOptions opts(gloo_context_->sec_ctx,gloo_context_->ctx);
        opts.setOutput(static_cast<T*>(buffer_data), (size_t) num_elements);
         void (*func)(void*, const void*, const void*, size_t) = &::gloo::sum<T>;
        opts.setReduceFunction(gloo::AllreduceOptions::Func(func));
        gloo::spipe_allreduce(opts);
        }
    else if(getenv("ALLREDUCE_GLEX") != NULL){
        gloo::APipeAllreduceOptions opts(gloo_context_->sec_ctx,gloo_context_->third_ctx);
        opts.setOutput(static_cast<T*>(buffer_data), (size_t) num_elements);
         void (*func)(void*, const void*, const void*, size_t) = &::gloo::sum<T>;
        opts.setReduceFunction(gloo::AllreduceOptions::Func(func));
        gloo::apipe_allreduce(opts);
    }
    else if(getenv("ALLREDUCE_ALLREDUCE") != NULL){
        gloo::APipeAllreduceOptions opts(gloo_context_->ctx,gloo_context_->sec_ctx);
        opts.setOutput(static_cast<T*>(buffer_data), (size_t) num_elements);
         void (*func)(void*, const void*, const void*, size_t) = &::gloo::sum<T>;
        opts.setReduceFunction(gloo::AllreduceOptions::Func(func));
        gloo::apipe_allreduce(opts);
    }
    else{
        gloo::PipeAllreduceOptions opts(gloo_context_->ctx,gloo_context_->sec_ctx,gloo_context_->four_ctx);
              // std::cout << "pipe start"<< std::endl;
        opts.setOutput(static_cast<T*>(buffer_data), (size_t) num_elements);
         void (*func)(void*, const void*, const void*, size_t) = &::gloo::sum<T>;
        opts.setReduceFunction(gloo::AllreduceOptions::Func(func));
        gloo::pipe_allreduce(opts);

    }
        // gloo::PipeAllreduceOptions opts(gloo_context_->ctx,gloo_context_->sec_ctx);
        // opts.setOutput(static_cast<T*>(buffer_data), (size_t) num_elements);
        // void (*func)(void*, const void*, const void*, size_t) = &::gloo::sum<T>;
        // opts.setReduceFunction(gloo::AllreduceOptions::Func(func));
        // gloo::pipe_allreduce(opts);

    }
    else{

    gloo::SharpAllreduceOptions opts(gloo_context_->ctx);
    opts.setOutput(static_cast<T*>(buffer_data), (size_t) num_elements);
    gloo::sharp_allreduce(opts);
    
    }

  }

#endif

  

}

template <typename T>
void GlooAlgorithms<T>::Allgather(void* buffer_data, void* buffer_out,
                                  int* recvcounts, int* displcmnts) {
  // create count index
  std::vector<size_t> counts(recvcounts, recvcounts + gloo_context_->ctx->size);

  gloo::AllgathervOptions opts(gloo_context_->ctx);
  opts.setInput<T>(static_cast<T*>(buffer_data) +
                       displcmnts[gloo_context_->ctx->rank],
                   counts[gloo_context_->ctx->rank]);
  opts.setOutput<T>(static_cast<T*>(buffer_out), counts);
// #if GLOO_HAVE_TIMELINE
//   opts.setTensorNames(tensor_names);
// #endif
  gloo::allgatherv(opts);
}

template <typename T>
void GlooAlgorithms<T>::Broadcast(void* buffer_data, int num_elements,
                                  int root_rank) {
  gloo::BroadcastOptions opts(gloo_context_->ctx);
  opts.setRoot(root_rank);
  opts.setOutput<T>(static_cast<T*>(buffer_data), (size_t) num_elements);
// #if GLOO_HAVE_TIMELINE
//   opts.setTensorNames(tensor_names);
// #endif
  gloo::broadcast(opts);
}

template <typename T>
void GlooAlgorithms<T>::setTensorNames(const std::vector<std::string>& names){
  tensor_names = names;
}

template <typename T> int GlooAlgorithms<T>::ElementSize() const {
  return sizeof(T);
}

GlooAllreduce::GlooAllreduce(GlooContext* gloo_context,
                             HorovodGlobalState* global_state)
    : AllreduceOp(global_state), gloo_context_(gloo_context) {}

Status GlooAllreduce::Execute(std::vector<TensorTableEntry>& entries,
                              const Response& response) {
  auto& first_entry = entries[0];

  void* buffer_data;
  int num_elements = (int)NumElements(entries);

  // Copy memory into the fusion buffer.
  auto& timeline = global_state_->timeline;
  std::vector<std::string> tensor_names;
  if (entries.size() > 1) {
    timeline.ActivityStartAll(entries, MEMCPY_IN_FUSION_BUFFER);
    const void* fused_input_data;
    size_t buffer_len;
    MemcpyInFusionBuffer(entries, fused_input_data, buffer_data, buffer_len);
    timeline.ActivityEndAll(entries);
    for(auto& e : entries){
      tensor_names.push_back(e.tensor_name);
    }
  } else {
    buffer_data = (void*)first_entry.output->data();
    std::memcpy(buffer_data, first_entry.tensor->data(),
                (size_t)first_entry.tensor->size());
    tensor_names.push_back(first_entry.tensor_name);
  }

  // Do allreduce.
  // ofstream fout("allreduce time.txt", std::ios::app);

  timeline.ActivityStartAll(entries, GLOO_ALLREDUCE);
  std::unique_ptr<IGlooAlgorithms> gloo_algos(
      GetAlgorithmsForType(first_entry.tensor->dtype(), gloo_context_));
  gloo_algos->setTensorNames(tensor_names);
  // auto start = std::chrono::high_resolution_clock::now();
  gloo_algos->Allreduce(buffer_data, num_elements);
  // auto end = std::chrono::high_resolution_clock::now();

  // if(gloo_context_->ctx->rank == 0){
  //   fout <<num_elements<<"  "<< std::chrono::nanoseconds(end - start).count() << std::endl;
  // }
  // fout.close();
  timeline.ActivityEndAll(entries);

  // Copy memory out of the fusion buffer.
  if (entries.size() > 1) {
    timeline.ActivityStartAll(entries, MEMCPY_OUT_FUSION_BUFFER);
    MemcpyOutFusionBuffer(buffer_data, entries);
    timeline.ActivityEndAll(entries);
  }

  return Status::OK();
}

bool GlooAllreduce::Enabled(const ParameterManager& param_manager,
                            const std::vector<TensorTableEntry>& entries,
                            const Response& response) const {
  return true;
}

GlooAllgather::GlooAllgather(GlooContext* gloo_context,
                             HorovodGlobalState* global_state)
    : AllgatherOp(global_state), gloo_context_(gloo_context) {}

bool GlooAllgather::Enabled(const ParameterManager& param_manager,
                            const std::vector<TensorTableEntry>& entries,
                            const Response& response) const {
  return true;
}

Status GlooAllgather::Execute(std::vector<TensorTableEntry>& entries,
                              const Response& response) {
  auto& timeline = global_state_->timeline;

  // Sizes of subcomponents of each entry from all ranks
  auto** entry_component_sizes = new int64_t*[entries.size()];

  // Offset of each subcomponent of every entry in the final buffer after
  // allgatherv
  auto** entry_component_offsets = new int64_t*[entries.size()];

  int global_size = global_state_->controller->GetSize();
  auto* recvcounts = new int[global_size]();
  auto* displcmnts = new int[global_size]();

  for (size_t ec = 0; ec < entries.size(); ++ec) {
    entry_component_sizes[ec] = new int64_t[global_size]();
    entry_component_offsets[ec] = new int64_t[global_size]();
  }

  auto& first_entry = entries[0];

  timeline.ActivityStartAll(entries, ALLOCATE_OUTPUT);
  Status status =
      AllocateOutput(entries, response, entry_component_sizes, recvcounts);
  if (!status.ok()) {
    return status;
  }
  timeline.ActivityEndAll(entries);

  SetDisplacements(recvcounts, displcmnts);
  SetEntryComponentOffsets(entries, entry_component_sizes, recvcounts,
                           entry_component_offsets);

  std::unique_ptr<IGlooAlgorithms> gloo_algos(
      GetAlgorithmsForType(first_entry.tensor->dtype(), gloo_context_));
  int element_size = gloo_algos->ElementSize();

  void* sendbuf = nullptr;
  void* buffer_data;
  std::vector<std::string> tensor_names;
  if (entries.size() > 1) {
    timeline.ActivityStartAll(entries, MEMCPY_IN_FUSION_BUFFER);
    MemcpyInFusionBuffer(entries, displcmnts, element_size, buffer_data);
    sendbuf = buffer_data;
    timeline.ActivityEndAll(entries);
    for(auto& e : entries){
      tensor_names.push_back(e.tensor_name);
    }
  } else {
    // need to move input data to its corresponding location in the output
    sendbuf = (void*)first_entry.tensor->data();
    buffer_data = (void*)first_entry.output->data();
    int buffer_offset = displcmnts[gloo_context_->ctx->rank] * element_size;
    std::memcpy((uint8_t*)buffer_data + buffer_offset, sendbuf,
                (size_t)first_entry.tensor->size());
    sendbuf = buffer_data;
    tensor_names.push_back(first_entry.tensor_name);
  }

  // call gloo allgather api
  gloo_algos->setTensorNames(tensor_names);
  global_state_->timeline.ActivityStartAll(entries, GLOO_ALLGATHER);
  gloo_algos->Allgather(sendbuf, buffer_data, recvcounts, displcmnts);
  global_state_->timeline.ActivityEndAll(entries);

  // if multiple tensors are gathered, restore the sequence from output
  if (entries.size() > 1) {
    timeline.ActivityStartAll(entries, MEMCPY_OUT_FUSION_BUFFER);
    MemcpyOutFusionBuffer(entry_component_offsets, entry_component_sizes,
                          buffer_data, element_size, entries);
    timeline.ActivityEndAll(entries);
  }

  delete[] recvcounts;
  delete[] displcmnts;

  for (size_t ec = 0; ec < entries.size(); ++ec) {
    delete[] entry_component_sizes[ec];
    delete[] entry_component_offsets[ec];
  }
  delete[] entry_component_sizes;
  delete[] entry_component_offsets;

  return Status::OK();
}

GlooBroadcast::GlooBroadcast(GlooContext* gloo_context,
                             HorovodGlobalState* global_state)
    : BroadcastOp(global_state), gloo_context_(gloo_context) {}

Status GlooBroadcast::Execute(std::vector<TensorTableEntry>& entries,
                              const Response& response) {
  assert(entries.size() == 1);
  auto e = entries[0];

  // On root rank, MPI_Bcast sends data, on other ranks it receives data.
  // for gloo broadcast, only output needs to be set if inplace

  void* data_ptr;
  if (global_state_->controller->GetRank() == e.root_rank) {
    data_ptr = (void*)e.tensor->data();
  } else {
    data_ptr = (void*)e.output->data();
  }
  std::vector<std::string> tensor_names;
  tensor_names.push_back(e.tensor_name);

  global_state_->timeline.ActivityStartAll(entries, GLOO_BCAST);
  std::unique_ptr<IGlooAlgorithms> gloo_algos(
      GetAlgorithmsForType(e.tensor->dtype(), gloo_context_));
  gloo_algos->setTensorNames(tensor_names);
  gloo_algos->Broadcast(data_ptr, (int)e.tensor->shape().num_elements(),
                        e.root_rank);
  global_state_->timeline.ActivityEndAll(entries);

  return Status::OK();
}

bool GlooBroadcast::Enabled(const ParameterManager& param_manager,
                            const std::vector<TensorTableEntry>& entries,
                            const Response& response) const {
  return true;
}

} // namespace common
} // namespace horovod
