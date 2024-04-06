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
// ============================================================================

#include "gloo_context.h"

#include <chrono>
#include <memory>

#include "gloo/rendezvous/context.h"
#include "gloo/rendezvous/file_store.h"
#include "gloo/rendezvous/prefix_store.h"
// #include "gloo/rendezvous/redis_store.h"
#include "gloo/transport/tcp/device.h"
#if GLOO_HAVE_TRANSPORT_IBVERBS
#include "gloo/transport/ibverbs/device.h"
#endif
#if GLOO_HAVE_TRANSPORT_UCX
#include "gloo/transport/ucx/device.h"
#endif

#if GLOO_HAVE_TRANSPORT_GLEX
#include "gloo/transport/glex/device.h"
#endif

#if GLOO_HAVE_TRANSPORT_GLEX_RDMA_T
#include "gloo/transport/glex_rdma_t/device.h"
#endif

#include <string.h>

#if HAVE_MPI
#include "gloo/mpi/context.h"
#endif

#include "http_store.h"
#include "memory_store.h"
#include "../utils/env_parser.h"

namespace horovod {
namespace common {

// Horovod Gloo rendezvous knobs.
#define HOROVOD_GLOO_TIMEOUT_SECONDS "HOROVOD_GLOO_TIMEOUT_SECONDS"
#define HOROVOD_GLOO_RENDEZVOUS_ADDR "HOROVOD_GLOO_RENDEZVOUS_ADDR"
#define HOROVOD_GLOO_RENDEZVOUS_PORT "HOROVOD_GLOO_RENDEZVOUS_PORT"
#define HOROVOD_GLOO_RENDEZVOUS_ADDR_2 "HOROVOD_GLOO_RENDEZVOUS_ADDR_2"
#define HOROVOD_GLOO_RENDEZVOUS_PORT_2 "HOROVOD_GLOO_RENDEZVOUS_PORT_2"
#define HOROVOD_GLOO_GLOBAL_PREFIX "global_"
#define HOROVOD_GLOO_LOCAL_PREFIX "local_"
#define HOROVOD_GLOO_CROSS_PREFIX "cross_"
#define HOROVOD_GLOO_SHARP_PREFIX "sharp_"
#define HOROVOD_GLOO_SHARP_PREFIX2 "sharp2_"
#define HOROVOD_GLOO_GLEX_PREFIX "glex_"
#define HOROVOD_RANK "HOROVOD_RANK"
#define HOROVOD_SIZE "HOROVOD_SIZE"
#define HOROVOD_LOCAL_RANK "HOROVOD_LOCAL_RANK"
#define HOROVOD_LOCAL_SIZE "HOROVOD_LOCAL_SIZE"
#define HOROVOD_CROSS_RANK "HOROVOD_CROSS_RANK"
#define HOROVOD_CROSS_SIZE "HOROVOD_CROSS_SIZE"
int sharpok = 0;
std::chrono::milliseconds GetTimeoutFromEnv() {
  auto s = std::chrono::seconds(GetIntEnvOrDefault(HOROVOD_GLOO_TIMEOUT_SECONDS, 30));
  return std::chrono::duration_cast<std::chrono::milliseconds>(s);
}

std::shared_ptr<gloo::Context> Rendezvous(const std::string& prefix,
                                          const char* server_addr_env, int server_port,
                                          int rank, int size,
                                          std::shared_ptr<gloo::transport::Device>& dev,
                                          std::chrono::milliseconds timeout) {
  std::unique_ptr<GlooStore> store;
  if (server_addr_env != nullptr) {
    std::string server_addr = server_addr_env;
    store.reset(new HTTPStore(server_addr, server_port, prefix, rank));
  } else {
    store.reset(new MemoryStore());
  }
  LOG(DEBUG) << prefix << " rendezvous started for rank=" << rank << ", size=" << size
             << ", dev={" << dev->str() << "}";

  auto context = std::make_shared<gloo::rendezvous::Context>(rank, size);
  context->setTimeout(timeout);
#if GLOO_HAVE_TIMELINE
  if(getenv("FLEX_TIMELINE_PREFIX")!=nullptr && prefix == HOROVOD_GLOO_GLOBAL_PREFIX && rank == 0){
    std::string gloo_timeline_prefix = getenv("FLEX_TIMELINE_PREFIX");
    std::string gloo_timeline_path = gloo_timeline_prefix +"_"+ prefix+".json";
    fprintf(stdout, "Set Gloo timeline!\n");
    //| 通过设置timeline的路径来启动
    context->setTimeline(gloo_timeline_path);
  }
#endif
  char* gloo_transport = std::getenv("HOROVOD_GLOO_TRANSPORT");
  // if(getenv("GLOO_SHARP_ALLREDUCE") == NULL || strcmp(gloo_transport, "glex_rdma_t")!=0){
  context->connectFullMesh(*store, dev);
//       fprintf(stdout, "sharp init00 \n");
// }
      std::cout << server_addr_env << "env"<<server_port  << std::endl;
// #if GLOO_HAVE_TRANSPORT_GLEX_RDMA_T
//   if(getenv("GLOO_SHARP_ALLREDUCE") != NULL&&prefix==HOROVOD_GLOO_SHARP_PREFIX){
//     fprintf(stdout, "sharp init \n");
//     context->sharp_init("mlx5_4:1");
//   }
//   else
// #endifHOROVOD_GLOO_GLOBAL_PREFIX
// if( getenv("GLOO_SHARP_ALLREDUCE") != NULL&&prefix!=HOROVOD_GLOO_SHARP_PREFIX){

if( getenv("GLOO_SHARP_ALLREDUCE") != NULL){
    if(prefix==HOROVOD_GLOO_SHARP_PREFIX){
      fprintf(stdout, "sharp init \n");
      context->sharp_init("mlx5_6:1");
    }
    // sharpok++;
}

  store->Finalize();
  return context;
}

#if HAVE_MPI
void GlooContext::InitializeFromMPI(MPIContext& mpi_ctx,
                                    const std::string& gloo_iface) {
  if (!enabled_) {
    return;
  }
  fprintf(stdout,"Initialize gloo from MPI with interface:%s\n", gloo_iface.c_str());
  // auto GLOO_SHARP_ALLREDUCE = getenv("GLOO_SHARP_ALLREDUCE");

  // TODO(sihan): Add support for multiple interfaces:
  //  https://github.com/facebookincubator/gloo/issues/190

  std::shared_ptr<::gloo::transport::Device> dev;
  char* gloo_transport = std::getenv("HOROVOD_GLOO_TRANSPORT");
  if(strcmp(gloo_transport, "ibverbs")==0){
#if GLOO_HAVE_TRANSPORT_IBVERBS
    fprintf(stdout, "Gloo initialize from MPI by ibverbs!\n");
    gloo::transport::ibverbs::attr attr;
    attr.port = 1;
    dev = gloo::transport::ibverbs::CreateDevice(attr);
#endif
  }else if(strcmp(gloo_transport, "ucx")==0){
#if GLOO_HAVE_TRANSPORT_UCX
    fprintf(stdout, "Gloo initialize from MPI by ucx!\n");
    dev = gloo::transport::ucx::CreateDevice();
#endif
  }else if(strcmp(gloo_transport, "glex")==0){
#if GLOO_HAVE_TRANSPORT_GLEX
    fprintf(stdout, "Gloo initialize from MPI by glex!\n");
    dev = gloo::transport::glex::CreateDevice();
#endif
  }else if(strcmp(gloo_transport, "glex_rdma_t")==0){
#if GLOO_HAVE_TRANSPORT_GLEX_RDMA_T
    fprintf(stdout, "Gloo initialize from MPI by glex_rdma_t!\n");
    dev = gloo::transport::glex_rdma_t::CreateDevice();
#endif
  }else{
    fprintf(stdout, "Gloo initialize from MPI by tcp!\n");
    gloo::transport::tcp::attr attr;
    attr.iface = gloo_iface;
    attr.ai_family = AF_UNSPEC;
    dev = gloo::transport::tcp::CreateDevice(attr);
  }
  auto timeout = GetTimeoutFromEnv();

  auto context =
      std::make_shared<gloo::mpi::Context>(mpi_ctx.GetMPICommunicator(GLOBAL));
  context->setTimeout(timeout);
  context->connectFullMesh(dev);
  ctx = context;

  auto cross_context =
      std::make_shared<gloo::mpi::Context>(mpi_ctx.GetMPICommunicator(CROSS));
  cross_context->setTimeout(timeout);
  cross_context->connectFullMesh(dev);
  cross_ctx = cross_context;

  auto local_context =
      std::make_shared<gloo::mpi::Context>(mpi_ctx.GetMPICommunicator(LOCAL));
  local_context->setTimeout(timeout);
  local_context->connectFullMesh(dev);
  local_ctx = local_context;
}
#endif

void GlooContext::Initialize(const std::string& gloo_iface) {
  if (!enabled_) {
    return;
  }
  fprintf(stdout,"Initialize gloo with interface:%s\n", gloo_iface.c_str());

  // Create a tcp device for communication
  // TODO(sihan): Add support for multiple interfaces:
  //  https://github.com/facebookincubator/gloo/issues/190
  std::shared_ptr<::gloo::transport::Device> dev;
  std::shared_ptr<::gloo::transport::Device> dev2;  
  std::shared_ptr<::gloo::transport::Device> dev3;    
  std::shared_ptr<::gloo::transport::Device> dev4;      
  char* gloo_transport = std::getenv("HOROVOD_GLOO_TRANSPORT");
if(getenv("DOUBLE_KEY") == NULL){
    if(strcmp(gloo_transport, "ibverbs")==0){
  #if GLOO_HAVE_TRANSPORT_IBVERBS
      fprintf(stdout, "Gloo initialize alone by ibverbs!\n");
      gloo::transport::ibverbs::attr attr;
      attr.port = 1;
      dev = gloo::transport::ibverbs::CreateDevice(attr);
  #endif
    }else if(strcmp(gloo_transport, "ucx")==0){
  #if GLOO_HAVE_TRANSPORT_UCX
      fprintf(stdout, "Gloo initialize alone by ucx!\n");
      dev = gloo::transport::ucx::CreateDevice();
  #endif
    }else if(strcmp(gloo_transport, "glex")==0){
  #if GLOO_HAVE_TRANSPORT_GLEX
      fprintf(stdout, "Gloo initialize alone by glex!\n");
      dev = gloo::transport::glex::CreateDevice();   
  #endif
    }else if(strcmp(gloo_transport, "glex_rdma_t")==0){
  #if GLOO_HAVE_TRANSPORT_GLEX_RDMA_T
      fprintf(stdout, "Gloo initialize alone by glex_rdma_t!\n");
      dev = gloo::transport::glex_rdma_t::CreateDevice();
      // if(getenv("DOUBLE_KEY") != NULL){
      //     gloo::transport::tcp::attr attr;
      //     attr.iface = gloo_iface;
      //     attr.ai_family = AF_UNSPEC;
      //     dev2 = gloo::transport::tcp::CreateDevice(attr);     
      // }
  #endif
    }else{
      fprintf(stdout, "Gloo initialize alone by tcp!\n");
      gloo::transport::tcp::attr attr;
      attr.iface = gloo_iface;
      attr.ai_family = AF_UNSPEC;
      dev = gloo::transport::tcp::CreateDevice(attr);
    }
}
else{
  char* gloo_transport2 = std::getenv("HOROVOD_GLOO_TRANSPORT_2");
  const char* gloo_iface2 = std::getenv("HOROVOD_GLOO_IFACE_2");  

  if(strcmp(gloo_transport, "tcp")==0){      
      gloo::transport::tcp::attr attr;
      attr.iface = gloo_iface;
      attr.ai_family = AF_UNSPEC;
      dev = gloo::transport::tcp::CreateDevice(attr);    
            }
  else{
#if GLOO_HAVE_TRANSPORT_GLEX_RDMA_T  
      fprintf(stdout, "Gloo initialize alone by glex_rdma_t!\n");     
      dev = gloo::transport::glex_rdma_t::CreateDevice();
#endif    

  }      
  if(strcmp(gloo_transport2, "glex_rdma_t")==0 ){
#if GLOO_HAVE_TRANSPORT_GLEX_RDMA_T  
      fprintf(stdout, "Gloo initialize alone by glex_rdma_t!\n");     
      dev3 = gloo::transport::glex_rdma_t::CreateDevice();
#endif      
  }

      gloo::transport::tcp::attr attr2;
      attr2.iface = gloo_iface2;
      attr2.ai_family = AF_UNSPEC;
      dev2 = gloo::transport::tcp::CreateDevice(attr2);    
      fprintf(stdout,"set interface2:%s!\n", gloo_iface2);
  if(getenv("THREE_KEY") != NULL){  
      const char* gloo_iface3 = std::getenv("HOROVOD_GLOO_IFACE_3");
      gloo::transport::tcp::attr attr3;
      attr3.iface = gloo_iface3;
      attr3.ai_family = AF_UNSPEC;
      dev4 = gloo::transport::tcp::CreateDevice(attr3);    
      fprintf(stdout,"set interface3:%s!\n", gloo_iface3);
  }

}



  auto timeout = GetTimeoutFromEnv();
  
  int rank = GetIntEnvOrDefault(HOROVOD_RANK, 0);
  int size = GetIntEnvOrDefault(HOROVOD_SIZE, 1);
  int local_rank = GetIntEnvOrDefault(HOROVOD_LOCAL_RANK, 0);
  int local_size = GetIntEnvOrDefault(HOROVOD_LOCAL_SIZE, 1);
  int cross_rank = GetIntEnvOrDefault(HOROVOD_CROSS_RANK, 0);
  int cross_size = GetIntEnvOrDefault(HOROVOD_CROSS_SIZE, 1);

  auto rendezvous_addr_env = std::getenv(HOROVOD_GLOO_RENDEZVOUS_ADDR);
  auto rendezvous_port = GetIntEnvOrDefault(HOROVOD_GLOO_RENDEZVOUS_PORT, -1);
  auto rendezvous_addr_env2 = std::getenv(HOROVOD_GLOO_RENDEZVOUS_ADDR_2);
  auto rendezvous_port2 = GetIntEnvOrDefault(HOROVOD_GLOO_RENDEZVOUS_PORT_2, -1);  
        // std::cout << rendezvous_addr_env2 << "env"<<rendezvous_port2  << std::endl;
  if (rendezvous_addr_env != nullptr) {
    LOG(DEBUG) << "rendezvous server address: " << rendezvous_addr_env;
  } else {
    LOG(DEBUG) << "no rendezvous server provided, assuming single process execution";
  }
  // if(strcmp(gloo_transport2, "glex_rdma_t")==0 || strcmp(gloo_transport, "glex_rdma_t")==0){
  //   ctx = Rendezvous(HOROVOD_GLOO_GLOBAL_PREFIX,
  //                   rendezvous_addr_env, rendezvous_port,
  //                   rank, size, dev3, timeout);
  //   LOG(DEBUG) << "Global Gloo context initialized.";

  //   local_ctx = Rendezvous(HOROVOD_GLOO_LOCAL_PREFIX + std::to_string(cross_rank),
  //                         rendezvous_addr_env, rendezvous_port,
  //                         local_rank, local_size, dev3, timeout);
  //   LOG(DEBUG) << "Local Gloo context initialized.";

  //   cross_ctx = Rendezvous(HOROVOD_GLOO_CROSS_PREFIX + std::to_string(local_rank),
  //                         rendezvous_addr_env, rendezvous_port,
  //                         cross_rank, cross_size, dev3, timeout);
  // }
  // else{
    ctx = Rendezvous(HOROVOD_GLOO_GLOBAL_PREFIX,
                    rendezvous_addr_env, rendezvous_port,
                    rank, size, dev3, timeout);
    LOG(DEBUG) << "Global Gloo context initialized.";

    local_ctx = Rendezvous(HOROVOD_GLOO_LOCAL_PREFIX + std::to_string(cross_rank),
                          rendezvous_addr_env, rendezvous_port,
                          local_rank, local_size, dev3, timeout);
    LOG(DEBUG) << "Local Gloo context initialized.";

    cross_ctx = Rendezvous(HOROVOD_GLOO_CROSS_PREFIX + std::to_string(local_rank),
                          rendezvous_addr_env, rendezvous_port,
                          cross_rank, cross_size, dev3, timeout);    
  // }
  if(getenv("DOUBLE_KEY") != NULL){                         
      sec_ctx = Rendezvous(HOROVOD_GLOO_SHARP_PREFIX,
                        rendezvous_addr_env, rendezvous_port,
                        rank, size, dev2, timeout);
        LOG(DEBUG) << "sec Gloo context initialized.";                  
  }
  if(getenv("ALLREDUCE_GLEX") != NULL || getenv("SHARP_GLEX") != NULL){
      third_ctx = Rendezvous(HOROVOD_GLOO_GLEX_PREFIX,
                        rendezvous_addr_env, rendezvous_port,
                        rank, size, dev3, timeout);
        LOG(DEBUG) << "third Gloo context initialized.";                          
                     }                     
  if(getenv("THREE_KEY") != NULL){                         
      four_ctx = Rendezvous(HOROVOD_GLOO_SHARP_PREFIX2,
                        rendezvous_addr_env, rendezvous_port,
                        rank, size, dev4, timeout);
        LOG(DEBUG) << "sec Gloo context initialized.";                  
  }

  LOG(DEBUG) << "Cross-node Gloo context initialized.";
  fprintf(stdout, "\nInitialize finish!\n");
}

void GlooContext::Finalize() {
  if (!enabled_) {
    return;
  }

  ctx.reset();
  cross_ctx.reset();
  local_ctx.reset();
  char* gloo_transport = std::getenv("HOROVOD_GLOO_TRANSPORT");
  if(getenv("DOUBLE_KEY") != NULL){
    sec_ctx.reset();
    third_ctx.reset();    
  }
  // if(getenv("DOUBLE_KEY") != NULL && getenv("SHARP_GLEX") != NULL){
  //   third_ctx.reset();
  //   }
}

std::shared_ptr<gloo::Context>
GlooContext::GetGlooContext(Communicator communicator) {
  switch (communicator) {
  case Communicator::GLOBAL:
    return ctx;
  case Communicator::LOCAL:
    return local_ctx;
  case Communicator::CROSS:
    return cross_ctx;
  default:
    throw std::logic_error("Unsupported communicator type.");
  }
}

} // namespace common
} // namespace horovod
