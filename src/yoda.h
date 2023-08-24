#ifndef __YODA__
#define __YODA__

#include "common.h"
#include "iostream"

struct yodaParam {
  int nGPUs;
  int nThreads;
  struct Distributed {
    int t_world_size;
    int p_world_size;
    int p_rank;
    std::unordered_map<int, std::vector<ncclComm_t>> tensor_parallel_comms;
    std::unordered_map<int, std::vector<ncclComm_t>> pipeline_parallel_comms;
    std::unordered_map<int, std::vector<ncclUniqueId>> tensor_parallel_uids;
    std::unordered_map<int, ncclUniqueId> pipeline_parallel_uids;
    Distributed(int tensor_parallel_size, int pipeline_parallel_size = 1)
        : t_world_size(tensor_parallel_size),
          p_world_size(pipeline_parallel_size) {
      if (p_world_size > 1) {
        printf("pipeline parallelism is not supported yet\n");
      }
      p_rank = 0;
      // check if the tensor parallel size is a power of 2
      if ((t_world_size & (t_world_size - 1)) != 0) {
        printf("tensor parallel size must be a power of 2\n");
        exit(1);
      }
      // initialize the nccl unique ids for adaptive tensor parallelism
      for (int t_size = 2; t_size <= t_world_size; t_size *= 2) {
        ncclUniqueId uid;
        ncclGetUniqueId(&uid);
      }
      if (p_rank == 0) {
        for (int i = 0; i < p_world_size; i++) {
          ncclGetUniqueId(&uids[i]);
        }
      }
    }
    ncclComm_t getLocalComm(int gpu_id, int t_rank_size) {
      return tensor_parallel_comms[t_rank_size][gpu_id];
    }
  };
  Distributed dist;
};

#endif