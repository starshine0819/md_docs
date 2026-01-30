# NCCL device/all_gather.h 函数实现详细分析

## 文件概述

`all_gather.h` 是 NCCL 设备端代码的 All-Gather 集合通信操作实现文件，位于 `/root/nccl/src/device/` 目录下。该文件实现了 All-Gather 操作的各种算法和协议变体，支持 Ring、PAT、NVLS 和 CollNet Direct 等不同拓扑结构。

## 核心函数详细分析

### 1. runRing 函数模板

```cpp
template<typename T, typename RedOp, typename Proto, bool isNetOffload = false>
__device__ __forceinline__ void runRing(int tid, int nthreads, struct ncclDevWorkColl* work)
```

**功能**：实现基于环形拓扑的 All-Gather 操作。

**实现原理**：
1. **初始化参数**：
   ```cpp
   ncclRing *ring = &ncclShmem.channel.ring;
   const int *ringRanks = ring->userRanks;
   const int nranks = ncclShmem.comm.nRanks;
   ssize_t count, partOffset, partCount, chunkCount;
   ncclCollCbdPart(work, ncclShmem.channelId, Proto::Id, sizeof(T), &count, &partOffset, &partCount, &chunkCount);
   ```
   - 获取环形拓扑信息
   - 计算分区参数

2. **网络卸载处理**：
   ```cpp
   if (isNetOffload) {
     workNthreads = WARP_SIZE;
     chunkCount = NCCL_MAX_NET_SIZE;
   } else {
     workNthreads = nthreads;
   }
   ```
   - 如果启用网络卸载，只使用一个 warp
   - 设置网络最大尺寸

3. **主循环处理**：
   ```cpp
   for (size_t elemOffset = 0; elemOffset < partCount; elemOffset += chunkCount) {
     /////////////// begin AllGather steps ///////////////
     nelem = min(chunkCount, partCount - elemOffset);
     dataOffset = partOffset + elemOffset;

     // step 0: push data to next GPU
     rankDest = ringRanks[0];
     offset = dataOffset + rankDest * count;

     if ((inputBuf + dataOffset == outputBuf + offset) || isNetOffload) {
       prims.directSend(dataOffset, offset, nelem);
     } else {
       prims.directCopySend(dataOffset, offset, nelem);
     }
   }
   ```
   - 按块处理数据
   - 第一步：将数据推送到下一个 GPU
   - 使用直接发送或直接复制发送

4. **中间步骤**：
   ```cpp
   // k-2 steps: copy to next GPU
   for (int j = 1; j < nranks - 1; ++j) {
     rankDest = ringRanks[nranks - j];
     offset = dataOffset + rankDest * count;
     prims.directRecvCopyDirectSend(offset, offset, nelem);
   }
   ```
   - 中间步骤：复制数据到下一个 GPU
   - 使用直接接收-复制-直接发送操作

5. **最终步骤**：
   ```cpp
   // Make final copy from buffer to dest.
   rankDest = ringRanks[1];
   offset = dataOffset + rankDest * count;
   prims.directRecv(offset, nelem);
   ```
   - 最终步骤：接收最终数据

6. **非工作线程处理**：
   ```cpp
   else if (inputBuf != outputBuf + ringRanks[0] * count) {
     inputBuf = inputBuf + partOffset;
     outputBuf = outputBuf + partOffset + ringRanks[0] * count;
     reduceCopy<COLL_UNROLL, RedOp, T, 0, 1, 1, 0, 1, 1, /*PreOpSrcs=*/0>
       (tid - workNthreads, nthreads - workNthreads, work->redOpArg, false, 1, (void**)&inputBuf, 1, (void**)&outputBuf, partCount);
   }
   ```
   - 非工作线程执行复制操作

### 2. Ring Simple 协议实现

#### 2.1 RunWorkColl 模板特化 (Simple 协议)

```cpp
template<typename T, typename RedOp>
struct RunWorkColl<ncclFuncAllGather, T, RedOp, NCCL_ALGO_RING, NCCL_PROTO_SIMPLE> {
  __device__ __forceinline__ void run(int tid, int nthreads, struct ncclDevWorkColl* work) {
    bool isNetOffload = work->isOneRPN && work->netRegUsed;
    if (isNetOffload)
      runRing<T, RedOp, ProtoSimple<1, 1>, true>(tid, nthreads, work);
    else
      runRing<T, RedOp, ProtoSimple<ALLGATHER_CHUNKSTEPS/ALLGATHER_SLICESTEPS, ALLGATHER_SLICESTEPS>, false>(tid, nthreads, work);
  }
};
```

**功能**：Ring 算法 + Simple 协议的 All-Gather 实现。

**实现原理**：
- 根据是否启用网络卸载选择不同的参数配置
- 网络卸载模式：使用简单参数
- 普通模式：使用分块和切片参数

#### 2.2 RunWorkColl 模板特化 (LL 协议)

```cpp
template<typename T, typename RedOp>
struct RunWorkColl<ncclFuncAllGather, T, RedOp, NCCL_ALGO_RING, NCCL_PROTO_LL> {
  __device__ __forceinline__ void run(int tid, int nthreads, struct ncclDevWorkColl* work) {
    runRing<T, RedOp, ProtoLL>(tid, nthreads, work);
  }
};
```

**功能**：Ring 算法 + LL 协议的 All-Gather 实现。

#### 2.3 RunWorkColl 模板特化 (LL128 协议)

```cpp
template<typename T, typename RedOp>
struct RunWorkColl<ncclFuncAllGather, T, RedOp, NCCL_ALGO_RING, NCCL_PROTO_LL128> {
  __device__ __forceinline__ void run(int tid, int nthreads, struct ncclDevWorkColl* work) {
    runRing<T, RedOp, ProtoLL128>(tid, nthreads, work);
  }
};
```

**功能**：Ring 算法 + LL128 协议的 All-Gather 实现。

### 3. PAT (Pattern-based) 算法实现

#### 3.1 RunWorkColl 模板特化 (PAT 算法)

```cpp
template<typename T, typename RedOp>
struct RunWorkColl<ncclFuncAllGather, T, RedOp, NCCL_ALGO_PAT, NCCL_PROTO_SIMPLE> {
  __device__ __forceinline__ void run(int tid, int nthreads, struct ncclDevWorkColl* work) {
#if __CUDA_ARCH__ >= 600
    using Proto = ProtoSimple<1, 1>;
    const int nranks = ncclShmem.comm.nRanks;
    const int rank = ncclShmem.comm.rank;
    size_t count, channelOffset, channelCount, chunkCount;
    ncclCollCbdPart(work, ncclShmem.channelId, Proto::Id, sizeof(T), &count, &channelOffset, &channelCount, &chunkCount);

    static constexpr int nworkers = NCCL_PAT_NWORKERS;
    struct ncclPatShmem* shmem = (struct ncclPatShmem*)ncclScratchForWarp(0);
    uint64_t pollCount = 0;
    __syncthreads(); // Don't start using shared mem until everyone arrives
    for (int i=tid; i<NCCL_SHMEM_PAT_STEPS; i+=nthreads) shmem->patSteps[i].flags = 0;
    if (tid == 0) shmem->localAccSize = 0;
    if (tid == nworkers) shmem->parallelFactor = 0;
    __syncthreads();

    if (tid == nworkers) { // Algo computation thread
      PatAGAlgorithm<T> patAlgo(chunkCount*sizeof(T), NCCL_STEPS, NCCL_PAT_NWORKERS/WARP_SIZE, channelOffset, channelOffset + channelCount, count, chunkCount, rank, nranks);
      int parallelFactor = shmem->parallelFactor = patAlgo.getParallelFactor();
      int step = 0;
      while (1) {
        struct ncclPatStep* ps = shmem->patSteps+(step%NCCL_SHMEM_PAT_STEPS);
        cuda::atomic_ref<int, cuda::thread_scope_block> poll(ps->flags);
        while (poll.load(cuda::memory_order_acquire) != 0) pollCount++; // Wait for workers to be done with step 'step-NCCL_SHMEM_PAT_STEPS'
        patAlgo.getNextOp(ps);
        int last = ps->last;
        step++;
        if (last == 2) break;
      }
    } else if (tid < nworkers) { // Worker threads
      T *inputBuf = (T*)work->sendbuff;
      T *outputBuf = (T*)work->recvbuff;
      int parallelFactor = 0;
      volatile int* pfPtr = &shmem->parallelFactor;
      while (parallelFactor == 0) parallelFactor = *pfPtr;

      int groupSize = nworkers/(WARP_SIZE*parallelFactor) * WARP_SIZE;
      int group = tid / groupSize;
      int nGroups = nworkers / groupSize;
      int tidInGroup = tid - group*groupSize;
      // We don't use recvPeers/sendPeers so let's pass shmem structs instead
      Primitives<T, RedOp, FanSymmetric<1>, 0, Proto, 0> prims
        (tidInGroup, groupSize, (int*)shmem->recvDims, (int*)shmem->sendDims, inputBuf, outputBuf, work->redOpArg, group, 0, 0, nullptr, nullptr, 0, primsModePatAg);

      int step = group;
      while(1) {
        struct ncclPatStep* ps = shmem->patSteps+(step%NCCL_SHMEM_PAT_STEPS);
        cuda::atomic_ref<int, cuda::thread_scope_block> poll(ps->flags);
        while (poll.load(cuda::memory_order_acquire) == 0) pollCount++; // Wait for compute thread
        int last = ps->last;
        prims.patCopy(ps, shmem);
        if (tidInGroup == 0) poll.store(0, cuda::memory_order_release); // Return element to compute thread
        if (last) break;
      }
    }
#endif
  }
};
```

**功能**：PAT (Pattern-based) 算法的 All-Gather 实现。

**实现原理**：
1. **初始化共享内存**：
   - 初始化 PAT 步骤标志
   - 同步所有线程

2. **算法计算线程**：
   - 创建 `PatAGAlgorithm` 对象
   - 计算并行因子
   - 生成 PAT 步骤操作

3. **工作线程**：
   - 等待并行因子
   - 分组工作线程
   - 使用 PAT 模式原语执行复制操作

### 4. NVLS (NVIDIA Virtual Link Service) 算法实现

#### 4.1 Scatterer 结构体

```cpp
template<typename T, typename RedOp>
struct RunWorkColl<ncclFuncAllGather, T, RedOp, NCCL_ALGO_NVLS, NCCL_PROTO_SIMPLE> {
  template<bool BcastSendNotRecv>
  struct Scatterer {
    struct ncclDevWorkColl* work;
    ssize_t chunkSize;
    ssize_t railGridOffset;

    template<int SlicePerChunk, int MinSrcs, int MaxSrcs, int MinDsts, int MaxDsts, int MultimemSrcs, int MultimemDsts>
    __device__ __forceinline__ void operator()(
        int tid, int tn, int slice, int maxSliceSize,
        int nSrcs, void** srcPtrs, int nDsts, void** dstPtrs, int32_t* dstSizes, uint32_t sendDirectFlag, uint32_t recvDirectFlag
      ) {
      static_assert(SlicePerChunk==1, "require: SlicePerChunk==1");
      static_assert(MaxDsts<=1 || MaxSrcs<=1, "require: MaxDsts<=1 || MaxSrcs<=1");

      struct ncclNvls* nvls = &ncclShmem.channel.nvls;
      int nNodes = ncclShmem.comm.nNodes;
      int nRails = nvls->nHeads;
      int part = ncclShmem.channelId - work->channelLo;
      char* inbuf = (char*)work->sendbuff;
      char* outbuf = (char*)work->recvbuff;
      ssize_t countPerRank = work->collnet.count;
      bool inPlace = (inbuf == outbuf + ncclShmem.comm.rank * countPerRank);
      ssize_t railAllBeg = min(railGridOffset + part * chunkSize, nNodes * countPerRank);
      ssize_t railAllEnd = min(railAllBeg + chunkSize, nNodes * countPerRank);
      int railAllSize = railAllEnd - railAllBeg;
      int rail = 0;
      int src = 0;

      if (BcastSendNotRecv) {
        rail = nvls->headRank;
      } else {
        if (work->regUsed) return;
        rail = 0;
      }
      if (tid < nDsts) dstSizes[tid] = railAllSize;
      do {
        int node = railAllBeg / countPerRank;
        int railAllOffset = 0;
        while (railAllOffset < railAllSize) {
          ssize_t railOneBeg = node * countPerRank;
          ssize_t railOneEnd = railOneBeg + countPerRank;
          ssize_t railOneOffset = (railAllBeg + railAllOffset) - railOneBeg;
          int delta = min(railAllEnd, railOneEnd) - (railAllBeg + railAllOffset);
          int rank = ncclShmem.comm.collNetDenseToUserRank[node * nRails + rail];
          ssize_t userOneBeg = rank * countPerRank + railOneOffset;
          int outIsDst = (inPlace && rank == ncclShmem.comm.rank) || BcastSendNotRecv || work->regUsed ? 0 : 1;
          if (nSrcs != 0 && outIsDst + nDsts != 0) {
            reduceCopy<ncclCollUnroll(), RedOp, T,
              /*MultimemSrcs,MinSrcs,MaxSrcs=*/MultimemSrcs, 1, 1,
              /*MultimemDsts=*/MultimemDsts, 0 + MultimemDsts + MinDsts, 1 + MaxDsts,
              /*PreOpSrcs=*/0>
              (tid, tn, 0, false,
                /*nSrcs=*/1, [=]__device__(int s/*==0*/) -> void* {
              return (char*)srcPtrs[src] + railAllOffset;
            },
                /*nDsts=*/outIsDst + nDsts, [=]__device__(int d) -> void* {
              return d < outIsDst ? outbuf + userOneBeg
                : work->regUsed ? (char*)dstPtrs[d - outIsDst] + userOneBeg
                : (char*)dstPtrs[d - outIsDst] + railAllOffset;
            }, delta);
          }
          railAllOffset += delta;
          node += 1;
        }
        rail += 1;
        src += 1;
      } while (!BcastSendNotRecv && src < nRails);
    }
  };
```

**功能**：NVLS 算法的散射器，处理数据在不同轨道间的分布。

**实现原理**：
1. **参数验证**：使用 `static_assert` 验证模板参数
2. **轨道计算**：计算轨道起始和结束位置
3. **数据分段**：按节点和轨道分段处理数据
4. **复制操作**：使用 `reduceCopy` 执行数据复制

#### 4.2 NVLS 主运行函数

```cpp
__device__ __forceinline__ void run(int tid, int/*nthreads*/, struct ncclDevWorkColl* work) {
  struct ncclNvls* nvls = &ncclShmem.channel.nvls;
  int nelem;

  const int nThreadsNetSend = work->oneNode ? 0 : (work->netRegUsed ? WARP_SIZE :  6 * WARP_SIZE);
  const int nThreadsGather = work->regUsed ? roundUp(nvls->nHeads << 2, WARP_SIZE) : 8 * WARP_SIZE;
  const int nThreadsBcast = NCCL_MAX_NTHREADS - nThreadsNetSend - nThreadsGather;

  const int tidEndGather = nThreadsGather;
  const int tidEndNetSend = tidEndGather + nThreadsNetSend;
  const int tidEndBcast = tidEndNetSend + nThreadsBcast;

  if (work->oneNode) {
    // 单节点处理逻辑
  } else {
    // 多节点处理逻辑
  }
}
```

**功能**：NVLS 算法的主运行函数。

**实现原理**：
- **线程分配**：根据工作类型分配不同数量的线程
- **单节点模式**：优化单节点内的 All-Gather
- **多节点模式**：处理跨节点的 All-Gather

### 5. CollNet Direct 算法实现

#### 5.1 CollNet Scatterer 结构体

```cpp
template<typename T, typename RedOp>
struct RunWorkColl<ncclFuncAllGather, T, RedOp, NCCL_ALGO_COLLNET_DIRECT, NCCL_PROTO_SIMPLE> {
  template<bool BcastSendNotRecv>
  struct Scatterer {
    struct ncclDevWorkColl* work;
    ssize_t chunkSize;
    ssize_t railGridOffset;

    template<int SlicePerChunk, int MinSrcs, int MaxSrcs, int MinDsts, int MaxDsts, int MultimemSrcs, int MultimemDsts>
    __device__ __forceinline__ void operator()(
        int tid, int tn, int slice, int maxSliceSize,
        int nSrcs, void** srcPtrs, int nDsts, void** dstPtrs, int32_t* dstSizes, uint32_t sendDirectFlag, uint32_t recvDirectFlag
      ) {
      static_assert(SlicePerChunk==1, "require: SlicePerChunk==1");
      static_assert(MaxDsts<=1 || MaxSrcs<=1, "require: MaxDsts<=1 || MaxSrcs<=1");

      struct ncclDirect* direct = &ncclShmem.channel.collnetDirect;
      int nNodes = ncclShmem.comm.nNodes;
      int nRails = direct->nHeads;
      int part = ncclShmem.channelId - work->channelLo;
      char* inbuf = (char*)work->sendbuff;
      char* outbuf = (char*)work->recvbuff;
      ssize_t countPerRank = work->collnet.count*sizeof(T);
      bool inPlace = (inbuf == outbuf + ncclShmem.comm.rank*countPerRank);

      ssize_t railAllBeg = min(railGridOffset + part*chunkSize, nNodes*countPerRank);
      ssize_t railAllEnd = min(railAllBeg + chunkSize, nNodes*countPerRank);
      int railAllSize = railAllEnd - railAllBeg;
      if (tid < nDsts) dstSizes[tid] = railAllSize;

      int src = 0;
      int rail;
      if (BcastSendNotRecv) {
        rail = direct->headRank;
      } else {
        rail = direct->headRank+1;
        if (rail == nRails) rail = 0;
      }
      do {
        int node = railAllBeg/countPerRank;
        int railAllOffset = 0;
        while (railAllOffset < railAllSize) {
          ssize_t railOneBeg = node*countPerRank;
          ssize_t railOneEnd = railOneBeg + countPerRank;
          ssize_t railOneOffset = (railAllBeg+railAllOffset) - railOneBeg;
          int delta = min(railAllEnd, railOneEnd) - (railAllBeg+railAllOffset);
          int rank = ncclShmem.comm.collNetDenseToUserRank[node*nRails + rail];
          ssize_t userOneBeg = rank*countPerRank + railOneOffset;
          int outIsDst = (inPlace && rank == ncclShmem.comm.rank) ? 0 : 1;
          if (nSrcs != 0 && outIsDst+nDsts != 0) {
            reduceCopy<ncclCollUnroll(), RedOp, T,
                     /*MultimemSrcs,MinSrcs,MaxSrcs=*/0,1,1,
                     /*MultimemDsts=*/0, 0+MinDsts, 1+MaxDsts,
                     /*PreOpSrcs=*/0>
            (tid, tn, 0, false,
             /*nSrcs=*/1, [=]__device__(int s/*==0*/) -> void* {
               return work->regUsed && (recvDirectFlag & NCCL_P2P_READ) ? (char*)srcPtrs[src] + userOneBeg : (char*)srcPtrs[src] + railAllOffset;
             },
             /*nDsts=*/outIsDst+nDsts, [=]__device__(int d) -> void* {
               return d < outIsDst ? outbuf + userOneBeg
                                   : work->regUsed && (sendDirectFlag & NCCL_P2P_WRITE) ? (char*)dstPtrs[d-outIsDst] + userOneBeg
                                   : (char*)dstPtrs[d-outIsDst] + railAllOffset;
             },
             delta);
          }
          railAllOffset += delta;
          node += 1;
        }
        src += 1;
        rail += 1;
        if (rail == nRails) rail = 0;
      } while (!BcastSendNotRecv && src < nRails-1);
    }
  };
```

**功能**：CollNet Direct 算法的散射器，处理直接连接网络的数据传输。

**实现原理**：
1. **轨道计算**：计算当前轨道的数据范围
2. **节点遍历**：按节点遍历处理数据
3. **动态轨道选择**：根据操作类型选择轨道
4. **条件复制**：根据注册使用情况选择不同的内存访问模式

#### 5.2 CollNet 主运行函数

```cpp
__device__ __forceinline__ void run(int tid, int/*nthreads*/, struct ncclDevWorkColl* work) {
  const int part = ncclShmem.channelId - work->channelLo;
  const int nChannels = work->channelHi - work->channelLo + 1;
  struct ncclDirect* direct = &ncclShmem.channel.collnetDirect;
  int const &nNodes = ncclShmem.comm.nNodes;
  ssize_t countPerRank = work->collnet.count;
  size_t chunkSize = work->collnet.chunkCount;
  const int hasDn = (direct->down[0] >= 0) ? 1 : 0;
  bool isMultiRail = (direct->nHeads > 1);
  int nWarps1 = 1;
  int nWarps2 = (isMultiRail ? 2 : 1);
  int nWarps3 = (isMultiRail ? 2 : 0);
  float denom = float(work->nWarps)/float(nWarps1+nWarps2+nWarps3);
  nWarps3 = int(denom*nWarps3);
  nWarps2 = int(denom*nWarps2);
  nWarps1 = work->nWarps - (nWarps2+nWarps3);

  using Proto = ProtoSimple<1, 1>;

  int tn = nWarps1*WARP_SIZE;
  if (tid < tn) {
    // Phase 1: send to network
  }
  tid -= tn;

  tn = nWarps2*WARP_SIZE;
  if (tid < tn) {
    // Phase 2: Recv network -> deposit output + send to bcast
  }
  tid -= tn;

  tn = nWarps3*WARP_SIZE;
  if (tid < tn) {
    // Phase 3: Recv bcast -> deposit output
  }
}
```

**功能**：CollNet Direct 算法的主运行函数。

**实现原理**：
- **三阶段处理**：分为发送、接收-广播、接收三个阶段
- **动态线程分配**：根据多轨道情况动态分配线程
- **网络优化**：支持网络注册和卸载优化

## All-Gather 算法特点

### 1. 环形算法
- **数据流动**：数据在环中依次传递
- **渐进构建**：逐步构建完整数据集合
- **负载均衡**：各节点负担相对均衡

### 2. PAT 算法
- **模式化通信**：基于预定义模式的通信
- **并行执行**：多个工作线程并行处理
- **动态调度**：运行时生成通信计划

### 3. NVLS 算法
- **虚拟链路**：利用 NVIDIA 虚拟链路服务
- **多轨传输**：支持多条并行数据轨道
- **层次化处理**：节点内和节点间分层处理

### 4. CollNet Direct 算法
- **直接连接**：绕过传统网络栈
- **硬件加速**：利用专用硬件加速
- **多阶段流水**：三阶段流水线处理

## 性能优化特性

### 1. 协议适应性
- **多协议支持**：支持 Simple、LL、LL128 等协议
- **动态选择**：根据场景选择最优协议

### 2. 内存优化
- **直接内存访问**：支持 GPU 直接内存访问
- **注册内存**：利用注册内存优化传输

### 3. 线程管理
- **动态分配**：根据任务需求动态分配线程
- **分组处理**：将线程分组执行不同任务

### 4. 网络优化
- **网络卸载**：支持网络卸载优化
- **多轨并行**：利用多条网络轨道并行传输

## 总结

`all_gather.h` 文件实现了 NCCL All-Gather 集合通信操作的多种算法和协议变体，提供了：

1. **多种拓扑支持**：Ring、PAT、NVLS、CollNet Direct
2. **多协议实现**：Simple、LL、LL128 协议
3. **优化策略**：网络卸载、直接内存访问、注册内存
4. **灵活配置**：动态线程分配、自适应参数
5. **高性能**：流水线处理、并行执行、硬件加速

该文件是 NCCL All-Gather 操作的核心实现，通过多种算法和优化技术实现了高效的集合通信。