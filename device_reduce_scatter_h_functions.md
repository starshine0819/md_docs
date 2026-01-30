# NCCL device/reduce_scatter_h_functions.md - Reduce-Scatter Implementation

## 文件概述

`reduce_scatter.h` 是 NCCL 设备端代码的 Reduce-Scatter 集合通信操作实现文件，位于 `/root/nccl/src/device/` 目录下。该文件实现了多种算法和协议的 Reduce-Scatter 操作，包括 Ring、PAT、NVLS 和 CollNet Direct 等不同拓扑结构。

## 核心函数详细分析

### 1. runRing 函数模板

```cpp
template<typename T, typename RedOp, typename Proto>
__device__ __forceinline__ void runRing(int tid, int nthreads, struct ncclDevWorkColl* work)
```

**功能**：实现基于环形拓扑的 Reduce-Scatter 操作。

**实现原理**：

#### 1.1 初始化阶段

```cpp
ncclRing *ring = &ncclShmem.channel.ring;
int const *ringRanks = ring->userRanks;
const int nranks = ncclShmem.comm.nRanks;
size_t count;
size_t gridOffset;
size_t channelCount;
size_t chunkCount;
ncclCollCbdPart(work, ncclShmem.channelId, Proto::Id, sizeof(T), &count, &gridOffset, &channelCount, &chunkCount);
```

**实现原理**：
- 获取环形拓扑信息和排名列表
- 获取参与进程数
- 计算计数、网格偏移、通道计数和块计数

#### 1.2 原语对象创建

```cpp
Primitives<T, RedOp, FanSymmetric<1>, 0, Proto, 0>
  prims(tid, nthreads, &ring->prev, &ring->next, work->sendbuff, work->recvbuff, work->redOpArg);
```

**实现原理**：
- 创建对称扇形原语对象
- 设置前驱和后继节点
- 传递发送和接收缓冲区及归约操作参数

#### 1.3 三阶段环形算法

##### 阶段1：推送数据到下一GPU

```cpp
for (size_t elemOffset = 0; elemOffset < channelCount; elemOffset += chunkCount) {
  nelem = min(chunkCount, channelCount - elemOffset);

  dataOffset = gridOffset + elemOffset;
  /////////////// begin ReduceScatter steps ///////////////
  // step 0: push data to next GPU
  rankDest = ringRanks[nranks-1];
  offset = dataOffset + rankDest * count;
  prims.send(offset, nelem);
```

**实现原理**：
- 将属于下一进程的数据块发送出去
- 计算目标进程和偏移量
- 使用 `send` 操作发送数据

##### 阶段2：归约并传送到下一GPU

```cpp
// k-2 steps: reduce and copy to next GPU
for (int j=2; j<nranks; ++j) {
  rankDest = ringRanks[nranks-j];
  offset = dataOffset + rankDest * count;
  prims.recvReduceSend(offset, nelem);
}
```

**实现原理**：
- 接收来自前一GPU的数据
- 与本地对应数据块进行归约
- 将归约结果发送到下一GPU

##### 阶段3：产生最终结果

```cpp
// step k-1: reduce this buffer and data, which will produce the final result
rankDest = ringRanks[0];
offset = dataOffset + rankDest * count;
prims.recvReduceCopy(offset, dataOffset, nelem, /*postOp=*/true);
```

**实现原理**：
- 接收最终的归约结果
- 将结果复制到本地接收缓冲区
- 应用后操作（如除法等）

### 2. PAT (Pattern-based) 算法实现

#### 2.1 PAT 算法结构

```cpp
template<typename T, typename RedOp>
struct RunWorkColl<ncclFuncReduceScatter, T, RedOp, NCCL_ALGO_PAT, NCCL_PROTO_SIMPLE> {
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
```

**实现原理**：
- **初始化共享内存**：初始化 PAT 步骤标志
- **同步**：确保所有线程到达
- **算法计算线程**：生成 PAT 操作序列
- **工作线程**：执行 PAT 归约操作

#### 2.2 算法计算线程

```cpp
if (tid == nworkers) { // Algo computation thread
  PatRSAlgorithm<T> patAlgo(chunkCount*sizeof(T), NCCL_STEPS, NCCL_PAT_NWORKERS/WARP_SIZE, channelOffset, channelOffset + channelCount, count, chunkCount, rank, nranks);
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
}
```

**实现原理**：
- **算法初始化**：创建 PAT 归约算法对象
- **并行因子**：计算并行执行因子
- **操作生成**：循环生成 PAT 步骤操作
- **同步机制**：使用原子操作进行同步

#### 2.3 工作线程

```cpp
else if (tid < nworkers) { // Worker threads
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
    (tidInGroup, groupSize, (int*)shmem->recvDims, (int*)shmem->sendDims, inputBuf, outputBuf, work->redOpArg, group, 0, 0, nullptr, nullptr, 0, primsModePatRs);

  int step = group;
  while(1) {
    struct ncclPatStep* ps = shmem->patSteps+(step%NCCL_SHMEM_PAT_STEPS);
    cuda::atomic_ref<int, cuda::thread_scope_block> poll(ps->flags);
    while (poll.load(cuda::memory_order_acquire) == 0) pollCount++; // Wait for compute thread
    int last = ps->last;
    prims.patReduce(ps, shmem);
    if (tidInGroup == 0) poll.store(0, cuda::memory_order_release); // Return element to compute thread
    if (last) break;
    step += nGroups;
  }
}
```

**实现原理**：
- **并行因子等待**：等待并行因子设置完成
- **线程分组**：根据并行因子分组工作线程
- **PAT 原语**：使用 PAT 模式归约原语
- **操作执行**：循环执行 PAT 归约操作

### 3. NVLS 算法实现

#### 3.1 Scatterer 结构体

```cpp
template<bool ReduceSendNotRecv>
struct Scatterer {
  struct ncclDevWorkColl* work;
  int chunkCount;
  ssize_t railGridOffset;

  template<int SlicePerChunk, int MinSrcs, int MaxSrcs, int MinDsts, int MaxDsts, int MultimemSrcs, int MultimemDsts>
  __device__ __forceinline__ void operator()(
      int tid, int tn, int slice, int maxSliceSize,
      int nSrcs, void** srcPtrs, int nDsts, void** dstPtrs, int32_t* dstSizes, uint32_t sendDirectFlag, uint32_t recvDirectFlag
    ) {
    static_assert(SlicePerChunk == 1, "require: SlicePerChunk==1");
    static_assert(MaxDsts <= 1 || MaxSrcs <= 1, "require: MaxDsts<=1 || MaxSrcs<=1");

    struct ncclNvls* nvls = &ncclShmem.channel.nvls;
    int nNodes = ncclShmem.comm.nNodes;
    int nRails = nvls->nHeads;
    int part = ncclShmem.channelId - work->channelLo;
    void* inbuf = (void*)work->sendbuff;
    ssize_t countPerRank = work->collnet.count;

    ssize_t railAllBeg = min(railGridOffset + part * chunkCount, nNodes * countPerRank);
    ssize_t railAllEnd = min(railAllBeg + chunkCount, nNodes * countPerRank);
    int railAllSize = railAllEnd - railAllBeg;
    int rail = nvls->headRank;
    int dst = 0;
    if (ReduceSendNotRecv) {
      if (work->regUsed) return;
      rail = 0;
      nSrcs = 1;
    } else {
      rail = nvls->headRank;
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
        if (nDsts != 0) {
          reduceCopy<ncclCollUnroll(), RedOp, T,
            /*MultimemSrcs=*/MultimemSrcs, 1, 1 + MaxSrcs,
            /*MultimemDsts,MinDsts,MaxDsts=*/MultimemDsts, 1, 1,
            /*PreOpSrcs=*/1>
            (tid, tn, work->redOpArg, false,
              /*nSrcs=*/nSrcs, [=]__device__(int s) {
            return work->regUsed ? (T*)srcPtrs[s] + userOneBeg :
              !ReduceSendNotRecv ? (T*)srcPtrs[s] + railAllOffset:
              (T*)inbuf + userOneBeg;
          },
              /*nDsts=*/1, [=]__device__(int d/*==0*/) {
            return (T*)dstPtrs[dst] + railAllOffset;
          }, delta);
        }
        railAllOffset += delta;
        node += 1;
      }
      dst += 1;
      rail += 1;
    } while (ReduceSendNotRecv && dst < nRails);
  }
};
```

**功能**：NVLS 算法的散射器，处理数据在不同轨道间的分布和归约。

**实现原理**：
- **参数验证**：使用 `static_assert` 验证模板参数
- **轨道计算**：计算轨道起始和结束位置
- **数据分段**：按节点和轨道分段处理数据
- **归约复制**：使用 `reduceCopy` 执行归约和复制操作

#### 3.2 NVLS 主运行函数

```cpp
__device__ __forceinline__ void run(int tid, int/*nthreads*/, struct ncclDevWorkColl* work) {
  struct ncclNvls* nvls = &ncclShmem.channel.nvls;
  int nelem;

  /* if we are direct NVLS, we only need to allocate 1 warp to scatter for sync;
   * if not, based on #ranks, we allocate 7 or 5 warps to reduce to saturate bandwidth
   * and the rest are allocated to scatter. */
  const int nThreadsNetRecv = work->oneNode ? 0 : (work->netRegUsed ? WARP_SIZE :  6 * WARP_SIZE);
  const int nThreadsScatter = work->regUsed ? roundUp(nvls->nHeads << 2, WARP_SIZE) : 8 * WARP_SIZE;
  const int nThreadsReduce = NCCL_MAX_NTHREADS - nThreadsNetRecv - nThreadsScatter;
  const int tidEndNetRecv = nThreadsNetRecv;
  const int tidEndScatter = tidEndNetRecv + nThreadsScatter;
  const int tidEndReduce = tidEndScatter + nThreadsReduce;

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
- **单节点模式**：优化单节点内的 Reduce-Scatter
- **多节点模式**：处理跨节点的 Reduce-Scatter

### 4. CollNet Direct 算法实现

#### 4.1 CollNet Scatterer 结构体

```cpp
template<bool ReduceSendNotRecv>
struct Scatterer {
  struct ncclDevWorkColl* work;
  int chunkSize;
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
    void* inbuf = (void*)work->sendbuff;
    ssize_t countPerRank = work->collnet.count;

    ssize_t railAllBeg = min(railGridOffset + part*chunkSize, nNodes*countPerRank);
    ssize_t railAllEnd = min(railAllBeg + chunkSize, nNodes*countPerRank);
    int railAllSize = railAllEnd - railAllBeg;
    if (tid < nDsts) dstSizes[tid] = railAllSize;

    int dst = 0;
    int rail;
    if (!ReduceSendNotRecv) {
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
        if (nDsts != 0) {
          reduceCopy<ncclCollUnroll(), RedOp, T,
                   /*MultimemSrcs=*/0, 1+MinSrcs, 1+MaxSrcs,
                   /*MultimemDsts,MinDsts,MaxDsts=*/0,1,1,
                   /*PreOpSrcs=*/1>
          (tid, tn, work->redOpArg, false,
           /*nSrcs=*/1+nSrcs, [=]__device__(int s) {
             return s==0 ? (T*)inbuf + userOneBeg
                         : work->regUsed && (recvDirectFlag & NCCL_P2P_READ)
                         ? (T*)srcPtrs[s-1] + userOneBeg
                         : (T*)srcPtrs[s-1] + railAllOffset;
           },
           /*nDsts=*/1, [=]__device__(int d/*==0*/) {
             return (T*)dstPtrs[dst] + railAllOffset;
           },
           delta);
        }
        railAllOffset += delta;
        node += 1;
      }
      dst += 1;
      rail += 1;
      if (rail == nRails) rail = 0;
    } while (ReduceSendNotRecv && dst < nRails-1);
  }
};
```

**功能**：CollNet Direct 算法的散射器，处理直接连接网络的数据传输。

**实现原理**：
- **轨道计算**：计算当前轨道的数据范围
- **节点遍历**：按节点遍历处理数据
- **动态轨道选择**：根据操作类型选择轨道
- **条件归约**：根据注册使用情况选择不同的内存访问模式

#### 4.2 CollNet 主运行函数

```cpp
__device__ __forceinline__ void run(int tid, int nthreads, struct ncclDevWorkColl* work) {
  const int part = ncclShmem.channelId - work->channelLo;
  const int nChannels = work->channelHi - work->channelLo + 1;
  struct ncclDirect* direct = &ncclShmem.channel.collnetDirect;
  int const &nNodes = ncclShmem.comm.nNodes;
  ssize_t chunkSize = int(work->collnet.chunkCount);
  ssize_t countPerRank = work->collnet.count;
  const int hasDn = (direct->down[0] >= 0) ? 1 : 0;

  if (direct->out == -1) __trap();
  bool isMultiRail = (direct->nHeads > 1);
  int nWarps1 = (isMultiRail ? 2 : 0);
  int nWarps2 = (isMultiRail ? 2 : 1);
  int nWarps3 = 1;
  float denom = float(work->nWarps)/float(nWarps1+nWarps2+nWarps3);
  nWarps3 = int(denom*nWarps3);
  nWarps2 = int(denom*nWarps2);
  nWarps1 = work->nWarps - (nWarps2+nWarps3);

  using Proto = ProtoSimple<1, 1>;

  int tn = nWarps1*WARP_SIZE;
  if (tid < tn) {
    // Phase 1: Scatter inputs to peers
  }
  tid -= tn;

  tn = nWarps2*WARP_SIZE;
  if (tid < tn) {
    // Phase 2: Reduce from peers + local input -> send to network
  }
  tid -= tn;

  tn = nWarps3*WARP_SIZE;
  if (tid < tn) {
    // Phase 3: recv from network
  }
}
```

**功能**：CollNet Direct 算法的主运行函数。

**实现原理**：
- **三阶段处理**：分为分散、归约-发送、接收三个阶段
- **动态线程分配**：根据多轨道情况动态分配线程
- **网络优化**：支持网络注册和卸载优化

### 5. 算法特化实现

#### 5.1 Ring + Simple 协议

```cpp
template<typename T, typename RedOp>
struct RunWorkColl<ncclFuncReduceScatter, T, RedOp, NCCL_ALGO_RING, NCCL_PROTO_SIMPLE> {
  __device__ __forceinline__ void run(int tid, int nthreads, struct ncclDevWorkColl* work) {
    using Proto = ProtoSimple<REDUCESCATTER_CHUNKSTEPS/REDUCESCATTER_SLICESTEPS, REDUCESCATTER_SLICESTEPS>;
    runRing<T, RedOp, Proto>(tid, nthreads, work);
  }
};
```

#### 5.2 Ring + LL 协议

```cpp
template<typename T, typename RedOp>
struct RunWorkColl<ncclFuncReduceScatter, T, RedOp, NCCL_ALGO_RING, NCCL_PROTO_LL> {
  __device__ __forceinline__ void run(int tid, int nthreads, struct ncclDevWorkColl* work) {
    runRing<T, RedOp, ProtoLL>(tid, nthreads, work);
  }
};
```

#### 5.3 Ring + LL128 协议

```cpp
template<typename T, typename RedOp>
struct RunWorkColl<ncclFuncReduceScatter, T, RedOp, NCCL_ALGO_RING, NCCL_PROTO_LL128> {
  __device__ __forceinline__ void run(int tid, int nthreads, struct ncclDevWorkColl* work) {
    runRing<T, RedOp, ProtoLL128>(tid, nthreads, work);
  }
};
```

## Reduce-Scatter 算法特点

### 1. 环形算法
- **数据流动**：数据在环中依次传递和归约
- **渐进构建**：逐步构建归约结果
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
- **多阶段流水**：分散-归约-接收流水线处理

## 性能优化特性

### 1. 协议适应性
- **多协议支持**：Simple、LL、LL128 协议
- **动态选择**：根据场景选择最优协议

### 2. 内存优化
- **直接内存访问**：支持 GPU 直接内存访问
- **注册内存**：利用注册内存优化传输
- **分块处理**：将大数据分成小块处理

### 3. 线程管理
- **动态分配**：根据任务需求动态分配线程
- **分组处理**：将线程分组执行不同任务

### 4. 网络优化
- **网络卸载**：支持网络卸载优化
- **多轨并行**：利用多条网络轨道并行传输

## 错误处理和可靠性

### 1. 进度跟踪
- **精确跟踪**：跟踪每个阶段的进度
- **完成确认**：确保所有操作完成

### 2. 边界检查
- **零大小处理**：正确处理零大小数据
- **缓冲区边界**：确保不超出缓冲区边界

### 3. 同步保证
- **线程一致**：确保所有线程状态一致
- **内存可见**：确保内存修改对所有线程可见

## 应用场景

### 1. 梯度聚合
- 在分布式训练中将各进程的梯度归约并分散
- 每个进程获得一部分归约后的梯度

### 2. 数据分析
- 将分布式数据进行聚合分析
- 每个节点获得分析结果的一部分

### 3. 模型同步
- 在分布式训练中同步模型参数
- 每个进程获得模型参数的不同部分

## 总结

`reduce_scatter.h` 文件实现了 NCCL Reduce-Scatter 集合通信操作的多种算法和协议变体，提供了：

1. **多种拓扑支持**：Ring、PAT、NVLS、CollNet Direct
2. **多协议实现**：Simple、LL、LL128 协议
3. **优化策略**：网络卸载、直接内存访问、注册内存
4. **灵活配置**：动态线程分配、自适应参数
5. **高性能**：流水线处理、并行执行、硬件加速

该文件是 NCCL Reduce-Scatter 操作的核心实现，通过多种算法和优化技术实现了高效的归约-分散通信。