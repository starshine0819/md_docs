# NCCL device/all_reduce_h_functions.md - All-Reduce Implementation

## 文件概述

`all_reduce.h` 是 NCCL 设备端代码的 All-Reduce 集合通信操作实现文件，位于 `/root/nccl/src/device/` 目录下。该文件实现了多种算法和协议的 All-Reduce 操作，包括 Ring、Tree、CollNet Direct、NVLS 等不同拓扑结构。

## 核心函数详细分析

### 1. runRing 函数模板

```cpp
template<typename T, typename RedOp, typename Proto>
__device__ __forceinline__ void runRing(int tid, int nthreads, struct ncclDevWorkColl* work)
```

**功能**：实现基于环形拓扑的 All-Reduce 操作。

**实现原理**：

#### 1.1 初始化阶段

```cpp
ncclRing *ring = &ncclShmem.channel.ring;
int ringIx = ring->index;
const int nranks = ncclShmem.comm.nRanks;
ssize_t gridOffset;
ssize_t channelCount;
ssize_t chunkCount;
ncclCollCbdPart(work, ncclShmem.channelId, Proto::Id, sizeof(T), (ssize_t*)nullptr, &gridOffset, &channelCount, &chunkCount);
const ssize_t loopCount = nranks * chunkCount;
```

- 获取环形拓扑信息
- 获取参与进程数
- 计算网格偏移、通道计数和块计数
- 计算循环计数（总数据量）

#### 1.2 原语对象创建

```cpp
Primitives<T, RedOp, FanSymmetric<1>, 1, Proto, 0> prims
  (tid, nthreads, &ring->prev, &ring->next, work->sendbuff, work->recvbuff, work->redOpArg, 0, 0, 0, work);
```

- 创建原语对象，使用对称扇形结构
- 设置前驱和后继节点
- 设置输入输出缓冲区

#### 1.3 主循环处理

```cpp
for (ssize_t elemOffset = 0; elemOffset < channelCount; elemOffset += loopCount) {
  ssize_t remCount = channelCount - elemOffset;
  ssize_t chunkOffset;

  if (remCount < loopCount) chunkCount = alignUp(divUp(remCount, nranks), 16/sizeof(T));
```

- 按循环大小处理数据
- 对于剩余数据，调整块大小

#### 1.4 环形算法四阶段

##### 阶段1：推送数据到下一GPU

```cpp
auto modRanks = [&]__device__(int r)->int {
  return r - (r >= nranks ? nranks : 0);
};

// step 0: push data to next GPU
chunk = modRanks(ringIx + nranks - 1);
chunkOffset = chunk * chunkCount;
offset = gridOffset + elemOffset + chunkOffset;
nelem = (int)min(chunkCount, remCount - chunkOffset);
prims.directSend(offset, offset, nelem);
```

- 计算数据块索引
- 发送自己的数据块到下一GPU

##### 阶段2：归约并传送到下一GPU

```cpp
// k-2 steps: reduce and copy to next GPU
for (int j = 2; j < nranks; ++j) {
  chunk = modRanks(ringIx + nranks - j);
  chunkOffset = chunk * chunkCount;
  offset = gridOffset + elemOffset + chunkOffset;
  nelem = (int)min(chunkCount, remCount - chunkOffset);
  prims.directRecvReduceDirectSend(offset, offset, nelem);
}
```

- 接收来自前一GPU的数据
- 与本地数据进行归约
- 将结果发送到下一GPU

##### 阶段3：产生最终结果

```cpp
// step k-1: reduce this buffer and data, which will produce the final
// result that we store in this data and push to the next GPU
chunk = ringIx + 0;
chunkOffset = chunk * chunkCount;
offset = gridOffset + elemOffset + chunkOffset;
nelem = (int)min(chunkCount, remCount - chunkOffset);
prims.directRecvReduceCopyDirectSend(offset, offset, nelem, /*postOp=*/true);
```

- 接收数据并进行最终归约
- 应用后操作（如除法）
- 将完整结果发送到下一GPU

##### 阶段4：传播结果

```cpp
// k-2 steps: copy to next GPU
for (int j = 1; j < nranks - 1; ++j) {
  chunk = modRanks(ringIx + nranks - j);
  chunkOffset = chunk * chunkCount;
  offset = gridOffset + elemOffset + chunkOffset;
  nelem = (int)min(chunkCount, remCount - chunkOffset);
  prims.directRecvCopyDirectSend(offset, offset, nelem);
}

// Make final copy from buffer to dest.
chunk = modRanks(ringIx + 1);
chunkOffset = chunk * chunkCount;
offset = gridOffset + elemOffset + chunkOffset;
nelem = (int)min(chunkCount, remCount - chunkOffset);
prims.directRecv(offset, nelem);
```

- 继续传播归约结果
- 最终接收结果到目标缓冲区

### 2. runTreeUpDown 函数模板

```cpp
template<typename T, typename RedOp, typename Proto>
__device__ __forceinline__ void runTreeUpDown(int tid, int nthreads, struct ncclDevWorkColl* work)
```

**功能**：实现树形拓扑的向上归约和向下广播算法。

**实现原理**：

#### 2.1 归约阶段

```cpp
{ // Reduce : max number of recv is 3, max number of send is 1 (binary tree + local)
  Primitives<T, RedOp, FanAsymmetric<NCCL_MAX_TREE_ARITY, 1>, /*Direct=*/1, Proto, 0> prims
    (tid, nthreads, tree->down, &tree->up, work->sendbuff, work->recvbuff, work->redOpArg, 0, 0, 0, work);
  
  if (tree->up == -1) {
    // 根节点：接收子节点数据，归约并复制到输出
    for (size_t elemOffset = 0; elemOffset < channelCount; elemOffset += chunkCount) {
      offset = gridOffset + elemOffset;
      nelem = min(chunkCount, channelCount - elemOffset);
      prims.directRecvReduceCopy(offset, offset, nelem, /*postOp=*/true);
    }
  }
  else if (tree->down[0] == -1) {
    // 叶子节点：发送数据到父节点
    for (size_t elemOffset = 0; elemOffset < channelCount; elemOffset += chunkCount) {
      offset = gridOffset + elemOffset;
      nelem = min(chunkCount, channelCount - elemOffset);
      prims.directSend(offset, offset, nelem);
    }
  }
  else {
    // 中间节点：接收子节点数据，归约并发送到父节点
    for (size_t elemOffset = 0; elemOffset < channelCount; elemOffset += chunkCount) {
      offset = gridOffset + elemOffset;
      nelem = min(chunkCount, channelCount - elemOffset);
      prims.directRecvReduceDirectSend(offset, offset, nelem);
    }
  }
}
```

#### 2.2 广播阶段

```cpp
{ // Broadcast : max number of recv is 1, max number of send is 3 (binary tree + local)
  Primitives<T, RedOp, FanAsymmetric<1, NCCL_MAX_TREE_ARITY>, /*Direct=*/1, Proto, 0> prims
    (tid, nthreads, &tree->up, tree->down, work->sendbuff, work->recvbuff, work->redOpArg, 0, 0, 0, work);
  
  if (tree->up == -1) {
    // 根节点：将归约结果发送给子节点
    for (size_t elemOffset = 0; elemOffset < channelCount; elemOffset += chunkCount) {
      offset = gridOffset + elemOffset;
      nelem = min(chunkCount, channelCount - elemOffset);
      prims.directSendFromOutput(offset, nelem);
    }
  }
  else if (tree->down[0] == -1) {
    // 叶子节点：从父节点接收广播数据
    for (size_t elemOffset = 0; elemOffset < channelCount; elemOffset += chunkCount) {
      offset = gridOffset + elemOffset;
      nelem = min(chunkCount, channelCount - elemOffset);
      prims.directRecv(offset, nelem);
    }
  }
  else {
    // 中间节点：从父节点接收并转发给子节点
    for (size_t elemOffset = 0; elemOffset < channelCount; elemOffset += chunkCount) {
      offset = gridOffset + elemOffset;
      nelem = min(chunkCount, channelCount - elemOffset);
      prims.directRecvCopyDirectSend(offset, offset, nelem);
    }
  }
}
```

### 3. runTreeSplit 函数模板

```cpp
template<typename T, typename RedOp, typename Proto>
__device__ __forceinline__ void runTreeSplit(int tid, int nthreads, struct ncclDevWorkColl* work)
```

**功能**：实现树形拓扑的分裂算法，分别处理归约和广播阶段。

**实现原理**：

#### 3.1 线程分割

```cpp
int nthreadsSplit;
if (Proto::Id == NCCL_PROTO_SIMPLE) {
  nthreadsSplit = nthreads/2;
  if (nthreadsSplit >= 256) nthreadsSplit += 64;
} else { // LL & LL128
  // Receiving from up to 3 sources is more compute intensive than sending
  // to 3 dests. Use 70% for reduce and 30% for bcast.
  nthreadsSplit = (nthreads*7/(10*WARP_SIZE))*WARP_SIZE;
}
```

- 根据协议类型分配线程
- LL/LL128 协议由于接收更复杂，分配更多线程给归约阶段

#### 3.2 根节点处理

```cpp
if (tree->up == -1) {
  // Reduce and broadcast. Max number of recv is 2, max number of send is 2
  Primitives<T, RedOp, FanSymmetric<NCCL_MAX_TREE_ARITY_TOP>, /*Direct=*/1, Proto, 0>
    prims(tid, nthreads, tree->down, tree->down, work->sendbuff, work->recvbuff, work->redOpArg, 0, 0, 0, work);
  for (size_t elemOffset = 0; elemOffset < channelCount; elemOffset += chunkCount) {
    offset = gridOffset + elemOffset;
    nelem = min(chunkCount, channelCount - elemOffset);
    prims.directRecvReduceCopyDirectSend(offset, offset, nelem, /*doPost=*/true);
  }
}
```

#### 3.3 归约阶段（下半部分线程）

```cpp
else if (tid < nthreadsSplit) {
  Primitives<T, RedOp, FanAsymmetric<NCCL_MAX_TREE_ARITY, 1>, /*Direct=*/1, Proto, 0>
    prims(tid, nthreadsSplit, tree->down, &tree->up, work->sendbuff, work->recvbuff, work->redOpArg, 0*Proto::MaxGroupWidth, 0, 0, work);
  
  if (tree->down[0] == -1) {
    // 叶子节点：发送数据
    for (size_t elemOffset = 0; elemOffset < channelCount; elemOffset += chunkCount) {
      offset = gridOffset + elemOffset;
      nelem = min(chunkCount, channelCount - elemOffset);
      prims.directSend(offset, offset, nelem);
    }
  }
  else {
    // 中间节点：接收归约发送
    for (size_t elemOffset = 0; elemOffset < channelCount; elemOffset += chunkCount) {
      offset = gridOffset + elemOffset;
      nelem = min(chunkCount, channelCount - elemOffset);
      prims.directRecvReduceDirectSend(offset, offset, nelem);
    }
  }
}
```

#### 3.4 广播阶段（上半部分线程）

```cpp
else {
  // Broadcast down. Max number of recv is 1, max number of send is 3
  Primitives<T, RedOp, FanAsymmetric<1, NCCL_MAX_TREE_ARITY>, /*Direct=*/1, Proto, 0>
    prims(tid-nthreadsSplit, nthreads-nthreadsSplit, &tree->up, tree->down, work->sendbuff, work->recvbuff,
        work->redOpArg, 1*Proto::MaxGroupWidth, 0, 0, work);
  
  if (tree->down[0] == -1) {
    // 叶子节点：接收数据
    for (size_t elemOffset = 0; elemOffset < channelCount; elemOffset += chunkCount) {
      offset = gridOffset + elemOffset;
      nelem = min(chunkCount, channelCount - elemOffset);
      prims.directRecv(offset, nelem);
    }
  }
  else {
    // 中间节点：接收复制发送
    for (size_t elemOffset = 0; elemOffset < channelCount; elemOffset += chunkCount) {
      offset = gridOffset + elemOffset;
      nelem = min(chunkCount, channelCount - elemOffset);
      prims.directRecvCopyDirectSend(offset, offset, nelem);
    }
  }
}
```

### 4. CollNet Direct 算法实现

#### 4.1 线程分配

```cpp
const int hasUp = (direct->up[0] >= 0) ? 1 : 0;
const int hasDn = (direct->down[0] >= 0) ? 1 : 0;
const int nThreadsScatter = WARP_SIZE + ((hasUp && hasDn) ? COLLNET_COPY_THREADS : hasUp ? 3*COLLNET_COPY_THREADS : 0);
const int nThreadsGather  =             ((hasUp && hasDn) ? COLLNET_COPY_THREADS : hasUp ? 2*COLLNET_COPY_THREADS : 0);
const int nThreadsBcast   = WARP_SIZE + ((hasUp && hasDn) ? COLLNET_COPY_THREADS : hasUp ? 0 : 2*COLLNET_COPY_THREADS);
const int nThreadsReduce = work->nWarps*WARP_SIZE - nThreadsScatter - nThreadsGather - nThreadsBcast;
```

- 根据连接情况动态分配线程
- 散布、收集、归约、广播阶段各有专门的线程

#### 4.2 散布阶段

```cpp
if (tid >= tidStartScatter && tid < tidStartReduce && hasUp) {
  // Scatter
  Primitives<T, RedOp, FanAsymmetric<0, NCCL_MAX_DIRECT_ARITY>, /*Direct=*/0, Proto, 0>
    prims(tid-tidStartScatter, nThreadsScatter, NULL, direct->up, work->sendbuff, work->recvbuff,
       work->redOpArg, 2*Proto::MaxGroupWidth, 1, 1, work);
  // ...
  for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
    ssize_t offset = gridOffset + offsetBase;
    ssize_t nelem = min(maxNelems, size - offset);
    prims.scatter(offset, nelem, chunkSize, peerOffset, direct->headRank, direct->shift);
  }
}
```

#### 4.3 归约阶段

```cpp
else if (tid >= tidStartReduce && direct->out != -1) {
  if (hasDn) {
    // Reduce, send to network
    Primitives<T, RedOp, FanAsymmetric<NCCL_MAX_DIRECT_ARITY, 1>, /*Direct=*/0, Proto, 0>
      prims(tid-tidStartReduce, nThreadsReduce, direct->down, &direct->out, work->sendbuff, work->recvbuff,
         work->redOpArg, 3*Proto::MaxGroupWidth, 1, 1, work);
    for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
      ssize_t offset = work->netRegUsed ? gridOffset + (bid + direct->headRank * nChannels) * chunkSize
                                : gridOffset + (bid * direct->nHeads + direct->headRank) * chunkSize;
      int nelem = min(chunkSize, size - offset);
      prims.recvReduceDirectSend(offset, offset, nelem);
    }
  }
}
```

#### 4.4 广播阶段

```cpp
else if (tid >= tidStartBcast && tid < tidStartScatter && direct->out != -1) {
  if (hasDn) {
    // Recv from network, broadcast
    Primitives<T, RedOp, FanAsymmetric<1, NCCL_MAX_DIRECT_ARITY>, /*Direct=*/0, Proto, 0>
      prims(tid-tidStartBcast, nThreadsBcast, &direct->out, direct->down, work->sendbuff, work->recvbuff,
         work->redOpArg, 1*Proto::MaxGroupWidth, 0, 0, work);
    for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
      ssize_t offset = work->netRegUsed ? gridOffset + (bid + direct->headRank * nChannels) * chunkSize
                                        : gridOffset + (bid * direct->nHeads + direct->headRank) * chunkSize;
      int nelem = min(chunkSize, size - offset);
      prims.directRecvCopyDirectSend(offset, offset, nelem, /*postOp=*/true);
    }
  }
}
```

### 5. NVLS 算法实现

#### 5.1 线程分配

```cpp
const int totalWarps = NCCL_MAX_NTHREADS/WARP_SIZE;
const int bcastWarps = hasOut ? (work->regUsed ? ((totalWarps - 2) >> 1) - 1 : 2) : 0;
const int reduceWarps = work->regUsed ? (totalWarps - bcastWarps - 2) : (hasOut ? 3 : nranks <= 6 ? 7 : 5);
const int scatterWarps = work->regUsed ? 1 : (totalWarps - reduceWarps - bcastWarps + 1) >> 1;
const int gatherWarps = work->regUsed ? 1 : (totalWarps - reduceWarps - bcastWarps) >> 1;
```

- 根据注册内存使用情况动态分配线程
- 单节点和多节点模式有不同的分配策略

#### 5.2 四阶段处理

- **散布阶段**：将数据分发到 NVLS 上行节点
- **收集阶段**：从 NVLS 上行节点收集数据
- **归约阶段**：在 NVLS 内部进行归约
- **广播阶段**：将归约结果广播到下行节点

### 6. 算法特化实现

#### 6.1 Ring 算法特化

```cpp
template<typename T, typename RedOp>
struct RunWorkColl<ncclFuncAllReduce, T, RedOp, NCCL_ALGO_RING, NCCL_PROTO_SIMPLE> {
  __device__ __forceinline__ void run(int tid, int nthreads, struct ncclDevWorkColl* work) {
    using Proto = ProtoSimple<ALLREDUCE_CHUNKSTEPS/ALLREDUCE_SLICESTEPS, ALLREDUCE_SLICESTEPS>;
    runRing<T, RedOp, Proto>(tid, nthreads, work);
  }
};
```

#### 6.2 Tree 算法特化

```cpp
template<typename T, typename RedOp>
struct RunWorkColl<ncclFuncAllReduce, T, RedOp, NCCL_ALGO_TREE, NCCL_PROTO_SIMPLE> {
  __device__ __forceinline__ void run(int tid, int nthreads, struct ncclDevWorkColl* work) {
    #if CUDART_VERSION >= 11020 && CUDART_VERSION < 11040 && __CUDA_ARCH__ >= 800
      runTreeUpDown<T, RedOp, ProtoSimple<1, 1>>(tid, nthreads, work);
    #else
      runTreeSplit<T, RedOp, ProtoSimple<1, 1>>(tid, nthreads, work);
    #endif
  }
};
```

## All-Reduce 算法特点

### 1. 环形算法
- **两阶段过程**：归约阶段和广播阶段
- **数据流控制**：精确控制数据在环中的流向
- **负载均衡**：各节点负担相对均衡

### 2. 树形算法
- **上下结构**：明确的归约和广播阶段
- **分层处理**：按树层级处理数据
- **线程优化**：根据节点类型优化线程分配

### 3. CollNet Direct 算法
- **直接网络**：绕过传统集合通信
- **多阶段流水**：散布-归约-收集-广播
- **硬件加速**：利用专用硬件优化

### 4. NVLS 算法
- **虚拟链接**：利用 NVIDIA 虚拟链接
- **层次化通信**：节点内和节点间分层
- **注册内存**：优化内存访问

## 性能优化特性

### 1. 协议适应性
- **多协议支持**：Simple、LL、LL128 协议
- **动态选择**：根据场景选择最优协议

### 2. 线程管理
- **动态分配**：根据任务需求分配线程
- **负载均衡**：平衡各阶段线程负载
- **分组处理**：线程分组执行不同任务

### 3. 内存优化
- **直接内存访问**：支持 GPU 直接内存访问
- **注册内存**：利用注册内存优化传输
- **缓存友好**：优化内存访问模式

### 4. 网络优化
- **多轨并行**：利用多条网络轨道
- **流量控制**：优化网络流量分布
- **卸载优化**：支持网络卸载

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

## 总结

`all_reduce.h` 文件实现了 NCCL All-Reduce 集合通信操作的多种算法和协议变体，提供了：

1. **多种拓扑支持**：Ring、Tree、CollNet Direct、NVLS
2. **多协议实现**：Simple、LL、LL128 协议
3. **优化策略**：动态线程分配、内存优化、网络优化
4. **灵活配置**：适应不同硬件环境和应用场景
5. **高性能**：流水线处理、并行执行、硬件加速

该文件是 NCCL All-Reduce 操作的核心实现，通过多种算法和优化技术实现了高效的归约-广播通信。