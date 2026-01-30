# NCCL device/common_h_header_functions.md - Common Header Functions

## 文件概述

`common.h` 是 NCCL 设备端代码的公共头文件，位于 `/root/nccl/src/device/` 目录下。该文件定义了设备端共享内存布局、同步原语、工作负载管理、内核入口点等核心组件，是 NCCL 设备端实现的基础架构。

## 核心数据结构

### 1. ncclShmemGroup 结构

```cpp
struct ncclShmemGroup {
  ncclConnInfo *recvConns[NCCL_MAX_ARITY];
  ncclConnInfo *sendConns[NCCL_MAX_ARITY];
  void* userInput;
  void* userOutput;
  void* srcs[NCCL_MAX_ARITY+1];
  void* dsts[NCCL_MAX_ARITY+1];
  union {
    unpackGroupShmem unpack;
  } devicePlugin;
  int32_t dstSizes[NCCL_MAX_ARITY+1];
  uint64_t redOpArgs;
};
```

**功能**：定义共享内存组结构，用于存储连接信息和数据源/目标。

**字段说明**：
- `recvConns/sendConns`：接收/发送连接信息数组
- `userInput/userOutput`：用户输入/输出缓冲区指针
- `srcs/dsts`：源/目标缓冲区指针数组
- `devicePlugin`：设备插件特定的共享内存结构
- `dstSizes`：目标缓冲区大小数组
- `redOpArgs`：归约操作参数

### 2. ncclShmemData 结构

```cpp
struct ncclShmemData {
  struct ncclDevKernelArgs args;
  int channelId;
  int aborted;
  alignas(16) struct ncclKernelComm comm;
  alignas(16) struct ncclDevChannel channel;

  int batchIx, nextBatchIx;
  enum ncclDevWorkType workType;
  uint8_t directMode;
  uint16_t funcId;
  int nWorks;
  int workSize;
  uint64_t workCounter;
  bool profilerEnabled;
  struct ncclShmemGroup groups[NCCL_MAX_GROUPS];

  alignas(16) char workStorage[ncclMaxDevWorkBatchBytes()];

  alignas(16) union {
    unpackShmem unpack;
  } devicePlugin;
};
```

**功能**：定义设备端共享内存的主要数据结构。

**字段说明**：
- `args`：内核参数
- `channelId`：通道ID
- `aborted`：中止标志
- `comm/channel`：通信和通道信息（16字节对齐）
- `batchIx/nextBatchIx`：当前/下一个批次索引
- `workType`：工作类型
- `funcId`：函数ID
- `nWorks/workSize`：工作项数量和大小
- `workCounter`：工作计数器
- `groups`：共享内存组数组
- `workStorage`：工作存储空间

### 3. 全局共享内存变量

```cpp
extern __shared__ ncclShmemData ncclShmem;
#if __CUDA_ARCH__ >= 700
  extern __shared__ ulong2 ncclShmemPerWarp[/*ncclShmemDynamicSize()/sizeof(ulong2)*/];
#else
  extern __shared__ ulong2 ncclShmemPerWarp[ncclShmemScratchWarpSize()*(NCCL_MAX_NTHREADS/WARP_SIZE)/sizeof(ulong2)];
#endif
```

**功能**：定义全局共享内存变量。

**实现原理**：
- `ncclShmem`：主要的共享内存数据结构
- `ncclShmemPerWarp`：每个warp的共享内存空间（CUDA 7.0+架构优化）

## 同步原语函数

### 1. 屏障同步函数

#### 1.1 barrier_sync 函数

```cpp
__device__ inline void barrier_sync(int name) {
  asm volatile("barrier.sync.aligned %0;" :: "r"(name) : "memory");
}

__device__ inline void barrier_sync(int name, int nThreads) {
  asm volatile("barrier.sync.aligned %0, %1;" :: "r"(name), "r"(nThreads) : "memory");
}
```

**功能**：线程块级别的同步屏障。

**实现原理**：
- 使用PTX汇编指令 `barrier.sync.aligned` 实现线程同步
- 支持指定参与线程数量的同步

#### 1.2 barrier_red_or 函数

```cpp
__device__ inline bool barrier_red_or(bool vote, int name) {
  int ans;
  asm volatile("{ .reg .pred p;"
      "  setp.ne.s32 p, %1, 0;"
      "  barrier.red.or.pred p, %2, p; "
      "  selp.s32 %0, 1, 0, p; }"
      : "=r"(ans) : "r"((int)vote), "r"(name) : "memory");
  return bool(ans);
}
```

**功能**：执行OR归约的同步操作。

**实现原理**：
- 使用PTX汇编实现归约操作
- 将布尔投票结果进行OR归约
- 返回归约结果

### 2. 工具函数

#### 2.1 ncclScratchForWarp 函数

```cpp
__device__ inline void* ncclScratchForWarp(int warp) {
  return (char*)ncclShmemPerWarp + warp*ncclShmemScratchWarpSize();
}
```

**功能**：获取指定warp的临时存储空间。

**实现原理**：
- 计算每个warp的存储偏移
- 返回该warp专用的存储空间指针

## 工作负载管理函数

### 1. loadWorkBatchToShmem 函数

```cpp
__device__ __forceinline__ void loadWorkBatchToShmem(
    int tid, int tn, struct ncclDevKernelArgs const* args, int batchIx
  ) {
  int lane = tid%WARP_SIZE;
  int workCursor = 0; // num works written in previous loop iterations.
  while (true) {
    struct ncclDevWorkBatch batch = ((struct ncclDevWorkBatch*)(args+1))[batchIx];

    // fnsOfBitset[n] = index of n'th set bit in batch.offsetBitset.
    uint8_t* fnsOfBitset = (uint8_t*)ncclScratchForWarp(threadIdx.x/WARP_SIZE);
    __syncwarp();
    if (uint32_t(batch.offsetBitset) & (1u<<lane)) {
      int nWorksBelow = __popc(uint32_t(batch.offsetBitset) & ((1u<<lane)-1));
      fnsOfBitset[nWorksBelow] = lane;
    }
    int nWorksLow32 = __popc(uint32_t(batch.offsetBitset)); // just of low 32 bits
    if (uint32_t(batch.offsetBitset>>32) & (1u<<lane)) {
      int nWorksBelow = nWorksLow32;
      nWorksBelow += __popc(uint32_t(batch.offsetBitset>>32) & ((1u<<lane)-1));
      fnsOfBitset[nWorksBelow] = 32 + lane;
    }
    int nWorks = nWorksLow32 + __popc(uint32_t(batch.offsetBitset>>32)); // add high 32 bits
    __syncwarp();
```

**功能**：将工作批次加载到共享内存中。

**实现原理**：

#### 1.1 位图解析
```cpp
// fnsOfBitset[n] = index of n'th set bit in batch.offsetBitset.
uint8_t* fnsOfBitset = (uint8_t*)ncclScratchForWarp(threadIdx.x/WARP_SIZE);
__syncwarp();
if (uint32_t(batch.offsetBitset) & (1u<<lane)) {
  int nWorksBelow = __popc(uint32_t(batch.offsetBitset) & ((1u<<lane)-1));
  fnsOfBitset[nWorksBelow] = lane;
}
```

- 解析工作项的位图，确定哪些工作项需要处理
- 使用warp级同步和人口计数指令

#### 1.2 工作项大小计算
```cpp
switch (batch.workType) {
case (int)ncclDevWorkTypeP2p:
  workSize = sizeof(struct ncclDevWorkP2p);
  nPacks = nWorks*(workSize/16);
  packInWork = tid%(workSize/16);
  dstWork = tid/(workSize/16);
  break;
// ... 其他工作类型
}
```

- 根据工作类型确定工作项大小
- 计算需要的打包数量和线程分配

#### 1.3 数据加载
```cpp
if (tid < nPacks) {
  int srcWork = fnsOfBitset[dstWork]; // find n'th set bit in batch.offsetBitset
  ulong2 tmp;
  if (ncclShmem.args.workStorageType == ncclDevWorkStorageTypeArgs) {
    char* src = (char*)args + (batch.offsetBase + srcWork*workSize + packInWork*16);
    tmp = *(ulong2*)src; // becomes ld.param.v2.u64
  } else {
    char* src = (char*)ncclShmem.args.workBuf + ((batch.offsetBase + srcWork*workSize + packInWork*16) & ncclShmem.args.workMask);
    tmp = *(ulong2*)src; // becomes ld.v2.u64
  }
  char* dst = ncclShmem.workStorage;
  dst += (workCursor + dstWork)*workSize + packInWork*16;
  *(ulong2*)dst = tmp;
}
```

- 从参数空间或缓冲区加载工作项数据
- 使用16字节对齐的内存访问

### 2. copyToShmem16 函数

```cpp
inline __device__ void copyToShmem16(int tid, void* dst, void const* src, int bytes) {
  int offset = 16*tid;
  if (offset < bytes) {
    uint64_t a=0, b=0;
    asm volatile("ld.v2.u64 {%0,%1},[%2];" : "=l"(a),"=l"(b) : "l"((char const*)src + offset) : "memory");
    uint32_t udst = (uint32_t)__cvta_generic_to_shared(dst);
    asm volatile("st.shared.v2.u64 [%0],{%1,%2};" :: "r"(udst + offset), "l"(a), "l"(b) : "memory");
  }
}
```

**功能**：将16字节对齐的数据复制到共享内存。

**实现原理**：
- 使用双64位加载/存储指令进行高效内存访问
- 将通用地址转换为共享内存地址

### 3. globaltimer 函数

```cpp
__device__ __forceinline__ unsigned long long int globaltimer() {
  unsigned long long int timer;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(timer));
  return timer;
}
```

**功能**：获取全局时钟计数器。

**实现原理**：
- 使用PTX汇编指令 `%%globaltimer` 获取硬件时钟

## 模板结构定义

### 1. RunWorkColl 模板

```cpp
template<ncclFunc_t Fn, typename T, typename RedOp, int Algo, int Proto>
struct RunWorkColl {
  __device__ void run(int tid, int tn, struct ncclDevWorkColl* work) {
    // Put NOT IMPLEMENTED behavior here.
  }
};
```

**功能**：集合工作项运行器的通用模板。

**实现原理**：
- 定义通用接口，具体实现由特化版本提供
- 支持不同函数、数据类型、归约操作、算法和协议的组合

### 2. RunWorkBatch 模板

```cpp
template<ncclFunc_t Fn, typename T, typename RedOp, int Algo, int Proto>
struct RunWorkBatch {
  __device__ __forceinline__ void run() {
    int tid = threadIdx.x;
    int tn = blockDim.x;

    if (RedOpArg<RedOp>::ArgUsed) {
      int nWorks = ncclShmem.nWorks;
      for (int w=tid; w < nWorks; w += tn) {
        struct ncclDevWorkColl* work = (ncclDevWorkColl*)(ncclShmem.workStorage + w*ncclShmem.workSize);
        if (work->redOpArgIsPtr) {
          work->redOpArg = RedOpArg<RedOp>::loadArg(reinterpret_cast<void*>(work->redOpArg));
        }
      }
      __syncthreads();
    }

    #pragma unroll 1
    for (int w=0; w < ncclShmem.nWorks; w++) {
      struct ncclDevWorkColl* work = (struct ncclDevWorkColl*)(ncclShmem.workStorage + w*ncclShmem.workSize);
      if (w != 0) {
        struct ncclDevWorkColl* workPrev = (struct ncclDevWorkColl*)(ncclShmem.workStorage + (w-1)*ncclShmem.workSize);
        if (work->nWarps != workPrev->nWarps) __syncthreads();
      }
      int subtn = work->nWarps*WARP_SIZE;
      if (tid < subtn) RunWorkColl<Fn, T, RedOp, Algo, Proto>().run(tid, subtn, work);
    }
  }
};
```

**功能**：工作批次运行器的通用实现。

**实现原理**：

#### 2.1 参数处理
```cpp
if (RedOpArg<RedOp>::ArgUsed) {
  int nWorks = ncclShmem.nWorks;
  for (int w=tid; w < nWorks; w += tn) {
    struct ncclDevWorkColl* work = (ncclDevWorkColl*)(ncclShmem.workStorage + w*ncclShmem.workSize);
    if (work->redOpArgIsPtr) {
      work->redOpArg = RedOpArg<RedOp>::loadArg(reinterpret_cast<void*>(work->redOpArg));
    }
  }
  __syncthreads();
}
```

- 检查归约操作参数是否需要处理
- 如果参数是指针，则加载实际参数值

#### 2.2 工作项循环
```cpp
#pragma unroll 1
for (int w=0; w < ncclShmem.nWorks; w++) {
  struct ncclDevWorkColl* work = (struct ncclDevWorkColl*)(ncclShmem.workStorage + w*ncclShmem.workSize);
  if (w != 0) {
    struct ncclDevWorkColl* workPrev = (struct ncclDevWorkColl*)(ncclShmem.workStorage + (w-1)*ncclShmem.workSize);
    if (work->nWarps != workPrev->nWarps) __syncthreads();
  }
  int subtn = work->nWarps*WARP_SIZE;
  if (tid < subtn) RunWorkColl<Fn, T, RedOp, Algo, Proto>().run(tid, subtn, work);
}
```

- 循环处理每个工作项
- 根据工作项的warp数量调整同步策略
- 调用相应的集合工作项运行器

## 性能分析函数

### 1. profilerEnabled 函数

```cpp
__device__ __forceinline__ bool profilerEnabled(int workItemIdx) {
  return (ncclShmem.workType == ncclDevWorkTypeP2p) ?
    ((struct ncclDevWorkP2p*)ncclShmem.workStorage)[workItemIdx].profilerEnabled :
    ((struct ncclDevWorkColl*)ncclShmem.workStorage)[workItemIdx].profilerEnabled;
}
```

**功能**：检查指定工作项是否启用性能分析。

### 2. profiler 函数

```cpp
__device__ __forceinline__ void profiler(int action) {
  if (threadIdx.x == 0) {
    int idx = 0;
    uint64_t wc = ncclShmem.channel.workCounter + 1;
    if (action == START) {
      for (; wc <= ncclShmem.channel.workCounter + ncclShmem.nWorks; wc++) {
        if (!profilerEnabled(idx++)) continue;
        ncclShmem.comm.workStarted[ncclShmem.channelId].data[wc%MAX_PROFILER_EVENTS_PER_CHANNEL].timestamp = globaltimer();
        ncclShmem.comm.workStarted[ncclShmem.channelId].data[wc%MAX_PROFILER_EVENTS_PER_CHANNEL].counter = wc;
      }
    } else {
      for (; wc <= ncclShmem.channel.workCounter + ncclShmem.nWorks; wc++) {
        if (!profilerEnabled(idx++)) continue;
        ncclShmem.comm.workCompleted[ncclShmem.channelId].data[wc%MAX_PROFILER_EVENTS_PER_CHANNEL].timestamp = globaltimer();
        ncclShmem.comm.workCompleted[ncclShmem.channelId].data[wc%MAX_PROFILER_EVENTS_PER_CHANNEL].counter = wc;
      }
      ncclShmem.channel.workCounter += ncclShmem.nWorks;
      if (action == FINI) ((ncclKernelCommAndChannels*)ncclShmem.args.comm)->channels[ncclShmem.channelId].workCounter = ncclShmem.channel.workCounter;
    }
  }
}
```

**功能**：性能分析器，记录工作项的开始和完成时间。

**实现原理**：
- 仅由线程0执行性能分析
- 记录时间戳和计数器
- 支持开始、停止和完成三个动作

## 内核主函数

### 1. ncclKernelMain 函数

```cpp
template<int SpecializedFnId, typename SpecializedRunWorkBatch>
__device__ __forceinline__ void ncclKernelMain(struct ncclDevKernelArgs const* args) {
  int tid = threadIdx.x;
  int tn = blockDim.x;

  // Copy kernel args to shmem and then only read those. Otherwise the compiler
  // will end up putting the args into thread local stack which is very wasteful.
  if (tid < sizeof(ncclDevKernelArgs)/sizeof(uint32_t)) {
    ((uint32_t*)&ncclShmem.args)[tid] = ((uint32_t*)args)[tid];
  }

  // To map blockId to channelId, we need the n'th set bit of channelMask which
  // is the inverse of counting the number of set bits among the the first n.
  if (tid < MAXCHANNELS && (args->channelMask & (1ull<<tid))) {
    int n = __popcll(args->channelMask & ((1ull<<tid)-1));
    if (blockIdx.x == n) ncclShmem.channelId = tid;
  }
  __syncthreads(); // publish ncclShmem.{args, channelId}
```

**功能**：NCCL内核的主入口点函数。

**实现原理**：

#### 1.1 参数初始化
```cpp
// Copy kernel args to shmem and then only read those. Otherwise the compiler
// will end up putting the args into thread local stack which is very wasteful.
if (tid < sizeof(ncclDevKernelArgs)/sizeof(uint32_t)) {
  ((uint32_t*)&ncclShmem.args)[tid] = ((uint32_t*)args)[tid];
}
```

- 将内核参数复制到共享内存以避免线程局部栈浪费

#### 1.2 通道映射
```cpp
if (tid < MAXCHANNELS && (args->channelMask & (1ull<<tid))) {
  int n = __popcll(args->channelMask & ((1ull<<tid)-1));
  if (blockIdx.x == n) ncclShmem.channelId = tid;
}
```

- 将块ID映射到通道ID
- 使用人口计数指令高效处理通道掩码

#### 1.3 通信和通道信息加载
```cpp
// Use first 2 warps to load comm and channel, and remaining load work batch.
switch (tid/WARP_SIZE) {
case 0:
  { void* dst = &ncclShmem.comm;
    void* src = ncclShmem.args.comm;
    int bytes = sizeof(ncclKernelComm);
    copyToShmem16(tid, dst, src, bytes);
  } break;
case 1:
  { // Get address of channel without incurring indirect load from ncclKernelComm::channels
    void* dst = &ncclShmem.channel;
    void* src = &((ncclKernelCommAndChannels*)ncclShmem.args.comm)->channels[ncclShmem.channelId];
    int bytes = sizeof(ncclDevChannel);
    copyToShmem16(tid-WARP_SIZE, dst, src, bytes);
  } break;
default:
  { int subtid = tid - 2*WARP_SIZE;
    int subtn = tn - 2*WARP_SIZE;
    loadWorkBatchToShmem(subtid, subtn, args, /*batchIx=*/blockIdx.x);
  } break;
}
```

- 前两个warp负责加载通信和通道信息
- 其余线程负责加载工作批次

#### 1.4 主循环
```cpp
while (ncclShmem.aborted == 0) {
  profiler(START);
  if (0 <= SpecializedFnId && ncclShmem.funcId == (unsigned)SpecializedFnId) {
    SpecializedRunWorkBatch().run();
  } else {
    ncclDevFuncTable[ncclShmem.funcId]();
  }

  if (ncclShmem.nextBatchIx == -1) break;
  int batchIx = ncclShmem.nextBatchIx;
  __syncthreads();
  profiler(STOP);
  loadWorkBatchToShmem(tid, tn, args, batchIx);
  __syncthreads();
}
profiler(FINI);
```

- 持续处理工作批次直到完成或中止
- 支持特化函数和通用函数表调用
- 处理多个连续批次

## 预处理器宏定义

### 1. 内核定义宏

```cpp
#define DEFINE_ncclDevKernel(suffix, coll, redop, ty, algo, proto, specializedFnId) \
  __global__ void ncclDevKernel_##suffix(ncclDevKernelArgs4K NCCL_GRID_CONSTANT const args4K) { \
    ncclKernelMain<specializedFnId, RunWorkBatch<coll, ty, redop<ty>, algo, proto>>(&args4K.args); \
  }

#define DEFINE_ncclDevFunc(suffix, coll, redop, ty, algo, proto) \
  __device__ void ncclDevFunc_##suffix() { \
    RunWorkBatch<coll, ty, redop<ty>, algo, proto>().run(); \
  }
```

**功能**：简化内核和函数的定义。

**实现原理**：
- 自动生成特定于集合操作、数据类型、算法和协议的内核函数
- 便于维护和扩展不同配置的实现

## 性能优化特性

### 1. 内存优化
- **共享内存使用**：最大化共享内存利用率
- **16字节对齐**：优化内存访问模式
- **参数空间访问**：避免参数空间不可寻址问题

### 2. 同步优化
- **warp级同步**：使用 `__syncwarp()` 优化warp内部同步
- **条件同步**：根据实际情况选择同步策略
- **屏障对齐**：使用对齐的屏障指令

### 3. 编译优化
- **强制内联**：使用 `__forceinline__` 减少函数调用开销
- **循环展开**：使用 `#pragma unroll` 优化循环
- **模板特化**：针对特定类型和操作优化

### 4. 架构优化
- **CUDA 7.0+ 支持**：使用 `__grid_constant__` 优化
- **PTX 指令**：直接使用PTX汇编进行底层优化
- **硬件特性利用**：利用CUDA硬件特性进行优化

## 错误处理和可靠性

### 1. 中止机制
- **中止标志**：使用 `aborted` 标志快速传播错误
- **条件检查**：在关键位置检查中止状态

### 2. 边界检查
- **数组边界**：使用 `MAXCHANNELS` 和 `NCCL_MAX_*` 常量
- **指针验证**：在解引用前验证指针有效性

### 3. 同步保证
- **线程一致**：确保所有线程状态一致
- **内存可见**：使用同步指令保证内存可见性

## 总结

`common.h` 文件实现了 NCCL 设备端的核心基础设施，提供了：

1. **共享内存管理**：定义共享内存布局和访问接口
2. **同步原语**：提供多种同步机制和屏障操作
3. **工作负载管理**：实现工作项的加载和分发
4. **性能分析**：集成性能监控和分析功能
5. **内核架构**：定义内核入口点和执行框架
6. **优化策略**：内存对齐、同步优化、编译优化
7. **错误处理**：中止机制和可靠性保障

该文件是 NCCL 设备端实现的基础，为上层集合通信操作提供了高效、可靠的运行时支持。