# NCCL device/sendrecv_h_functions.md - Send/Recv Implementation

## 文件概述

`sendrecv.h` 是 NCCL 设备端代码的点对点 Send/Recv 操作实现文件，位于 `/root/nccl/src/device/` 目录下。该文件实现了点对点通信的 Send/Recv 操作，支持在同一设备内或跨设备之间传输数据。

## 核心结构详细分析

### 1. RunWorkBatch 模板特化

```cpp
template<typename T, typename RedOp>
struct RunWorkBatch<ncclFuncSendRecv, T, RedOp, NCCL_ALGO_RING, NCCL_PROTO_SIMPLE> {
  static_assert(sizeof(T)==1, "SendRecv only works on single byte types T.");
```

**功能**：Send/Recv 操作的批处理运行器。

**实现原理**：
- **类型约束**：使用 `static_assert` 确保只支持单字节类型
- **函数限制**：Send/Recv 操作只适用于单字节类型

### 2. runSend 函数模板

```cpp
template<typename Proto>
__device__ void runSend(int tid, int tn, int group, struct ncclDevWorkP2p* work) {
  size_t bytes = work->sendBytes;
  bool useLargeChunk = (work->sendIpcReg && ncclShmem.comm.isAllNvlink) || work->sendNetReg;
  int chunkSize = useLargeChunk ? NCCL_MAX_NET_SIZE : u32fp8Decode(work->sendChunkSize_u32fp8);
  int stepSize = useLargeChunk ? NCCL_MAX_NET_SIZE : ncclShmem.comm.p2pChunkSize;
  Primitives<T, RedOp, FanAsymmetric<0, 1>, 1, Proto, 1>
    prims(tid, tn, nullptr, &work->sendRank, work->sendAddr, nullptr,
          /*redOpArg(ignored)=*/0, group, 1, 1, nullptr, work, stepSize);
  size_t cursor = 0;
  do {
    int n = min(size_t(chunkSize), bytes-cursor);
    prims.directSend(cursor, cursor, n);
    cursor += n;
  } while (cursor < bytes);
}
```

**功能**：执行发送操作的函数。

**实现原理**：

#### 2.1 参数初始化
```cpp
size_t bytes = work->sendBytes;
bool useLargeChunk = (work->sendIpcReg && ncclShmem.comm.isAllNvlink) || work->sendNetReg;
int chunkSize = useLargeChunk ? NCCL_MAX_NET_SIZE : u32fp8Decode(work->sendChunkSize_u32fp8);
int stepSize = useLargeChunk ? NCCL_MAX_NET_SIZE : ncclShmem.comm.p2pChunkSize;
```

- **字节数获取**：获取需要发送的字节数
- **大块使用判断**：根据 IPC 注册或网络注册决定是否使用大块
- **块大小设置**：根据是否使用大块设置相应的块大小
- **步骤大小设置**：设置传输步骤大小

#### 2.2 原语对象创建
```cpp
Primitives<T, RedOp, FanAsymmetric<0, 1>, 1, Proto, 1>
  prims(tid, tn, nullptr, &work->sendRank, work->sendAddr, nullptr,
        /*redOpArg(ignored)=*/0, group, 1, 1, nullptr, work, stepSize);
```

- **扇形配置**：`FanAsymmetric<0, 1>` 表示无接收方，1个发送方
- **连接设置**：设置发送目标等级
- **缓冲区设置**：设置发送地址，接收地址为空

#### 2.3 分块发送循环
```cpp
size_t cursor = 0;
do {
  int n = min(size_t(chunkSize), bytes-cursor);
  prims.directSend(cursor, cursor, n);
  cursor += n;
} while (cursor < bytes);
```

- **游标初始化**：从0开始
- **块大小计算**：每次发送不超过块大小的字节数
- **直接发送**：使用 `directSend` 执行发送操作
- **游标更新**：移动游标到下一位置

### 3. runRecv 函数模板

```cpp
template<typename Proto>
__device__ void runRecv(int tid, int tn, int group, struct ncclDevWorkP2p* work) {
  size_t bytes = work->recvBytes;
  bool useLargeChunk = (work->recvIpcReg && ncclShmem.comm.isAllNvlink) || work->recvNetReg;
  int chunkSize = useLargeChunk ? NCCL_MAX_NET_SIZE : u32fp8Decode(work->recvChunkSize_u32fp8);
  int stepSize = useLargeChunk ? NCCL_MAX_NET_SIZE : ncclShmem.comm.p2pChunkSize;
  Primitives<T, RedOp, FanAsymmetric<1, 0>, 1, Proto, 1>
    prims(tid, tn, &work->recvRank, nullptr, nullptr, work->recvAddr,
          /*redOpArg(ignored)=*/0, group, 1, 1, nullptr, work, stepSize);
  size_t cursor = 0;
  do {
    int n = min(size_t(chunkSize), bytes-cursor);
    prims.directRecv(cursor, n);
    cursor += n;
  } while (cursor < bytes);
}
```

**功能**：执行接收操作的函数。

**实现原理**：

#### 3.1 参数初始化
与 `runSend` 相同，但使用接收相关参数

#### 3.2 原语对象创建
```cpp
Primitives<T, RedOp, FanAsymmetric<1, 0>, 1, Proto, 1>
  prims(tid, tn, &work->recvRank, nullptr, nullptr, work->recvAddr,
        /*redOpArg(ignored)=*/0, group, 1, 1, nullptr, work, stepSize);
```

- **扇形配置**：`FanAsymmetric<1, 0>` 表示1个接收方，无发送方
- **连接设置**：设置接收来源等级
- **缓冲区设置**：设置接收地址，发送地址为空

#### 3.3 分块接收循环
```cpp
size_t cursor = 0;
do {
  int n = min(size_t(chunkSize), bytes-cursor);
  prims.directRecv(cursor, n);
  cursor += n;
} while (cursor < bytes);
```

- **直接接收**：使用 `directRecv` 执行接收操作

### 4. 主运行函数

```cpp
__device__ __forceinline__ void run() {
  const int tid = threadIdx.x;
  const int tn = blockDim.x;
  const int wid = tid/WARP_SIZE;
  const int nWarps = tn/WARP_SIZE;
  const int lane = tid%WARP_SIZE;
```

**功能**：Send/Recv 操作的主运行函数。

**实现原理**：

#### 4.1 线程信息初始化
```cpp
const int tid = threadIdx.x;
const int tn = blockDim.x;
const int wid = tid/WARP_SIZE;
const int nWarps = tn/WARP_SIZE;
const int lane = tid%WARP_SIZE;
```

- **线程ID**：获取当前线程ID
- **线程总数**：获取线程块中线程总数
- **warpID**：计算warp ID
- **warp总数**：计算线程块中warp总数
- **lane ID**：计算在warp中的lane ID

#### 4.2 共享结构定义
```cpp
struct Shared {
  uint32_t workSendMask; // bitmasks of which work indices have send/recv
  uint32_t workRecvMask;
};
Shared* shared = (Shared*)ncclScratchForWarp(0);
```

- **掩码结构**：定义包含发送和接收工作掩码的共享结构
- **共享内存**：分配共享内存空间

#### 4.3 工作项初始化
```cpp
struct ncclDevWorkP2p* works = (ncclDevWorkP2p*)ncclShmem.workStorage;
int nWorks = ncclShmem.nWorks;
```

- **工作项获取**：获取点对点工作项存储
- **工作数获取**：获取工作项数量

#### 4.4 工作项分区计算
```cpp
if (wid == 0) {
  // Modify the memory range of each work[] to reflect this channel's
  // partition of the work. Since integer divides are very heavy it's
  // best to do them all in one warp.
  int workIx = lane%16;
  int isSend = lane < 16 ? 0 : 1;
  bool hasWork = false;
  if (workIx < nWorks) {
    struct ncclDevWorkP2p* work = &works[workIx];
    size_t bytes = isSend ? work->sendBytes : work->recvBytes;
    int nParts = isSend ? work->nSendChannels : work->nRecvChannels;
    int part = ncclP2pChannelToPart(work->nP2pChannels, work->channelBase, ncclShmem.channelId);
    hasWork = (part < nParts);
    if (nParts != 0) {
      size_t partBeg, partEnd;
      ncclP2pPartBounds(nParts, part, bytes, &partBeg, &partEnd);
      (isSend ? work->sendAddr : work->recvAddr) = (char*)(isSend ? work->sendAddr : work->recvAddr) + partBeg;
      (isSend ? work->sendBytes : work->recvBytes) = partEnd - partBeg;
    }
  }
  uint32_t mask = __ballot_sync(~0u, hasWork);
  if (lane == 0) {
    shared->workSendMask = mask>>16;
    shared->workRecvMask = mask & 0xffff;
  }
}
```

**实现原理**：
- **分区计算**：计算当前通道在工作项中的分区
- **边界计算**：计算分区的起始和结束位置
- **地址调整**：调整发送/接收地址到分区位置
- **字节数调整**：调整发送/接收字节数为分区大小
- **掩码生成**：使用 `__ballot_sync` 生成发送/接收掩码

#### 4.5 warp分配计算
```cpp
// The fastest way to compute a warp uniform division x/y in [0,32) is to
// use each lane to guess a solution and count the ones that don't exceed
// the numerator:
//   __popc(__ballot_sync(~0u, y*(lane+1) <= x))
// That takes 1/3 the time of standard division and about 3/4 the time of
// approximate floating point division:
//   __float2int_rd(__fdividef(float(x),float(y))).

// nWarpPerWork = nWarps/nWorks
int nWarpPerWork = __popc(__ballot_sync(~0u, nWorks*(lane+1) <= nWarps));
int nRecvWarpPerWork = nWarpPerWork<=4 ? nWarpPerWork/2 : (nWarpPerWork-1)/2;
int nSendWarpPerWork = nWarpPerWork<=4 ? nRecvWarpPerWork : nRecvWarpPerWork+1;
```

**实现原理**：
- **高效除法**：使用位操作替代标准除法
- **warp分配**：计算每个工作项分配的warp数
- **接收warp分配**：计算接收操作的warp数
- **发送warp分配**：计算发送操作的warp数

#### 4.6 工作项索引计算
```cpp
// The work index this warp belongs to: workIx = wid/nWarpPerWork
int workIx = __popc(__ballot_sync(~0u, (lane+1)*nWarpPerWork <= wid));
```

**实现原理**：
- **工作索引**：计算当前warp所属的工作项索引

#### 4.7 同步操作
```cpp
__syncthreads(); // Wait for works[] and shared->* to be updated by warp=0
uint32_t workSendMask = shared->workSendMask;
uint32_t workRecvMask = shared->workRecvMask;
__syncthreads(); // release scratch space used by shared->*
```

**实现原理**：
- **第一次同步**：等待warp 0完成工作项初始化
- **掩码获取**：获取发送和接收掩码
- **第二次同步**：释放共享内存空间

#### 4.8 线程范围计算
```cpp
if (nWorks <= workIx) return;

// Thread range for whole work (send & recv combined)
int subtid = tid - workIx*nWarpPerWork*WARP_SIZE;
int subtn = nWarpPerWork*WARP_SIZE;
```

**实现原理**：
- **边界检查**：检查工作索引是否超出范围
- **子线程ID**：计算在当前工作项中的线程ID
- **子线程总数**：计算当前工作项中的线程总数

#### 4.9 组ID计算
```cpp
// A send primtive of sufficient size requires 2 cuda barrier ids.
constexpr int nSendWarpsForExtraGroup = NCCL_SIMPLE_EXTRA_GROUP_IF_NTHREADS_GE/WARP_SIZE;
// Count up all group ids used below this workIx:
int group, extra;
// Each recv gets one group id:
group = __popc(workRecvMask & ((1<<workIx)-1));
// Sends accompanying recvs get one and maybe an extra:
extra = (nSendWarpPerWork >= nSendWarpsForExtraGroup) ? 1 : 0;
group += __popc((workSendMask & workRecvMask) & ((1<<workIx)-1))*(1+extra);
// Sends without recvs use more warps so compute extra accordingly:
extra = (nWarpPerWork >= nSendWarpsForExtraGroup) ? 1 : 0;
group += __popc((workSendMask & ~workRecvMask) & ((1<<workIx)-1))*(1+extra);
```

**实现原理**：
- **额外组计算**：计算需要额外组ID的发送warp数
- **组ID分配**：为每个工作项分配唯一的组ID
- **额外ID计算**：根据warp数决定是否需要额外的组ID

#### 4.10 工作项处理
```cpp
struct ncclDevWorkP2p* work = &works[workIx];
bool hasSend = 1 & (workSendMask>>workIx);
bool hasRecv = 1 & (workRecvMask>>workIx);
bool isCopy = work->sendRank == ncclShmem.comm.rank;
bool isSend = !hasRecv || (hasSend && subtid < nSendWarpPerWork*WARP_SIZE);
```

**实现原理**：
- **工作项获取**：获取当前工作项
- **发送检查**：检查是否包含发送操作
- **接收检查**：检查是否包含接收操作
- **复制检查**：检查是否为本地复制操作
- **发送判断**：判断当前线程是否执行发送

#### 4.11 线程ID调整
```cpp
if (!isCopy && hasSend && hasRecv) {
  // Translate thread ids to reflect just this send or recv as opposed to whole work.
  if (isSend) {
    subtn = nSendWarpPerWork*WARP_SIZE;
  } else {
    subtid -= nSendWarpPerWork*WARP_SIZE;
    subtn = nRecvWarpPerWork*WARP_SIZE;
    group += 1 + (nSendWarpPerWork >= nSendWarpsForExtraGroup ? 1 : 0);
  }
}
```

**实现原理**：
- **线程ID转换**：将线程ID调整为仅针对发送或接收操作
- **发送线程**：调整发送操作的线程范围
- **接收线程**：调整接收操作的线程ID和范围
- **组ID调整**：为接收操作分配新的组ID

#### 4.12 操作执行
```cpp
if (isCopy) {
  reduceCopy<COLL_UNROLL, RedOp, T, 0,1,1, 0,1,1, /*PreOpSrcs=*/0>
    (subtid, subtn, 0, false, 1, &work->sendAddr, 1, &work->recvAddr, (ssize_t)work->sendBytes);
} else if (isSend) {
  if (work->sendProtoLL) {
    runSend<ProtoLL>(subtid, subtn, group, work);
  } else {
    runSend<ProtoSimple<1,1>>(subtid, subtn, group, work);
  }
} else {
  if (work->recvProtoLL) {
    runRecv<ProtoLL>(subtid, subtn, group, work);
  } else {
    runRecv<ProtoSimple<1,1>>(subtid, subtn, group, work);
  }
}
```

**实现原理**：
- **本地复制**：使用 `reduceCopy` 执行本地内存复制
- **发送操作**：根据协议类型调用 `runSend`
- **接收操作**：根据协议类型调用 `runRecv`
- **协议选择**：支持 LL 和 Simple 协议

## Send/Recv 算法特点

### 1. 点对点通信
- **直接传输**：在两个特定进程之间直接传输
- **单向操作**：支持单独的发送或接收操作
- **双向操作**：支持同时进行发送和接收

### 2. 动态分区
- **通道分区**：根据通道ID动态计算分区
- **内存划分**：将大内存区域划分为多个分区
- **负载均衡**：在多个通道间平衡负载

### 3. 自适应协议
- **LL协议**：低延迟协议，适用于小数据
- **Simple协议**：高性能协议，适用于大数据
- **协议选择**：根据工作项配置选择协议

### 4. 内存优化
- **大块支持**：根据IPC/网络注册支持大块传输
- **分块处理**：将大数据分成小块处理
- **直接内存访问**：支持GPU直接内存访问

## 性能优化特性

### 1. 线程管理
- **warp分配**：智能分配warp用于发送和接收
- **负载均衡**：平衡发送和接收线程负载
- **高效除法**：使用位操作优化除法运算

### 2. 内存访问
- **缓存友好**：优化内存访问模式
- **分块处理**：减少单次传输大小
- **直接访问**：支持直接内存访问

### 3. 同步优化
- **warp同步**：使用warp级同步
- **块级同步**：使用块级同步协调
- **掩码操作**：使用掩码快速判断状态

### 4. 协议优化
- **动态选择**：根据场景选择最优协议
- **参数调整**：根据硬件特性调整参数

## 错误处理和可靠性

### 1. 边界检查
- **索引验证**：验证工作项索引范围
- **字节验证**：验证传输字节数
- **缓冲区边界**：确保不超出缓冲区边界

### 2. 同步保证
- **线程一致**：确保所有线程状态一致
- **内存可见**：确保内存修改对所有线程可见

### 3. 类型安全
- **类型约束**：使用 `static_assert` 确保类型安全
- **单字节限制**：仅支持单字节类型

## 应用场景

### 1. 点对点通信
- 在分布式系统中进行进程间直接通信
- 用于特定的数据传输需求

### 2. 数据分发
- 将数据从一个进程分发到另一个进程
- 用于数据预处理和后处理

### 3. 模型同步
- 在分布式训练中同步模型参数
- 用于特定的同步需求

## 总结

`sendrecv.h` 文件实现了 NCCL 点对点 Send/Recv 操作，提供了：

1. **点对点通信**：在两个进程间直接传输数据
2. **动态分区**：根据通道动态划分工作
3. **自适应协议**：支持 LL 和 Simple 协议
4. **性能优化**：warp分配、内存优化、同步优化
5. **类型安全**：限制为单字节类型确保安全
6. **灵活性**：支持发送、接收、复制等多种操作

该文件是 NCCL 点对点通信的核心实现，通过智能的资源分配和协议选择实现了高效的点对点数据传输。