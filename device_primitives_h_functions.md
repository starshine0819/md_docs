# NCCL device/primitives_h_functions.md - Communication Primitives Framework

## 文件概述

`primitives.h` 是 NCCL 设备端通信原语框架的核心定义文件，位于 `/root/nccl/src/device/` 目录下。该文件定义了 NCCL 的三种主要通信协议（Simple、LL、LL128）、扇入/扇出结构以及原始操作框架，为高效 GPU 间通信提供了基础架构。

## 核心常量定义

### 1. 中断检查常量

```cpp
#define NCCL_SPINS_BEFORE_CHECK_ABORT 10000
```

**功能**：定义在检查中止标志之前的自旋次数。

**实现原理**：
- **性能优化**：避免频繁检查中止标志影响性能
- **平衡策略**：在性能和响应性之间取得平衡
- **自旋计数**：每10000次自旋检查一次中止标志

## 协议类定义

### 1. ProtoSimple 结构

```cpp
template<int SlicePerChunk_1, int StepPerSlice_1, int Unroll_1 = COLL_UNROLL, int MultimemSrcs_1 = 0, int MultimemDsts_1 = 0>
struct ProtoSimple {
  static constexpr int Id = NCCL_PROTO_SIMPLE;
  static constexpr int SlicePerChunk = SlicePerChunk_1;
  static constexpr int StepPerSlice = StepPerChunk_1;
  static constexpr int Unroll = Unroll_1;
  static constexpr int MultimemSrcs = MultimemSrcs_1;
  static constexpr int MultimemDsts = MultimemDsts_1;

  // Data bytes (no flags etc) in one step of the fifo queue.
  __device__ static int calcBytePerStep() {
    return ncclShmem.comm.buffSizes[NCCL_PROTO_SIMPLE]/NCCL_STEPS;
  }
  // Granularity of data bytes transferred per thread.
  __device__ static int calcBytePerGrain() {
    return sizeof(uint64_t); // Bogus value? Nobody queries this metric for simple.
  }
  // Group width is how many consecutive group values a subchannel occupies.
  static constexpr int MaxGroupWidth = 2;
};
```

**功能**：定义 Simple 协议的参数和计算方法。

**实现原理**：

#### 1.1 模板参数
- **SlicePerChunk**：每个块的切片数量
- **StepPerSlice**：每个切片的步骤数量
- **Unroll**：循环展开参数，默认为 `COLL_UNROLL`
- **MultimemSrcs/Dsts**：多内存源/目标数量

#### 1.2 字节计算方法
```cpp
__device__ static int calcBytePerStep() {
  return ncclShmem.comm.buffSizes[NCCL_PROTO_SIMPLE]/NCCL_STEPS;
}
```

- **缓冲区大小**：从共享内存获取 Simple 协议的缓冲区大小
- **步骤划分**：按步骤数平均分配字节数
- **FIFO队列**：计算 FIFO 队列每步的数据字节数

#### 1.3 粒度计算方法
```cpp
__device__ static int calcBytePerGrain() {
  return sizeof(uint64_t); // Bogus value? Nobody queries this metric for simple.
}
```

- **粒度单位**：返回 64 位作为粒度单位
- **注释说明**：Simple 协议通常不需要查询此指标

#### 1.4 最大组宽度
- **值为2**：Simple 协议的最大组宽度为 2

### 2. ProtoLL 结构

```cpp
struct ProtoLL {
  static constexpr int Id = NCCL_PROTO_LL;

  // Data bytes (no flags etc) in one step of the fifo queue.
  __device__ static int calcBytePerStep() {
    return ncclShmem.comm.buffSizes[NCCL_PROTO_LL]/NCCL_STEPS/2; // Half is data
  }
  // Granularity of data bytes transferred per thread.
  __device__ static int calcBytePerGrain() {
    return sizeof(uint64_t); // One 16-byte line has 8-bytes of data
  }
  // Group width is how many consecutive group values a subchannel occupies.
  static constexpr int MaxGroupWidth = 1;
};
```

**功能**：定义 Low-Latency (LL) 协议的参数和计算方法。

**实现原理**：

#### 2.1 字节计算方法
```cpp
__device__ static int calcBytePerStep() {
  return ncclShmem.comm.buffSizes[NCCL_PROTO_LL]/NCCL_STEPS/2; // Half is data
}
```

- **数据减半**：LL 协议中一半是数据，一半是控制信息
- **缓冲区分配**：从 LL 协议缓冲区大小计算
- **高效传输**：适用于小消息的低延迟传输

#### 2.2 粒度计算方法
```cpp
__device__ static int calcBytePerGrain() {
  return sizeof(uint64_t); // One 16-byte line has 8-bytes of data
}
```

- **数据比例**：16 字节行中有 8 字节数据
- **LL特性**：反映 LL 协议的数据密度

#### 2.3 最大组宽度
- **值为1**：LL 协议的最大组宽度为 1

### 3. ProtoLL128 结构

```cpp
struct ProtoLL128 {
  static constexpr int Id = NCCL_PROTO_LL128;

  // Data bytes (no flags etc) in one step of the fifo queue.
  __device__ static int calcBytePerStep() {
    return (ncclShmem.comm.buffSizes[NCCL_PROTO_LL128]/NCCL_STEPS)*NCCL_LL128_DATAELEMS/NCCL_LL128_LINEELEMS;
  }
  // Granularity of data bytes transferred per thread.
  __device__ static int calcBytePerGrain() {
    return NCCL_LL128_SHMEM_ELEMS_PER_THREAD*NCCL_LL128_DATAELEMS*sizeof(uint64_t)/NCCL_LL128_LINEELEMS;
  }
  // Group width is how many consecutive group values a subchannel occupies.
  static constexpr int MaxGroupWidth = 1;
};
```

**功能**：定义 LL128 协议的参数和计算方法。

**实现原理**：

#### 3.1 字节计算方法
```cpp
__device__ static int calcBytePerStep() {
  return (ncclShmem.comm.buffSizes[NCCL_PROTO_LL128]/NCCL_STEPS)*NCCL_LL128_DATAELEMS/NCCL_LL128_LINEELEMS;
}
```

- **LL128特性**：使用数据元素比例计算实际数据字节数
- **缓冲区分配**：基于 LL128 缓冲区大小
- **元素比例**：`NCCL_LL128_DATAELEMS/NCCL_LL128_LINEELEMS` 表示数据元素占总元素的比例

#### 3.2 粒度计算方法
```cpp
__device__ static int calcBytePerGrain() {
  return NCCL_LL128_SHMEM_ELEMS_PER_THREAD*NCCL_LL128_DATAELEMS*sizeof(uint64_t)/NCCL_LL128_LINEELEMS;
}
```

- **线程粒度**：基于每个线程的共享内存元素数
- **数据密度**：考虑数据元素与总元素的比例
- **字节计算**：最终转换为字节数

#### 3.3 最大组宽度
- **值为1**：LL128 协议的最大组宽度为 1

## 扇入/扇出类定义

### 1. FanAsymmetric 结构

```cpp
template<int MaxRecv_, int MaxSend_>
struct FanAsymmetric {
  static constexpr int MaxRecv = MaxRecv_, MaxSend = MaxSend_;
  int nr, ns;
  FanAsymmetric() = default;
  __device__ FanAsymmetric(int nrecv, int nsend): nr(nrecv), ns(nsend) {
    // assert(nrecv <= MaxRecv && nsend <= MaxSend);
  }
  __device__ int nrecv() const { return MaxRecv ? nr : 0; }
  __device__ int nsend() const { return MaxSend ? ns : 0; }
};
```

**功能**：定义非对称扇入/扇出结构。

**实现原理**：

#### 1.1 模板参数
- **MaxRecv_**：最大接收数量的编译时常量
- **MaxSend_**：最大发送数量的编译时常量

#### 1.2 成员变量
- **nr**：运行时接收数量
- **ns**：运行时发送数量

#### 1.3 构造函数
```cpp
__device__ FanAsymmetric(int nrecv, int nsend): nr(nrecv), ns(nsend) {
  // assert(nrecv <= MaxRecv && nsend <= MaxSend);
}
```

- **参数传递**：将接收和发送数量存储到成员变量
- **边界检查**：通过注释提示应进行边界检查

#### 1.4 查询方法
```cpp
__device__ int nrecv() const { return MaxRecv ? nr : 0; }
__device__ int nsend() const { return MaxSend ? ns : 0; }
```

- **条件返回**：如果最大值为0，则返回0；否则返回实际值
- **零优化**：避免对零值进行不必要的访问

### 2. FanSymmetric 结构

```cpp
template<int MaxArity>
struct FanSymmetric {
  static constexpr int MaxRecv = MaxArity, MaxSend = MaxArity;
  int n;
  FanSymmetric() = default;
  __device__ FanSymmetric(int nrecv, int nsend): n(nrecv) {
    // assert(nrecv == nsend && nrecv <= MaxArity);
  }
  __device__ int nrecv() const { return n; }
  __device__ int nsend() const { return n; }
};
```

**功能**：定义对称扇入/扇出结构。

**实现原理**：

#### 2.1 模板参数
- **MaxArity**：最大元数（接收=发送）

#### 2.2 成员变量
- **n**：统一的接收/发送数量

#### 2.3 构造函数
```cpp
__device__ FanSymmetric(int nrecv, int nsend): n(nrecv) {
  // assert(nrecv == nsend && nrecv <= MaxArity);
}
```

- **对称保证**：假设接收数量等于发送数量
- **单一存储**：只需存储一个数值

#### 2.4 查询方法
```cpp
__device__ int nrecv() const { return n; }
__device__ int nsend() const { return n; }
```

- **统一返回**：接收和发送都返回同一个值
- **内存优化**：节省寄存器空间

## 原始操作框架

### 1. Primitives 类声明

```cpp
template<typename T, typename RedOp, typename Fan, int Direct, typename Proto, int P2p, bool isNetOffload = false>
class Primitives;
```

**功能**：声明原始操作类的模板。

**模板参数**：
- **T**：数据类型
- **RedOp**：归约操作类型
- **Fan**：扇入/扇出类型
- **Direct**：直接访问标志
- **Proto**：协议类型
- **P2p**：点对点标志
- **isNetOffload**：网络卸载标志

### 2. PrimitivesWithoutDirect 结构

```cpp
template<typename RealPrimitives>
struct PrimitivesWithoutDirect {
  __device__ void directSend(intptr_t inpIx, intptr_t outIx, int eltN) {
    static_cast<RealPrimitives*>(this)->send(inpIx, eltN);
  }
  __device__ void directSendFromOutput(intptr_t outIx, int eltN) {
    static_cast<RealPrimitives*>(this)->sendFromOutput(outIx, eltN);
  }
  __device__ void directRecv(intptr_t outIx, int eltN) {
    static_cast<RealPrimitives*>(this)->recv(outIx, eltN, /*postOp=*/false);
  }
  __device__ void directCopySend(intptr_t inpIx, intptr_t outIx, int eltN, bool postOp=false) {
    static_cast<RealPrimitives*>(this)->copySend(inpIx, outIx, eltN, postOp);
  }
  __device__ void directRecvCopyDirectSend(intptr_t inpIx, intptr_t outIx, int eltN, bool postOp=false) {
    static_cast<RealPrimitives*>(this)->recvCopySend(outIx, eltN, /*postOp=*/false);
  }
  __device__ void directRecvDirectSend(intptr_t inpIx, intptr_t outIx, int eltN, bool postOp=false) {
    return;
  }
  __device__ void recvReduceCopyDirectSend(intptr_t inpIx, intptr_t outIx, int eltN, bool postOp=false) {
    // Direct is only for the send part
    static_cast<RealPrimitives*>(this)->recvReduceCopySend(inpIx, outIx, eltN, postOp);
  }
  __device__ __forceinline__ void directRecvReduceDirectSend(intptr_t inpIx, intptr_t outIx, ssize_t eltN, bool postOp=false) {
    static_cast<RealPrimitives*>(this)->recvReduceSend(inpIx, eltN);
  }
  __device__ __forceinline__ void directRecvReduceCopyDirectSend(intptr_t inpIx, intptr_t outIx, ssize_t eltN, bool postOp=false) {
    static_cast<RealPrimitives*>(this)->recvReduceCopySend(inpIx, outIx, eltN, postOp);
  }
};
```

**功能**：为不支持直接操作的原始操作提供默认实现。

**实现原理**：

#### 2.1 CRTP 模式
- **静态转换**：使用 `static_cast<RealPrimitives*>` 实现 CRTP
- **动态调度**：在编译时确定实际类型

#### 2.2 直接发送操作
```cpp
__device__ void directSend(intptr_t inpIx, intptr_t outIx, int eltN) {
  static_cast<RealPrimitives*>(this)->send(inpIx, eltN);
}
```

- **委托实现**：将直接发送委托给实际的 send 方法
- **简化接口**：提供简化的直接发送接口

#### 2.3 直接发送从输出
```cpp
__device__ void directSendFromOutput(intptr_t outIx, int eltN) {
  static_cast<RealPrimitives*>(this)->sendFromOutput(outIx, eltN);
}
```

- **输出发送**：从输出缓冲区直接发送
- **性能优化**：避免中间复制

#### 2.4 直接接收操作
```cpp
__device__ void directRecv(intptr_t outIx, int eltN) {
  static_cast<RealPrimitives*>(this)->recv(outIx, eltN, /*postOp=*/false);
}
```

- **直接接收**：将数据直接接收到底部
- **无后处理**：明确指定不进行后处理

#### 2.5 直接复制发送
```cpp
__device__ void directCopySend(intptr_t inpIx, intptr_t outIx, int eltN, bool postOp=false) {
  static_cast<RealPrimitives*>(this)->copySend(inpIx, outIx, eltN, postOp);
}
```

- **复制发送**：复制数据并发送
- **后处理支持**：可选的后处理操作

#### 2.6 接收复制直接发送
```cpp
__device__ void directRecvCopyDirectSend(intptr_t inpIx, intptr_t outIx, int eltN, bool postOp=false) {
  static_cast<RealPrimitives*>(this)->recvCopySend(outIx, eltN, /*postOp=*/false);
}
```

- **链式操作**：接收、复制、发送的组合操作
- **优化路径**：为特定操作模式提供优化

#### 2.7 接收归约复制直接发送
```cpp
__device__ void recvReduceCopyDirectSend(intptr_t inpIx, intptr_t outIx, int eltN, bool postOp=false) {
  // Direct is only for the send part
  static_cast<RealPrimitives*>(this)->recvReduceCopySend(inpIx, outIx, eltN, postOp);
}
```

- **归约操作**：执行接收、归约、复制、发送
- **发送优化**：直接发送部分优化

## 中止检查函数

### 1. checkAbort 函数

```cpp
__device__ inline int checkAbort(int &abortCache, const int abortValue, int &spins) {
  if (abortCache & abortValue) return 1;
  if (++spins < NCCL_SPINS_BEFORE_CHECK_ABORT) return 0;
  spins = 0;
  int abort = *ncclShmem.comm.abortFlag;
  if (abort) {
    ncclShmem.aborted = abort;
    abortCache |= abortValue;
  }
  return abort;
}
```

**功能**：检查通信中止标志。

**实现原理**：

#### 1.1 缓存检查
```cpp
if (abortCache & abortValue) return 1;
```

- **快速路径**：如果缓存中已有中止标志，立即返回
- **性能优化**：避免重复检查

#### 1.2 自旋计数
```cpp
if (++spins < NCCL_SPINS_BEFORE_CHECK_ABORT) return 0;
spins = 0;
```

- **频率控制**：控制中止检查的频率
- **性能平衡**：避免过于频繁的全局内存访问

#### 1.3 全局检查
```cpp
int abort = *ncclShmem.comm.abortFlag;
if (abort) {
  ncclShmem.aborted = abort;
  abortCache |= abortValue;
}
```

- **标志读取**：从共享内存读取中止标志
- **状态更新**：更新本地中止状态
- **缓存更新**：更新中止缓存

#### 1.4 返回结果
- **中止状态**：返回中止标志值
- **错误传播**：将中止状态传播给调用者

## 设计特点

### 1. 模板化设计
- **类型安全**：使用模板确保类型安全
- **编译时优化**：在编译时确定所有参数
- **零成本抽象**：避免运行时开销

### 2. 协议抽象
- **统一接口**：为不同协议提供统一的计算接口
- **性能优化**：每个协议都有针对性的优化
- **可扩展性**：易于添加新的协议类型

### 3. 内存效率
- **寄存器优化**：最小化寄存器使用
- **共享内存**：有效利用共享内存
- **对称优化**：对称结构节省空间

### 4. 错误处理
- **中止机制**：完善的通信中止处理
- **快速响应**：平衡性能和响应性
- **状态同步**：保持全局状态一致

## 性能优化特性

### 1. 循环展开
- **编译时展开**：使用 `COLL_UNROLL` 控制展开
- **减少开销**：降低循环控制开销

### 2. 内存访问优化
- **批量访问**：优化内存访问模式
- **对齐访问**：确保内存访问对齐
- **缓存友好**：优化缓存使用

### 3. 寄存器优化
- **最小化使用**：减少寄存器占用
- **预测寄存器**：优化预测寄存器使用
- **对称结构**：对称结构节省寄存器

## 应用场景

### 1. 多 GPU 通信
- **高效传输**：支持多种协议的高效数据传输
- **灵活配置**：可根据消息大小选择最优协议
- **扇形操作**：支持复杂的扇入/扇出通信模式

### 2. 集合通信
- **归约操作**：支持各种归约操作
- **同步机制**：提供可靠的同步机制
- **错误恢复**：具备错误检测和恢复能力

### 3. 网络通信
- **协议适配**：支持网络通信的协议适配
- **卸载优化**：支持网络卸载优化
- **点对点**：支持点对点通信

## 总结

`primitives.h` 文件定义了 NCCL 通信原语的核心框架，提供了：

1. **协议抽象**：Simple、LL、LL128 三种协议的统一抽象
2. **扇形结构**：对称和非对称扇入/扇出结构
3. **模板框架**：类型安全的原始操作框架
4. **性能优化**：多种性能优化技术
5. **错误处理**：完善的中止检查机制
6. **可扩展性**：易于扩展新的协议和功能

该文件是 NCCL 通信原语的基础架构，为高效 GPU 间通信提供了灵活且高性能的抽象层。