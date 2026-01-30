# NCCL device/primitives.h 函数实现详细分析

## 文件概述

`primitives.h` 是 NCCL 设备端代码的基元操作头文件，位于 `/root/nccl/src/device/` 目录下。该文件定义了 NCCL 通信协议的基础结构、扇入/扇出类、原语模板以及相关的辅助函数，为不同的通信协议（Simple、LL、LL128）提供了统一的接口和实现框架。

## 核心数据结构详细分析

### 1. ProtoSimple 结构体

```cpp
template<int SlicePerChunk_1, int StepPerSlice_1, int Unroll_1 = COLL_UNROLL, int MultimemSrcs_1 = 0, int MultimemDsts_1 = 0>
struct ProtoSimple {
  static constexpr int Id = NCCL_PROTO_SIMPLE;
  static constexpr int SlicePerChunk = SlicePerChunk_1;
  static constexpr int StepPerSlice = StepPerSlice_1;
  static constexpr int Unroll = Unroll_1;
  static constexpr int MultimemSrcs = MultimemSrcs_1;
  static constexpr int MultimemDsts = MultimemDsts_1;

  __device__ static int calcBytePerStep() {
    return ncclShmem.comm.buffSizes[NCCL_PROTO_SIMPLE]/NCCL_STEPS;
  }
  __device__ static int calcBytePerGrain() {
    return sizeof(uint64_t); // Bogus value? Nobody queries this metric for simple.
  }
  static constexpr int MaxGroupWidth = 2;
};
```

**功能**：定义 Simple 协议的参数和计算方法。

**实现原理**：
- **模板参数**：
  - `SlicePerChunk_1`：每个块的切片数量
  - `StepPerSlice_1`：每个切片的步骤数量
  - `Unroll_1`：循环展开因子（默认为 `COLL_UNROLL`）
  - `MultimemSrcs_1`：多内存源数量
  - `MultimemDsts_1`：多内存目标数量
- **协议标识**：`Id` 设置为 `NCCL_PROTO_SIMPLE`
- **字节计算**：
  - `calcBytePerStep()`：计算每步的字节数，等于缓冲区大小除以步骤数
  - `calcBytePerGrain()`：返回 `sizeof(uint64_t)`，注释表明这是个虚拟值
- **组宽度**：`MaxGroupWidth` 设置为 2，表示子通道占据的连续组值数量

### 2. ProtoLL 结构体

```cpp
struct ProtoLL {
  static constexpr int Id = NCCL_PROTO_LL;

  __device__ static int calcBytePerStep() {
    return ncclShmem.comm.buffSizes[NCCL_PROTO_LL]/NCCL_STEPS/2; // Half is data
  }
  __device__ static int calcBytePerGrain() {
    return sizeof(uint64_t); // One 16-byte line has 8-bytes of data
  }
  static constexpr int MaxGroupWidth = 1;
};
```

**功能**：定义 Low-Latency (LL) 协议的参数和计算方法。

**实现原理**：
- **协议标识**：`Id` 设置为 `NCCL_PROTO_LL`
- **字节计算**：
  - `calcBytePerStep()`：计算每步的字节数，等于缓冲区大小除以步骤数再除以2（因为一半是数据，一半是标志）
  - `calcBytePerGrain()`：返回 `sizeof(uint64_t)`，因为每 16 字节行中有 8 字节数据
- **组宽度**：`MaxGroupWidth` 设置为 1

### 3. ProtoLL128 结构体

```cpp
struct ProtoLL128 {
  static constexpr int Id = NCCL_PROTO_LL128;

  __device__ static int calcBytePerStep() {
    return (ncclShmem.comm.buffSizes[NCCL_PROTO_LL128]/NCCL_STEPS)*NCCL_LL128_DATAELEMS/NCCL_LL128_LINEELEMS;
  }
  __device__ static int calcBytePerGrain() {
    return NCCL_LL128_SHMEM_ELEMS_PER_THREAD*NCCL_LL128_DATAELEMS*sizeof(uint64_t)/NCCL_LL128_LINEELEMS;
  }
  static constexpr int MaxGroupWidth = 1;
};
```

**功能**：定义 LL128 协议的参数和计算方法。

**实现原理**：
- **协议标识**：`Id` 设置为 `NCCL_PROTO_LL128`
- **字节计算**：
  - `calcBytePerStep()`：计算每步的字节数，考虑了数据元素和行元素的比例
  - `calcBytePerGrain()`：计算每线程的字节数，基于共享内存元素数、数据元素数和行元素数
- **组宽度**：`MaxGroupWidth` 设置为 1

## 扇入/扇出类

### 1. FanAsymmetric 结构体

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

**功能**：不对称的扇入/扇出类，独立存储接收和发送数量。

**实现原理**：
- **模板参数**：`MaxRecv_` 和 `MaxSend_` 指定接收和发送的最大数量
- **成员变量**：`nr`（接收数量）和 `ns`（发送数量）
- **构造函数**：接受接收和发送数量，注释中提到应检查范围
- **访问方法**：
  - `nrecv()`：返回接收数量，如果最大接收数为 0 则返回 0
  - `nsend()`：返回发送数量，如果最大发送数为 0 则返回 0

### 2. FanSymmetric 结构体

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

**功能**：对称的扇入/扇出类，保证接收和发送数量相等。

**实现原理**：
- **模板参数**：`MaxArity` 指定最大扇入/扇出数量
- **成员变量**：只存储一个值 `n`，因为接收和发送数量相等
- **构造函数**：接受接收和发送数量，假设两者相等
- **访问方法**：`nrecv()` 和 `nsend()` 都返回相同的值 `n`
- **优化**：节省寄存器使用，减少谓词寄存器在循环展开中的使用

## 原语模板

### 1. Primitives 模板类

```cpp
template<typename T, typename RedOp, typename Fan, int Direct, typename Proto, int P2p, bool isNetOffload = false>
class Primitives;
```

**功能**：定义原语类的模板，为不同的协议提供统一接口。

**模板参数**：
- `T`：数据类型
- `RedOp`：归约操作类型
- `Fan`：扇入/扇出类
- `Direct`：直接通信标志
- `Proto`：协议类（ProtoSimple、ProtoLL、ProtoLL128）
- `P2p`：点对点通信标志
- `isNetOffload`：网络卸载标志（默认为 false）

### 2. PrimitivesWithoutDirect 结构体

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

**功能**：为不支持直接通信的原语提供默认实现。

**实现原理**：
- **CRTP 模式**：使用 `static_cast<RealPrimitives*>(this)` 实现 CRTP（奇异递归模板模式）
- **方法委托**：将直接通信方法委托给实际的原语实现
- **简化实现**：对于 `directRecvDirectSend`，直接返回（空实现）
- **强制内联**：某些方法使用 `__forceinline__` 确保内联

## 辅助函数

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

**功能**：检查中止标志，实现周期性的中止检查。

**实现原理**：
1. **缓存检查**：首先检查中止缓存，如果已经标记为中止则返回 1
2. **自旋计数**：递增自旋计数，如果未达到检查间隔则返回 0
3. **重置计数**：达到检查间隔后重置自旋计数
4. **中止标志读取**：从共享内存读取中止标志
5. **中止处理**：
   - 如果中止标志被设置，更新共享内存中的中止状态
   - 将中止值标记到缓存中
   - 返回中止状态

**参数说明**：
- `abortCache`：中止缓存引用，用于快速检查
- `abortValue`：中止值，用于位掩码操作
- `spins`：自旋计数引用，用于控制检查频率

**优化策略**：
- 避免频繁的全局内存访问
- 使用缓存机制提高检查效率
- 周期性检查避免过度开销

## 设计模式和架构

### 1. 模板特化模式
- 使用模板参数化协议行为
- 为不同协议提供特化的实现

### 2. CRTP 模式
- `PrimitivesWithoutDirect` 使用 CRTP 模式
- 实现静态多态性

### 3. 策略模式
- 协议类作为策略参数传递
- 实现算法与协议的解耦

### 4. 类型安全
- 使用 `static constexpr` 确保编译时常量
- 模板参数提供编译时类型检查

## 性能优化特性

### 1. 编译时优化
- `static constexpr` 成员允许编译时计算
- 模板特化生成优化代码

### 2. 寄存器优化
- `FanSymmetric` 节省寄存器使用
- 减少谓词寄存器在循环中的使用

### 3. 内存访问优化
- 缓存中止状态减少全局内存访问
- 周期性检查平衡性能和响应性

### 4. 内联优化
- 使用 `__forceinline__` 确保关键函数内联

## 错误处理和可靠性

### 1. 中止机制
- `checkAbort` 函数实现中止检查
- 支持快速中止传播

### 2. 边界检查
- 模板参数提供静态边界检查
- 注释中提及运行时断言

### 3. 类型安全
- 严格的模板参数约束
- 避免运行时类型错误

## 总结

`primitives.h` 文件是 NCCL 设备端通信原语的核心框架，提供了：

1. **协议抽象**：为 Simple、LL、LL128 协议提供统一接口
2. **扇入/扇出管理**：支持对称和不对称的通信模式
3. **原语模板**：通用的通信原语实现框架
4. **辅助功能**：中止检查、默认实现等
5. **性能优化**：编译时优化、寄存器优化、内存访问优化

该文件的设计充分考虑了灵活性、性能和类型安全，通过模板和策略模式实现了高效的设备端通信原语框架。