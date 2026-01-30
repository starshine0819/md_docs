# NCCL device/op128_h_functions.md - 128-bit Operations Implementation

## 文件概述

`op128.h` 是 NCCL 设备端代码的 128 位操作实现文件，位于 `/root/nccl/src/device/` 目录下。该文件定义了 128 位数据的加载、存储、地址转换和字节包操作等核心功能，是 NCCL 高效内存操作的基础组件。

## 核心函数详细分析

### 1. 128 位内存访问函数

#### 1.1 load128 函数

```cpp
inline __device__ void load128(const uint64_t* ptr, uint64_t &v0, uint64_t &v1) {
  asm volatile("ld.volatile.global.v2.u64 {%0,%1}, [%2];"
      : "=l"(v0), "=l"(v1) : "l"(ptr) : "memory");
}
```

**功能**：从全局内存加载 128 位数据（两个 64 位值）。

**实现原理**：
- **PTX 指令**：使用 `ld.volatile.global.v2.u64` 指令
- **双值加载**：同时加载两个 64 位值到 v0 和 v1
- **易失性访问**：确保内存访问的可见性和顺序性
- **向量加载**：使用向量加载指令提高效率

#### 1.2 store128 函数

```cpp
inline __device__ void store128(uint64_t* ptr, uint64_t v0, uint64_t v1) {
  asm volatile("st.volatile.global.v2.u64 [%2], {%0,%1};"
      :: "l"(v0), "l"(v1), "l"(ptr) : "memory");
}
```

**功能**：将 128 位数据存储到全局内存。

**实现原理**：
- **PTX 指令**：使用 `st.volatile.global.v2.u64` 指令
- **双值存储**：同时存储两个 64 位值
- **易失性存储**：确保内存访问的可见性

### 2. 共享内存操作函数

#### 2.1 shmemCvtPtr 函数

```cpp
inline __device__ uint64_t* shmemCvtPtr(volatile uint64_t* shmemGenericPtr) {
  uint64_t* shmemAsmPtr;
  asm volatile("cvta.to.shared.u64 %0, %1;" : "=l"(shmemAsmPtr) : "l"(shmemGenericPtr) : "memory");
  return shmemAsmPtr;
}
```

**功能**：将通用共享内存指针转换为汇编共享内存指针。

**实现原理**：
- **地址转换**：使用 `cvta.to.shared.u64` 指令进行地址空间转换
- **共享内存**：转换为共享内存地址空间
- **类型安全**：确保指针类型的安全转换

#### 2.2 loadShmem128 函数

```cpp
inline __device__ void loadShmem128(uint64_t* shmemAsmPtr, uint64_t &v0, uint64_t &v1) {
  asm volatile("ld.volatile.shared.v2.u64 {%0,%1}, [%2];"
      : "=l"(v0), "=l"(v1) : "l"(shmemAsmPtr) : "memory");
}
```

**功能**：从共享内存加载 128 位数据。

**实现原理**：
- **共享内存访问**：使用 `ld.volatile.shared.v2.u64` 指令
- **高速访问**：共享内存提供更快的访问速度
- **双值加载**：一次性加载两个 64 位值

#### 2.3 storeShmem128 函数

```cpp
inline __device__ void storeShmem128(uint64_t* shmemAsmPtr, uint64_t v0, uint64_t v1) {
  asm volatile("st.volatile.shared.v2.u64 [%2], {%0,%1};"
      :: "l"(v0), "l"(v1), "l"(shmemAsmPtr) : "memory");
}
```

**功能**：将 128 位数据存储到共享内存。

**实现原理**：
- **共享内存存储**：使用 `st.volatile.shared.v2.u64` 指令
- **高速存储**：利用共享内存的高速特性

### 3. 非对齐共享内存加载函数

#### 3.1 loadShmemMisaligned128 函数

```cpp
template<typename T>
inline __device__ void loadShmemMisaligned128(T *ptr, uint64_t &v0, uint64_t &v1) {
  union {
    uint32_t tmp4[4];
    uint64_t tmp8[2];
  };
  if(sizeof(T) < 4) {
    uint32_t *ptr4 = reinterpret_cast<uint32_t*>(reinterpret_cast<uintptr_t>(ptr) & -uintptr_t(4));
    #pragma unroll
    for(int e=0; e < 4; e++) {
      // Produce 4 bytes of sub-register type by reading 2 4-byte
      // aligned values and shifting.
      uint32_t lo, hi;
      asm volatile("ld.shared.b32 %0,[%1];" : "=r"(lo) : "l"(ptr4+e+0) : "memory");
      asm volatile("ld.shared.b32 %0,[%1];" : "=r"(hi) : "l"(ptr4+e+1) : "memory");
      tmp4[e] = __funnelshift_r(lo, hi, 8*(int(reinterpret_cast<uintptr_t>(ptr))%4));
    }
  }
  else if(sizeof(T) == 4) {
    #pragma unroll
    for(int e=0; e < 4; e++)
      asm volatile("ld.shared.b32 %0,[%1];" : "=r"(tmp4[e]) : "l"(ptr+e) : "memory");
  }
  else /*sizeof(T)==8*/ {
    #pragma unroll
    for(int e=0; e < 2; e++)
      asm volatile("ld.shared.b64 %0,[%1];" : "=l"(tmp8[e]) : "l"(ptr+e) : "memory");
  }
  v0 = tmp8[0];
  v1 = tmp8[1];
}
```

**功能**：从共享内存加载非对齐的 128 位数据。

**实现原理**：

##### 3.1.1 小于 4 字节类型处理
```cpp
if(sizeof(T) < 4) {
  uint32_t *ptr4 = reinterpret_cast<uint32_t*>(reinterpret_cast<uintptr_t>(ptr) & -uintptr_t(4));
  #pragma unroll
  for(int e=0; e < 4; e++) {
    uint32_t lo, hi;
    asm volatile("ld.shared.b32 %0,[%1];" : "=r"(lo) : "l"(ptr4+e+0) : "memory");
    asm volatile("ld.shared.b32 %0,[%1];" : "=r"(hi) : "l"(ptr4+e+1) : "memory");
    tmp4[e] = __funnelshift_r(lo, hi, 8*(int(reinterpret_cast<uintptr_t>(ptr))%4));
  }
}
```

- **对齐转换**：将指针转换为 4 字节对齐
- **双值读取**：读取相邻的两个 4 字节值
- **漏斗移位**：使用 `__funnelshift_r` 进行字节对齐

##### 3.1.2 4 字节类型处理
```cpp
else if(sizeof(T) == 4) {
  #pragma unroll
  for(int e=0; e < 4; e++)
    asm volatile("ld.shared.b32 %0,[%1];" : "=r"(tmp4[e]) : "l"(ptr+e) : "memory");
}
```

- **直接加载**：使用 32 位指令直接加载

##### 3.1.3 8 字节类型处理
```cpp
else /*sizeof(T)==8*/ {
  #pragma unroll
  for(int e=0; e < 2; e++)
    asm volatile("ld.shared.b64 %0,[%1];" : "=l"(tmp8[e]) : "l"(ptr+e) : "memory");
}
```

- **64 位加载**：使用 64 位指令加载

### 4. 地址转换函数

#### 4.1 cvta_to_shared 函数

```cpp
template<typename T>
__device__ __forceinline__ uint32_t cvta_to_shared(T* ptr) {
  return (uint32_t)__cvta_generic_to_shared(ptr);
}
```

**功能**：将通用指针转换为共享内存地址。

#### 4.2 cvta_to_global 函数

```cpp
template<typename T>
__device__ __forceinline__ uintptr_t cvta_to_global(T* ptr) {
  return (uintptr_t)__cvta_generic_to_global(ptr);
}
```

**功能**：将通用指针转换为全局内存地址。

#### 4.3 cvta_from_shared 函数

```cpp
template<typename T>
__device__ __forceinline__ T* cvta_from_shared(uint32_t shptr) {
  T* ans;
  asm("cvta.shared.u64 %0, %1;" : "=l"(ans) : "l"(uint64_t(shptr)));
  return ans;
}
```

**功能**：从共享内存地址转换为指针。

#### 4.4 cvta_from_global 函数

```cpp
template<typename T>
__device__ __forceinline__ T* cvta_from_global(uintptr_t gptr) {
  T* ans;
  asm("cvta.global.u64 %0, %1;" : "=l"(ans) : "l"(gptr));
  return ans;
}
```

**功能**：从全局内存地址转换为指针。

### 5. 字节包（BytePack）结构

#### 5.1 BytePack 模板定义

```cpp
template<int Size>
union BytePack;
template<>
union BytePack<0> {};
template<>
union BytePack<1> {
  uint8_t u8[1], native;
};
template<>
union BytePack<2> {
  BytePack<1> half[2];
  BytePack<1> b1[2];
  uint8_t u8[2];
  uint16_t u16[1], native;
};
template<>
union BytePack<4> {
  BytePack<2> half[2];
  BytePack<1> b1[4];
  BytePack<2> b2[2];
  uint8_t u8[4];
  uint16_t u16[2];
  uint32_t u32[1], native;
};
template<>
union BytePack<8> {
  BytePack<4> half[2];
  BytePack<1> b1[8];
  BytePack<2> b2[4];
  BytePack<4> b4[2];
  uint8_t u8[8];
  uint16_t u16[4];
  uint32_t u32[2];
  uint64_t u64[1], native;
};
template<>
union alignas(16) BytePack<16> {
  BytePack<8> half[2];
  BytePack<1> b1[16];
  BytePack<2> b2[8];
  BytePack<4> b4[4];
  BytePack<8> b8[2];
  uint8_t u8[16];
  uint16_t u16[8];
  uint32_t u32[4];
  uint64_t u64[2];
  ulong2 ul2[1], native;
};
```

**功能**：定义不同大小的字节包联合体。

**实现原理**：
- **层次结构**：通过递归定义构建层次结构
- **多种视图**：提供不同数据类型的访问视图
- **对齐保证**：16 字节对齐确保高效访问
- **原生访问**：提供原生数据类型访问

#### 5.2 BytePackOf 结构

```cpp
template<typename T>
struct BytePackOf {
  static constexpr int Size = sizeof(T);
  using Pack = BytePack<Size>;
};
template<>
struct BytePackOf<BytePack<0>> {
  static constexpr int Size = 0;
  using Pack = BytePack<0>;
};
```

**功能**：获取类型对应的字节包类型。

#### 5.3 类型转换函数

```cpp
template<typename T>
__device__ __forceinline__ typename BytePackOf<T>::Pack toPack(T value)  {
  union { typename BytePackOf<T>::Pack p; T v; };
  v = value;
  return p;
}

template<typename T>
__device__ __forceinline__ T fromPack(typename BytePackOf<T>::Pack pack)  {
  union { typename BytePackOf<T>::Pack p; T v; };
  p = pack;
  return v;
}
```

**功能**：在数据类型和字节包之间进行转换。

### 6. 内存加载/存储函数模板

#### 6.1 全局内存操作

```cpp
template<int Size> __device__ BytePack<Size> ld_global(uintptr_t addr);
template<int Size> __device__ BytePack<Size> ld_volatile_global(uintptr_t addr);
template<int Size> __device__ void st_global(uintptr_t addr, BytePack<Size> value);
```

**功能**：定义全局内存的加载和存储操作模板。

#### 6.2 共享内存操作

```cpp
template<int Size> __device__ BytePack<Size> ld_shared(uint32_t addr);
template<int Size> __device__ BytePack<Size> ld_volatile_shared(uint32_t addr);
template<int Size> __device__ void st_shared(uint32_t addr, BytePack<Size> value);
```

**功能**：定义共享内存的加载和存储操作模板。

#### 6.3 特化实现

```cpp
#define DEFINE_ld_st__size(bytes, data_cxx_ty, data_ptx_ty, data_reg_ty) \
  DEFINE_ld_st__size_space(bytes, data_cxx_ty, data_ptx_ty, data_reg_ty, global, uintptr_t, l) \
  DEFINE_ld_st__size_space(bytes, data_cxx_ty, data_ptx_ty, data_reg_ty, shared, uint32_t, r) \
  DEFINE_ld_st_gpu_relaxed__size(bytes, data_cxx_ty, data_ptx_ty, data_reg_ty)

DEFINE_ld_st__size(1, uint32_t, b8, r)  // 1字节使用32位寄存器
DEFINE_ld_st__size(2, uint16_t, b16, h)  // 2字节使用16位寄存器
DEFINE_ld_st__size(4, uint32_t, b32, r)  // 4字节使用32位寄存器
DEFINE_ld_st__size(8, uint64_t, b64, l)  // 8字节使用64位寄存器
```

**功能**：为不同大小的数据定义具体的加载/存储实现。

**实现原理**：
- **寄存器适配**：根据数据大小选择合适的寄存器类型
- **PTX 指令**：生成相应的 PTX 汇编指令
- **内存空间**：支持全局和共享内存空间

#### 6.4 16 字节操作

```cpp
template<>
__device__ __forceinline__ BytePack<16> ld_relaxed_gpu_global<16>(uintptr_t addr) {
  BytePack<16> ans;
  asm volatile("ld." PTX_relaxed_gpu ".global.v2.b64 {%0,%1}, [%2];" : "=l"(ans.u64[0]), "=l"(ans.u64[1]) : "l"(addr) : "memory");
  return ans;
}
```

**功能**：16 字节的全局内存加载操作。

**实现原理**：
- **双 64 位加载**：使用两个 64 位值加载 16 字节
- **松弛语义**：根据架构支持松弛内存语义

### 7. 原子内存操作函数

#### 7.1 易失性加载

```cpp
__device__ __forceinline__ uint64_t ld_volatile_global(uint64_t *ptr) {
  uint64_t ans;
  asm volatile("ld.volatile.global.u64 %0, [%1];" : "=l"(ans) : "l"(cvta_to_global(ptr)) : "memory");
  return ans;
}
```

#### 7.2 松弛系统加载

```cpp
__device__ __forceinline__ uint64_t ld_relaxed_sys_global(uint64_t *ptr) {
  uint64_t ans;
  #if __CUDA_ARCH__ >= 700
    asm volatile("ld.relaxed.sys.global.u64 %0, [%1];" : "=l"(ans) : "l"(cvta_to_global(ptr)) : "memory");
  #else
    asm volatile("ld.volatile.global.u64 %0, [%1];" : "=l"(ans) : "l"(cvta_to_global(ptr)) : "memory");
  #endif
  return ans;
}
```

#### 7.3 获取语义加载

```cpp
__device__ __forceinline__ uint64_t ld_acquire_sys_global(uint64_t *ptr) {
  uint64_t ans;
  #if __CUDA_ARCH__ >= 700
    asm volatile("ld.acquire.sys.global.u64 %0, [%1];" : "=l"(ans) : "l"(cvta_to_global(ptr)) : "memory");
  #else
    asm volatile("ld.volatile.sys.global.u64 [%1]; membar.gl;" : "=l"(ans) : "l"(cvta_to_global(ptr)) : "memory");
  #endif
  return ans;
}
```

**功能**：提供不同内存排序语义的加载操作。

#### 7.4 存储操作

```cpp
__device__ __forceinline__ void st_volatile_global(uint64_t *ptr, uint64_t val) {
  asm volatile("st.volatile.global.u64 [%0], %1;" :: "l"(cvta_to_global(ptr)), "l"(val) : "memory");
}
```

#### 7.5 内存栅栏函数

```cpp
__device__ __forceinline__ void fence_acq_rel_sys() {
  #if __CUDA_ARCH__ >= 700
    asm volatile("fence.acq_rel.sys;" ::: "memory");
  #else
    asm volatile("membar.sys;" ::: "memory");
  #endif
}
```

**功能**：提供内存栅栏操作，确保内存访问顺序。

### 8. 多内存存储函数

#### 8.1 multimem_st_global 函数

```cpp
template<int Size>
__device__ __forceinline__ void multimem_st_global(uintptr_t addr, BytePack<Size> val);

#if __CUDA_ARCH__ >= 900 && CUDART_VERSION >= 12010
template<>
__device__ __forceinline__ void multimem_st_global<16>(uintptr_t addr, BytePack<16> val) {
  asm volatile("multimem.st.global.v4.f32 [%0], {%1,%2,%3,%4};"
    :: "l"(addr), "r"(val.u32[0]), "r"(val.u32[1]), "r"(val.u32[2]), "r"(val.u32[3])
    : "memory");
}
#endif
```

**功能**：使用多内存子系统进行全局存储。

**实现原理**：
- **硬件特性**：利用支持多内存的硬件特性
- **向量存储**：使用 4 个 32 位值进行存储
- **条件编译**：仅在支持的架构上启用

### 9. 数据包操作函数

#### 9.1 loadPack 函数

```cpp
template<typename Pack, typename T>
__device__ __forceinline__ Pack loadPack(T* ptr, int ix, int end) {
  constexpr int Size = sizeof(Pack);
  ptr += ix;
  int n = end - ix;
  if (alignof(T) == Size && sizeof(T) == Size) {
    return *(Pack*)ptr;
  } else if ((Size+3)/4 + 1 < Size/sizeof(T)) {
    // 处理未对齐情况
  } else {
    // 处理一般情况
  }
}
```

**功能**：从数组中加载数据包。

#### 9.2 storePack 函数

```cpp
template<typename Pack, typename T>
__device__ __forceinline__ void storePack(T* ptr, int ix, int end, Pack val) {
  constexpr int Size = sizeof(Pack);
  union { Pack tmp; BytePack<sizeof(T)> part[Size/sizeof(T)]; };
  tmp = val;
  ptr += ix;
  int n = end - ix;
  #pragma unroll
  for (int i=0; i < Size/sizeof(T); i++) {
    if (i < 1 || i < n) ((BytePack<sizeof(T)>*)ptr)[i] = part[i];
  }
}
```

**功能**：将数据包存储到数组中。

### 10. warp级展开复制函数

#### 10.1 copyGlobalShared_WarpUnrolled 函数

```cpp
template<int EltSize, int MaxBytes, bool Multimem, typename IntBytes>
__device__ __forceinline__ void copyGlobalShared_WarpUnrolled(
    int lane, uintptr_t dstAddr, uint32_t srcAddr, IntBytes nBytesAhead
  ) {
  static_assert(std::is_signed<IntBytes>::value, "`IntBytes` must be a signed integral type.");
  int nBytes = min(nBytesAhead, (IntBytes)MaxBytes);
  int nFrontBytes = min(nBytes, (16 - int(dstAddr%16))%16);
  int nMiddleBytes = (nBytes-nFrontBytes) & -16;
  int nBackBytes = (nBytes-nFrontBytes) % 16;

  { int backLane = WARP_SIZE-1 - lane;
    bool hasFront = lane*EltSize < nFrontBytes;
    bool hasBack = backLane*EltSize < nBackBytes;
    int offset = hasFront ? lane*EltSize : (nBytes - (backLane+1)*EltSize);
    if (hasFront | hasBack) {
      BytePack<EltSize> tmp = ld_shared<EltSize>(srcAddr+offset);
      st_global<EltSize>(dstAddr+offset, tmp);
    }
  }

  // 中间部分处理...
}
```

**功能**：warp级展开的全局-共享内存复制。

**实现原理**：
- **前端处理**：处理前端未对齐部分
- **中间处理**：处理对齐的中间部分
- **后端处理**：处理后端未对齐部分
- **多内存支持**：可选择使用多内存存储

## 设计特点

### 1. 高效内存访问
- **128 位操作**：支持 128 位数据的高效访问
- **对齐优化**：优化对齐和非对齐内存访问
- **批量操作**：支持批量数据传输

### 2. 内存类型支持
- **全局内存**：支持全局内存的高效访问
- **共享内存**：支持共享内存的高速访问
- **多内存**：支持多内存系统的访问

### 3. 内存语义
- **易失性**：支持易失性内存访问
- **松弛语义**：支持松弛内存语义
- **获取/释放**：支持获取和释放语义

### 4. 类型安全
- **模板设计**：使用模板确保类型安全
- **静态断言**：使用静态断言验证类型要求
- **联合体**：使用联合体提供多种访问视图

## 性能优化特性

### 1. 循环展开
- **编译时展开**：使用 `#pragma unroll` 进行循环展开
- **减少开销**：减少循环控制开销
- **提高并行度**：提高指令级并行度

### 2. 内存对齐
- **16 字节对齐**：确保 16 字节对齐访问
- **对齐检测**：检测和处理对齐情况
- **优化路径**：为对齐情况提供优化路径

### 3. 寄存器优化
- **合适寄存器**：为不同数据大小选择合适的寄存器
- **向量操作**：使用向量寄存器进行批量操作
- **高效转换**：优化寄存器间的数据转换

### 4. 架构适配
- **CUDA 架构**：根据 CUDA 架构版本选择最优实现
- **硬件特性**：利用特定硬件特性进行优化
- **版本检测**：使用版本宏进行条件编译

## 应用场景

### 1. 高性能通信
- **批量传输**：支持大批量数据的高效传输
- **对齐优化**：优化对齐数据的传输性能
- **内存带宽**：充分利用内存带宽

### 2. 数据处理
- **向量化操作**：支持向量化的数据处理
- **并行访问**：支持并行内存访问
- **类型转换**：高效的数据类型转换

### 3. 同步操作
- **原子访问**：支持原子内存访问
- **内存栅栏**：提供内存同步机制
- **一致性**：确保内存访问的一致性

## 错误处理和可靠性

### 1. 类型安全
- **模板约束**：确保模板参数的有效性
- **静态检查**：使用静态断言进行编译时检查
- **类型转换**：安全的类型转换机制

### 2. 内存安全
- **边界检查**：在某些操作中进行边界检查
- **指针验证**：验证指针的有效性
- **访问对齐**：确保内存访问对齐

### 3. 架构兼容
- **版本检测**：检测 CUDA 架构版本
- **功能检测**：检测硬件功能支持
- **降级实现**：提供降级实现方案

## 总结

`op128.h` 文件实现了 NCCL 设备端的 128 位操作核心功能，提供了：

1. **高效内存访问**：支持 128 位数据的高效加载和存储
2. **多内存类型**：支持全局、共享和多内存系统的访问
3. **内存语义**：提供多种内存访问语义
4. **类型安全**：使用模板和联合体确保类型安全
5. **性能优化**：循环展开、对齐优化、寄存器优化
6. **架构适配**：根据 CUDA 架构版本优化实现
7. **错误处理**：全面的类型安全和架构兼容性检查

该文件是 NCCL 高性能内存操作的基础，为上层通信原语提供了高效的内存访问能力。