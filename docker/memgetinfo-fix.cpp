#if 0
/opt/rocm/llvm/bin/clang++ \
    -std=c++20 \
    -I${ROCM_PATH}/include \
    -D__HIP_PLATFORM_AMD__ \
    -fPIC -shared -g -O3 \
    -L${ROCM_PATH}/lib \
    -lamdhip64 \
    -o libpreload-me.so preload-me.cpp
exit 0
#endif

#include <stdio.h>
#include <dlfcn.h>
#include <cassert>
#include <string>
#include <fstream>

#define hipError_t int
#define hipSuccess 0
#define hipErrorNotSupported 801

typedef struct {
    // 32-bit Atomics
    unsigned hasGlobalInt32Atomics : 1;     ///< 32-bit integer atomics for global memory.
    unsigned hasGlobalFloatAtomicExch : 1;  ///< 32-bit float atomic exch for global memory.
    unsigned hasSharedInt32Atomics : 1;     ///< 32-bit integer atomics for shared memory.
    unsigned hasSharedFloatAtomicExch : 1;  ///< 32-bit float atomic exch for shared memory.
    unsigned hasFloatAtomicAdd : 1;  ///< 32-bit float atomic add in global and shared memory.

    // 64-bit Atomics
    unsigned hasGlobalInt64Atomics : 1;  ///< 64-bit integer atomics for global memory.
    unsigned hasSharedInt64Atomics : 1;  ///< 64-bit integer atomics for shared memory.

    // Doubles
    unsigned hasDoubles : 1;  ///< Double-precision floating point.

    // Warp cross-lane operations
    unsigned hasWarpVote : 1;     ///< Warp vote instructions (__any, __all).
    unsigned hasWarpBallot : 1;   ///< Warp ballot instructions (__ballot).
    unsigned hasWarpShuffle : 1;  ///< Warp shuffle operations. (__shfl_*).
    unsigned hasFunnelShift : 1;  ///< Funnel two words into one with shift&mask caps.

    // Sync
    unsigned hasThreadFenceSystem : 1;  ///< __threadfence_system.
    unsigned hasSyncThreadsExt : 1;     ///< __syncthreads_count, syncthreads_and, syncthreads_or.

    // Misc
    unsigned hasSurfaceFuncs : 1;        ///< Surface functions.
    unsigned has3dGrid : 1;              ///< Grid and group dims are 3D (rather than 2D).
    unsigned hasDynamicParallelism : 1;  ///< Dynamic parallelism.
} hipDeviceArch_t;

typedef struct hipDeviceProp_t {
    char name[256];            ///< Device name.
    size_t totalGlobalMem;     ///< Size of global memory region (in bytes).
    size_t sharedMemPerBlock;  ///< Size of shared memory region (in bytes).
    int regsPerBlock;          ///< Registers per block.
    int warpSize;              ///< Warp size.
    int maxThreadsPerBlock;    ///< Max work items per work group or workgroup max size.
    int maxThreadsDim[3];      ///< Max number of threads in each dimension (XYZ) of a block.
    int maxGridSize[3];        ///< Max grid dimensions (XYZ).
    int clockRate;             ///< Max clock frequency of the multiProcessors in khz.
    int memoryClockRate;       ///< Max global memory clock frequency in khz.
    int memoryBusWidth;        ///< Global memory bus width in bits.
    size_t totalConstMem;      ///< Size of shared memory region (in bytes).
    int major;  ///< Major compute capability.  On HCC, this is an approximation and features may
                ///< differ from CUDA CC.  See the arch feature flags for portable ways to query
                ///< feature caps.
    int minor;  ///< Minor compute capability.  On HCC, this is an approximation and features may
                ///< differ from CUDA CC.  See the arch feature flags for portable ways to query
                ///< feature caps.
    int multiProcessorCount;          ///< Number of multi-processors (compute units).
    int l2CacheSize;                  ///< L2 cache size.
    int maxThreadsPerMultiProcessor;  ///< Maximum resident threads per multi-processor.
    int computeMode;                  ///< Compute mode.
    int clockInstructionRate;  ///< Frequency in khz of the timer used by the device-side "clock*"
                               ///< instructions.  New for HIP.
    hipDeviceArch_t arch;      ///< Architectural feature flags.  New for HIP.
    int concurrentKernels;     ///< Device can possibly execute multiple kernels concurrently.
    int pciDomainID;           ///< PCI Domain ID
    int pciBusID;              ///< PCI Bus ID.
    int pciDeviceID;           ///< PCI Device ID.
    size_t maxSharedMemoryPerMultiProcessor;  ///< Maximum Shared Memory Per Multiprocessor.
    int isMultiGpuBoard;                      ///< 1 if device is on a multi-GPU board, 0 if not.
    int canMapHostMemory;                     ///< Check whether HIP can map host memory
    int gcnArch;                              ///< DEPRECATED: use gcnArchName instead
    char gcnArchName[256];                    ///< AMD GCN Arch Name.
    int integrated;            ///< APU vs dGPU
    int cooperativeLaunch;            ///< HIP device supports cooperative launch
    int cooperativeMultiDeviceLaunch; ///< HIP device supports cooperative launch on multiple devices
    int maxTexture1DLinear;    ///< Maximum size for 1D textures bound to linear memory
    int maxTexture1D;          ///< Maximum number of elements in 1D images
        int maxTexture2D[2];       ///< Maximum dimensions (width, height) of 2D images, in image elements
    int maxTexture3D[3];       ///< Maximum dimensions (width, height, depth) of 3D images, in image elements
    unsigned int* hdpMemFlushCntl;      ///< Addres of HDP_MEM_COHERENCY_FLUSH_CNTL register
    unsigned int* hdpRegFlushCntl;      ///< Addres of HDP_REG_COHERENCY_FLUSH_CNTL register
    size_t memPitch;                 ///<Maximum pitch in bytes allowed by memory copies
    size_t textureAlignment;         ///<Alignment requirement for textures
    size_t texturePitchAlignment;    ///<Pitch alignment requirement for texture references bound to pitched memory
    int kernelExecTimeoutEnabled;    ///<Run time limit for kernels executed on the device
    int ECCEnabled;                  ///<Device has ECC support enabled
    int tccDriver;                   ///< 1:If device is Tesla device using TCC driver, else 0
    int cooperativeMultiDeviceUnmatchedFunc;        ///< HIP device supports cooperative launch on multiple
                                                    ///devices with unmatched functions
    int cooperativeMultiDeviceUnmatchedGridDim;     ///< HIP device supports cooperative launch on multiple
                                                    ///devices with unmatched grid dimensions
    int cooperativeMultiDeviceUnmatchedBlockDim;    ///< HIP device supports cooperative launch on multiple
                                                    ///devices with unmatched block dimensions
    int cooperativeMultiDeviceUnmatchedSharedMem;   ///< HIP device supports cooperative launch on multiple
                                                    ///devices with unmatched shared memories
    int isLargeBar;                  ///< 1: if it is a large PCI bar device, else 0
    int asicRevision;                ///< Revision of the GPU in this device
    int managedMemory;               ///< Device supports allocating managed memory on this system
    int directManagedMemAccessFromHost; ///< Host can directly access managed memory on the device without migration
    int concurrentManagedAccess;     ///< Device can coherently access managed memory concurrently with the CPU
    int pageableMemoryAccess;        ///< Device supports coherently accessing pageable memory
                                     ///< without calling hipHostRegister on it
    int pageableMemoryAccessUsesHostPageTables; ///< Device accesses pageable memory via the host's page tables
} hipDeviceProp_t;

extern "C" {
    hipError_t hipGetDevice (int* device);
    hipError_t hipGetDeviceCount (int* count);
    hipError_t hipGetDeviceProperties (hipDeviceProp_t* p, int device);
}

namespace {


// Initializes the symbol of the original runtime symbol and return 0 if success
template<typename T>
int lazy_init(T *&fptr, const char *name) {
    void *&ptr = reinterpret_cast<void *&>(fptr);

    if (ptr) return 0;

    ptr = dlsym(RTLD_NEXT, name);

    assert(ptr);

    return ptr ? 0 : -1;
}

hipError_t (*hipMemGetInfo_orig)(size_t *, size_t *) = nullptr;

}

extern "C" {
hipError_t 	hipMemGetInfo (size_t* free, size_t* total){

  if(lazy_init(hipMemGetInfo_orig, "hipMemGetInfo")) return hipErrorNotSupported;

  hipError_t ret;
  ret = hipMemGetInfo_orig(free, total);
  if(ret) return ret;

  int device;
  ret = hipGetDevice(&device);
  if(ret) return ret;

  hipDeviceProp_t p;
  ret = hipGetDeviceProperties(&p, device);
  if(ret) return ret;

  const int logical_device_offset = 4;
  int logical_device = -1;
  switch(p.pciBusID) {
    case 0xc1: 
      logical_device = 0 + logical_device_offset;
      break;
    case 0xc6: 
      logical_device = 1 + logical_device_offset;
      break;
    case 0xc9: 
      logical_device = 2 + logical_device_offset;
      break;
    case 0xce: 
      logical_device = 3 + logical_device_offset;
      break;
    case 0xd1: 
      logical_device = 4 + logical_device_offset;
      break;
    case 0xd6: 
      logical_device = 5 + logical_device_offset;
      break;
    case 0xd9: 
      logical_device = 6 + logical_device_offset;
      break;
    case 0xde: 
      logical_device = 7 + logical_device_offset;
      break;
    default: 
      return hipErrorNotSupported;
  }
  
  // Read the data from the KFD.
  std::string fileName = std::string("/sys/class/kfd/kfd/topology/nodes/") + std::to_string(logical_device) + std::string("/mem_banks/0/used_memory");  
  
	std::ifstream file;
	file.open(fileName);
	if (!file) return hipErrorNotSupported;

  std::string deviceSize;	
	size_t deviceMemSize;

	file >> deviceSize;
	file.close();         
	        
  if ((deviceMemSize=strtol(deviceSize.c_str(),NULL,10)))
	  *free = *total - deviceMemSize;
	else
    return hipErrorNotSupported;
	
  return hipSuccess;
}

} // extern "C"