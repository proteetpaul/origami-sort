/*--------------------------------------------------------------------------------------------
 - Origami: A High-Performance Mergesort Framework											 -
 - Copyright(C) 2021 Arif Arman, Dmitri Loguinov											 -
 - Produced via research carried out by the Texas A&M Internet Research Lab                  -
 -                                                                                           -
 - This program is free software : you can redistribute it and/or modify                     -
 - it under the terms of the GNU General Public License as published by                      -
 - the Free Software Foundation, either version 3 of the License, or                         -
 - (at your option) any later version.                                                       -
 -                                                                                           -
 - This program is distributed in the hope that it will be useful,                           -
 - but WITHOUT ANY WARRANTY; without even the implied warranty of                            -
 - MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.See the                               -
 - GNU General Public License for more details.                                              -
 -                                                                                           -
 - You should have received a copy of the GNU General Public License                         -
 - along with this program. If not, see < http://www.gnu.org/licenses/>.                     -
 --------------------------------------------------------------------------------------------*/

#pragma once

#include <stdio.h>
#include <time.h>
#include <math.h>
#include <algorithm>
#include <random>       // std::default_random_engine
#include <chrono>       // std::chrono::system_clock
#include <assert.h>
#include <cstdint>
#include <iostream>
#include <fstream>
#include <thread>
#include <omp.h>
#include <queue>
#include <string.h>
#include <stdlib.h>

// Platform-specific includes
#ifdef _WIN32
    #include <Windows.h>
    #include <intrin.h>		// MMX
#else
    #include <sys/mman.h>
    #include <unistd.h>
    #include <x86intrin.h>	// MMX and intrinsics for Linux
#endif

// SIMD headers (common across platforms)
#include <xmmintrin.h>	// SSE
#include <emmintrin.h>	// SSE2
#include <smmintrin.h>	// SSE4
#include <immintrin.h>	// AVX/AVX2/AVX-512 (includes zmmintrin.h when AVX-512 flags are enabled)

using namespace std::chrono;
using hrc = std::chrono::high_resolution_clock;

// typedef key
typedef int64_t  i64;
typedef uint64_t ui64;
typedef uint32_t ui;

// Compatibility typedefs for Windows types
#ifdef _WIN32
    // Windows types are already defined
#else
    typedef int64_t __int64;
    // Windows threading types - these would need pthread equivalents for full functionality
    typedef void* HANDLE;
    struct CRITICAL_SECTION { void* dummy; };
    struct SYNCHRONIZATION_BARRIER { void* dummy; };
    #define INFINITE 0xFFFFFFFF
    #define WAIT_OBJECT_0 0
    #define WAIT_TIMEOUT 0x102
    #define MEM_COMMIT 0x1000
    #define MEM_RELEASE 0x8000
    #define PAGE_READWRITE 0x04
    // Stub functions - these would need proper pthread implementations
    #define GetCurrentThread() ((HANDLE)0)
    #define SetThreadAffinityMask(h, m) (0)
    #define CreateEvent(a, b, c, d) ((HANDLE)0)
    #define SetEvent(h) (0)
    #define WaitForSingleObject(h, t) (0)
    #define CloseHandle(h) (0)
    #define InitializeCriticalSection(cs) ((void)0)
    #define EnterCriticalSection(cs) ((void)0)
    #define LeaveCriticalSection(cs) ((void)0)
    #define DeleteCriticalSection(cs) ((void)0)
    #define InitializeSynchronizationBarrier(b, n, t) (0)
    #define EnterSynchronizationBarrier(b, f) (0)
    #define DeleteSynchronizationBarrier(b) ((void)0)
    #define InterlockedIncrement64(p) (++(*(p)))
    #define SwitchToThread() ((void)0)
#endif

#pragma pack(push, 1)
template <typename Keytype, typename Valuetype>
struct KeyValue {
	Keytype key;
	Valuetype value;
	// NOTE: operator overloads not used in origami for KV pair as it is slow; used mainly for correctness checking with std::sort 
	bool operator <(const KeyValue& kv) const {
		return key < kv.key;
	}
	bool operator >(const KeyValue& kv) const {
		return key > kv.key;
	}
	bool operator !=(const KeyValue& kv) const {
		return key != kv.key;
	}
	bool operator <=(const KeyValue& kv) const {
		return key <= kv.key;
	}
};
#pragma pack(pop)
template struct KeyValue<ui, ui>;
template struct KeyValue<i64, i64>;

#define MAX_THREADS	64

// typedef simd stuff
typedef __m128i sse;
typedef __m128	ssef;
typedef __m128d	ssed;
typedef __m256i avx2;
typedef __m256	avx2f;
typedef __m256d avx2d;
typedef __m512i avx512;
typedef __m512	avx512f;
typedef __m512d avx512d;

// Use std::min/max to avoid conflicts, but keep macros for compatibility
#ifndef _WIN32
    #include <algorithm>
    #define MIN(x, y) std::min(x, y)
    #define MAX(x, y) std::max(x, y)
#else
    #define MIN(x, y) ((x)<(y)?(x):(y))
    #define MAX(x, y) ((x)<(y)?(y):(x))
#endif 
#define FOR(i,n,k)				for (ui64 (i) = 0; (i) < (n); (i)+=(k)) 
#define FOR_INIT(i, init, n, k)	for (ui64 (i) = (init); (i) < (n); (i) += (k)) 
#define PRINT_ARR(arr, n)		{ FOR((i), (n), 1) printf("%lu ", (arr)[(i)]); printf("\n"); }
#define PRINT_ARR64(arr, n)		{ FOR((i), (n), 1) printf("%llX ", ((ui64*)arr)[(i)]); printf("\n"); }
#define PRINT_DASH(n)			{ FOR(i, (n), 1) printf("-"); printf("\n"); }
#define ELAPSED(st, en)			( duration_cast<duration<double>>(en - st).count() )
#define ELAPSED_MS(st, en)		( duration_cast<duration<double, std::milli>>(en - st).count() )
#ifdef _WIN32
    #define NOINLINE			__declspec(noinline)
    #define FORCEINLINE			__forceinline
#else
    #define NOINLINE			__attribute__((noinline))
    #define FORCEINLINE			__attribute__((always_inline)) inline
#endif
#define KB(x)					(x << 10)
#define MB(x)					(x << 20)
#define GB(x)					(x << 30)
#define HERE(x)					printf("Here %3lu\n", (x));
#define MAX_PATH_LEN			512
#define MAX_PRINTOUT			1024
#ifdef _WIN32
    #define PRINT(fmt, ...)			{ char buf_PRINT[MAX_PRINTOUT] = "%s: "; strcat_s(buf_PRINT, MAX_PRINTOUT, fmt); printf (buf_PRINT, __FUNCTION__, ##__VA_ARGS__); }
    #define ReportError(fmt, ...)	{ PRINT(fmt, ##__VA_ARGS__); getchar(); exit(-1); }
#else
    #define PRINT(fmt, ...)			{ char buf_PRINT[MAX_PRINTOUT]; snprintf(buf_PRINT, MAX_PRINTOUT, "%s: " fmt, __FUNCTION__, ##__VA_ARGS__); printf("%s", buf_PRINT); }
    #define ReportError(fmt, ...)	{ PRINT(fmt, ##__VA_ARGS__); exit(-1); }
#endif

#define LOAD(rg, ptr)			{ rg = *(ptr); }
#define STORE(rg, ptr)			{ *(ptr) = rg; }
#ifdef _WIN32
    #define VALLOC(sz)				(VirtualAlloc(NULL, (sz), MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE))
    #define VFREE(ptr)				(VirtualFree((ptr), 0, MEM_RELEASE))
#else
    // For Linux, use aligned_alloc for large aligned allocations (similar behavior to VirtualAlloc)
    // Note: VirtualAlloc provides page-aligned memory, aligned_alloc does the same
    #include <malloc.h>
    static inline void* linux_valloc(size_t sz) {
        // Round up to page size for alignment (typically 4096 bytes)
        const size_t page_size = 4096;
        size_t aligned_sz = (sz + page_size - 1) & ~(page_size - 1);
        void* ptr = aligned_alloc(page_size, aligned_sz);
        return ptr;
    }
    #define VALLOC(sz)				linux_valloc(sz)
    #define VFREE(ptr)				{ if(ptr) free(ptr); }
#endif

#define SWAP SWAPv2
#define SWAP2 SWAPv3
#define SWAPKV(x, y) {\
	{\
		const bool first = a##x.key < a##y.key;\
		const Item vTmp = first ? a##x : a##y;\
		a##y = first ? a##y : a##x;\
		a##x = vTmp;\
	}\
}
#define SWAPKV2(x,y) {\
    const bool bLess = a##x < a##y; \
    const int64_t tmp = bLess ? a##x : a##y; \
    a##y = bLess ? a##y : a##x; \
	a##x = tmp; \
    int64_t tmp2 = bLess ? b##x : b##y; \
    b##y = bLess ? b##y : b##x; \
    b##x = tmp2; \
}

const int rol_const = _MM_SHUFFLE(0, 3, 2, 1);
const int ror_const = _MM_SHUFFLE(2, 1, 0, 3);
const int shuff_32_const = _MM_SHUFFLE(2, 3, 0, 1);			// 32 bit shuffle within lane
const int shuff_64_const = _MM_SHUFFLE(1, 0, 3, 2);

#define SWAPv2(x,y) {\
	{\
		const Key tmp = MIN(a##x, a##y); \
		a##y = MAX(a##x, a##y); \
		a##x = tmp; \
	}\
}
#define SWAPv3(x,y) {\
	{\
		const Item tmp = MIN(a##x, a##y); \
		a##y = MAX(a##x, a##y); \
		a##x = tmp; \
	}\
}
#include "config.h"
