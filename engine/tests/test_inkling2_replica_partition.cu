// Standalone grouping/scatter test; no weights. Run once on each compute device.
// nvcc -std=c++17 engine/tests/test_inkling2_replica_partition.cu -o replica_partition.exe
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cstring>
#include "../src/inkling2/inkling2_kernels.cuh"
using namespace strata::inkling2;
#define CHECK(x) do { auto e=(x); if(e!=cudaSuccess) { std::fprintf(stderr,"%s\n",cudaGetErrorString(e)); std::exit(2); } } while(0)
template<class T> struct Buf {
    T* p=nullptr; explicit Buf(size_t n) { CHECK(cudaMalloc(&p,n*sizeof(T))); }
    ~Buf(){ cudaFree(p); }
};
__global__ void scatter_grouped(const int* starts,const int* ng,const int* dst,const int* tok,
                               int* coverage,float* parts,int entries) {
    for(int i=threadIdx.x;i<entries;i+=blockDim.x) if(i<starts[*ng]) {
        const int entry=dst[i];
        atomicAdd(&coverage[entry],1);
        // Value depends on original token and top-k entry, independent of grouping order.
        parts[entry]=float(tok[i]*100 + entry%8);
    }
}
int main(int argc,char**argv) {
    CHECK(cudaSetDevice(argc>1?std::atoi(argv[1]):0));
    constexpr int ne=128;
    for(int T : {1,2,17,64}) for(int pattern=0;pattern<4;++pattern) {
        const int entries=T*8;
        std::vector<int> ids(entries), local(ne,-1), cold(ne), coverage(entries);
        std::vector<float> expected(entries), actual(entries);
        for(int e=0;e<ne;++e) {
            bool hot=pattern==1 || (pattern==2 && e%2==0) || (pattern==3 && e>=120);
            local[e]=hot?e:-1; cold[e]=hot?-1:e;
        }
        for(int i=0;i<entries;++i) { ids[i]=(i*37+123)%ne; expected[i]=float((i/8)*100+i%8); }
        Buf<int> dids(entries), map(ne), starts(ne+1), ng(1), dst(entries), tok(entries), count(entries);
        Buf<unsigned long long> ptrs(ne); Buf<unsigned char> blobs(ne); Buf<float> parts(entries);
        CHECK(cudaMemcpy(dids.p,ids.data(),entries*sizeof(int),cudaMemcpyHostToDevice));
        CHECK(cudaMemset(count.p,0,entries*sizeof(int)));
        CHECK(cudaMemset(parts.p,0xff,entries*sizeof(float)));
        for(auto* owner : {&local,&cold}) {
            CHECK(cudaMemcpy(map.p,owner->data(),ne*sizeof(int),cudaMemcpyHostToDevice));
            k_group_experts<<<1,128>>>(dids.p,map.p,blobs.p,1,entries,ne,ne,
                ptrs.p,starts.p,ng.p,dst.p,tok.p,8);
            scatter_grouped<<<1,128>>>(starts.p,ng.p,dst.p,tok.p,count.p,parts.p,entries);
            CHECK(cudaGetLastError());
        }
        CHECK(cudaMemcpy(coverage.data(),count.p,entries*sizeof(int),cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(actual.data(),parts.p,entries*sizeof(float),cudaMemcpyDeviceToHost));
        for(int i=0;i<entries;++i) if(coverage[i]!=1 || actual[i]!=expected[i]) {
            std::fprintf(stderr,"FAIL T=%d pattern=%d entry=%d owners=%d actual=%f expected=%f\n",T,pattern,i,coverage[i],actual[i],expected[i]); return 1;
        }
    }
    std::puts("PASS replica grouping ownership and original-entry scatter");
}
