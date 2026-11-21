// Single portable pinned slot, event generations reused with host enqueue backlog.
// nvcc -std=c++17 --cudart shared ... test_inkling2_replica_relay.cu -o relay.exe
// relay.exe PRIMARY_DEVICE REMOTE_DEVICE; no model, no P2P, no CPU wait per iteration.
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "../src/inkling2/inkling2_replica_async.cuh"
#define CK(x) do {auto e=(x);if(e!=cudaSuccess){std::fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));std::exit(2);}}while(0)
constexpr int N=2304, top=8, maxT=64, iterations=40;
__host__ __device__ uint32_t image(int iter,int index) { return 0x7fc00000u ^ (uint32_t(iter+1)*0x9e3779b9u) ^ (uint32_t(index)*0x85ebca6bu); }
__global__ void produce(uint32_t* p,int words,int entries,int iter) {
    for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<words;i+=blockDim.x*gridDim.x)
        p[i]=i<entries ? uint32_t((i*37+iter)%128) : image(iter,i);
}
__global__ void remote_compute(const uint32_t* input,uint32_t* output,int entries,int qwords) {
    for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<entries*N;i+=blockDim.x*gridDim.x) {
        const int entry=i/N, col=i%N;
        // Input is opaque Q8_1 words (including scale/sum bits), copied unchanged.
        // Each return word encodes its original top-k entry and column.
        output[i]=input[entries+(entry/top*(N/32)*9+col)%qwords] ^ input[entry] ^ uint32_t(i*17);
    }
}
static void test_route_stats() {
    int *ids=nullptr,*slots=nullptr,*miss=nullptr;
    unsigned long long *freq=nullptr,*hits=nullptr;
    const int source[8]={0,1,2,3,127,128,-1,1};
    int map[128],actualMiss[8]; for(int e=0;e<128;++e) map[e]=e%2?-1:e;
    CK(cudaMalloc(&ids,sizeof(source)));CK(cudaMalloc(&slots,sizeof(map)));CK(cudaMalloc(&miss,sizeof(source)));
    CK(cudaMalloc(&freq,128*8));CK(cudaMalloc(&hits,3*8));CK(cudaMemset(freq,0,128*8));
    CK(cudaMemcpy(ids,source,sizeof(source),cudaMemcpyHostToDevice));CK(cudaMemcpy(slots,map,sizeof(map),cudaMemcpyHostToDevice));
    for(int request=1;request<=2;++request) {
        CK(cudaMemset(hits,0,3*8)); // Request reset must preserve lifetime expert frequencies.
        strata::inkling2::k_replica_route_stats<<<1,32>>>(ids,slots,miss,freq,hits,8,128);
        unsigned long long f[128],h[3];
        CK(cudaMemcpy(f,freq,sizeof(f),cudaMemcpyDeviceToHost));CK(cudaMemcpy(h,hits,sizeof(h),cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(actualMiss,miss,sizeof(actualMiss),cudaMemcpyDeviceToHost));
        if(h[0]!=2 || h[1]!=4 || h[2]!=2) { std::fprintf(stderr,"FAIL route hit counters\n");std::exit(1); }
        for(int e=0;e<128;++e) {
            const int expected=(e==1?2:(e==0||e==2||e==3||e==127?1:0))*request;
            if(f[e]!=(unsigned long long)expected) {std::fprintf(stderr,"FAIL lifetime frequency\n");std::exit(1);}
        }
        for(int i=0;i<8;++i) if(actualMiss[i]!=(source[i]<0||source[i]>=128||source[i]%2!=0)) {
            std::fprintf(stderr,"FAIL route miss mask\n");std::exit(1);
        }
    }
    CK(cudaFree(ids));CK(cudaFree(slots));CK(cudaFree(miss));CK(cudaFree(freq));CK(cudaFree(hits));
}
int main(int argc,char**argv) {
    if(argc!=3) return 2;
    const int primary=std::atoi(argv[1]),remote=std::atoi(argv[2]);
    if(primary==remote) return 2;
    constexpr int maxEntries=maxT*top,maxQWords=maxT*(N/32)*9;
    constexpr size_t maxInput=maxEntries+maxQWords,maxOutput=size_t(maxEntries)*N;
    uint32_t *input=nullptr,*returned=nullptr,*archive=nullptr,*remoteInput=nullptr,*remoteOutput=nullptr;
    uint32_t *hostInput=nullptr,*hostOutput=nullptr;
    cudaStream_t ps,rs;cudaEvent_t ready,done;
    CK(cudaSetDevice(primary));test_route_stats();CK(cudaStreamCreateWithFlags(&ps,cudaStreamNonBlocking));
    CK(cudaEventCreateWithFlags(&ready,cudaEventDisableTiming));
    CK(cudaMalloc(&input,maxInput*4));CK(cudaMalloc(&returned,maxOutput*4));CK(cudaMalloc(&archive,iterations*maxOutput*4));
    CK(cudaHostAlloc(&hostInput,maxInput*4,cudaHostAllocPortable));CK(cudaHostAlloc(&hostOutput,maxOutput*4,cudaHostAllocPortable));
    CK(cudaSetDevice(remote));CK(cudaStreamCreateWithFlags(&rs,cudaStreamNonBlocking));
    CK(cudaEventCreateWithFlags(&done,cudaEventDisableTiming));
    CK(cudaMalloc(&remoteInput,maxInput*4));CK(cudaMalloc(&remoteOutput,maxOutput*4));
    // Vary T across generations and return to large sizes, exposing stale tails/overwrites.
    const int ts[]={1,64,2,17,63,1,8,32};
    for(int iter=0;iter<iterations;++iter) {
        const int T=ts[iter%8],entries=T*top,qwords=T*(N/32)*9,words=entries+qwords;
        CK(cudaSetDevice(primary));
        produce<<<128,256,0,ps>>>(input,words,entries,iter);
        CK(cudaMemcpyAsync(hostInput,input,words*4,cudaMemcpyDeviceToHost,ps));
        CK(cudaEventRecord(ready,ps));
        CK(cudaSetDevice(remote));
        CK(cudaStreamWaitEvent(rs,ready,0));
        CK(cudaMemcpyAsync(remoteInput,hostInput,words*4,cudaMemcpyHostToDevice,rs));
        remote_compute<<<128,256,0,rs>>>(remoteInput,remoteOutput,entries,qwords);
        CK(cudaMemcpyAsync(hostOutput,remoteOutput,size_t(entries)*N*4,cudaMemcpyDeviceToHost,rs));
        CK(cudaEventRecord(done,rs));
        CK(cudaSetDevice(primary));
        CK(cudaStreamWaitEvent(ps,done,0));
        CK(cudaMemcpyAsync(returned,hostOutput,size_t(entries)*N*4,cudaMemcpyHostToDevice,ps));
        CK(cudaMemcpyAsync(archive+iter*maxOutput,returned,size_t(entries)*N*4,cudaMemcpyDeviceToDevice,ps));
        CK(cudaGetLastError());
    }
    // The only CPU completion wait follows all 40 enqueued generations.
    CK(cudaSetDevice(primary));CK(cudaStreamSynchronize(ps));
    std::vector<uint32_t> actual(maxOutput);
    for(int iter=0;iter<iterations;++iter) {
        const int T=ts[iter%8],entries=T*top,qwords=T*(N/32)*9;
        CK(cudaMemcpy(actual.data(),archive+iter*maxOutput,size_t(entries)*N*4,cudaMemcpyDeviceToHost));
        for(int i=0;i<entries*N;++i) {
            const int entry=i/N,col=i%N;
            const int index=entries+(entry/top*(N/32)*9+col)%qwords;
            const uint32_t expected=image(iter,index)^uint32_t((entry*37+iter)%128)^uint32_t(i*17);
            if(actual[i]!=expected) {std::fprintf(stderr,"FAIL generation=%d T=%d entry=%d column=%d\n",iter,T,entry,col);return 1;}
        }
    }
    CK(cudaFree(input));CK(cudaFree(returned));CK(cudaFree(archive));CK(cudaEventDestroy(ready));CK(cudaStreamDestroy(ps));
    CK(cudaSetDevice(remote));CK(cudaFree(remoteInput));CK(cudaFree(remoteOutput));CK(cudaEventDestroy(done));CK(cudaStreamDestroy(rs));
    CK(cudaFreeHost(hostInput));CK(cudaFreeHost(hostOutput));
    std::puts("PASS all 40 relay generations: Q8_1 opaque bits, ordered top-8 output, variable T, single pinned slot");
}
