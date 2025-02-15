#include <cstdio>
#include <cuda.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// 每个block 有 256个线程
#define THREAD_PER_BLOCK 256

// 归约操作:
// 将一个大数组分割成多个块（blocks），每个块由多个线程（threads）处理。
// 每个块内的线程协作计算该块的部分和。
// 最终将所有块的部分和汇总到输出数组中


__global__ void reduce(float *d_input, float *d_output){
   // 这个是设置一个偏移量，保证每个线程在block中都是一个索引
   // eg: 0 = 0 + 256*idx  idx表示是第几个block就是block的索引  索引乘以block的长度
   printf("进入核函数========");
   float *input_begin = d_input + blockDim.x*blockIdx.x;
   printf("进入核函数========");
   for (int i=1; i<blockDim.x; i*2 ){

        //只有偶数列才进行相邻元素相加
        if(threadIdx.x%(i*2)==0){
            input_begin[threadIdx.x] += input_begin[threadIdx.x +1];
        }
        // 每一层计算结束了再进行下一层相加
        __syncthreads();
   }

   if(threadIdx.x ==0){
    // 因为都是加在第一个元素上的
    d_output[blockIdx.x] = input_begin[0];
   }

}

bool check(float *out, float *res, int n){
    for(int i=0;i<n;i++){
        if (abs(out[i] - res[i]) >0.005)
            return false;
    }
    return true;
}

int main()
{
    // printf("Hello reduce");    
    const int N = 32*1024*1024;
    // CPU分配内存
    float *input = (float *)malloc(N*sizeof(float));
    float *d_input;
    cudaMalloc((void **)&d_input, N*sizeof(float));
    

    // 输出数组的每个元素是输入数组中对应块的部分和
    // THREAD_PER_BLOCK 表示每个线程快处理256个元素
    int block_num = N / THREAD_PER_BLOCK;
    float *output = (float *)malloc((N / THREAD_PER_BLOCK) * sizeof(float));
    float *d_output;
    cudaMalloc((void **)&d_output, (N/THREAD_PER_BLOCK)*sizeof(float));
    float *result = (float *) malloc((N / THREAD_PER_BLOCK) *sizeof(float));
    
    // input 数组进行赋值
    for(int i=0;i<N;i++){
        input[i]=2.0 * (float)drand48() - 1.0;
    }

    // CPU calculate
    /**
     * 算出每个block里面的值，然后进行求和 
     * 在 CPU 上计算每个块的预期结果，并存储在 res 中
    **/
    for(int i=0; i<block_num; i++){
        float cur = 0;
        for (int j = 0;j< THREAD_PER_BLOCK; j++){
            cur += input[i * THREAD_PER_BLOCK +j];
        }
        result[i] = cur;
    }

    // 将数据复制到设备
    /**
     *  将输入数组input复制到设备内存d_input
     * 
    */
    cudaMemcpy(d_input,input,N*sizeof(float), cudaMemcpyHostToDevice);

    // 启动核函数
    // Setting BLOCK and GRID dim
    dim3 Grid(N/THREAD_PER_BLOCK,1);
    dim3 Block(THREAD_PER_BLOCK,1);

    printf("--------- \n");
    reduce<<<Grid,Block>>>(d_input, d_output);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
        // Possibly: exit(-1) if program cannot continue....
    }

    printf("+++++++++++++++ \n");
    // Copy GPU output to CPU (Note: first param is cpu out and second param is gpu out)
    cudaMemcpy(output,d_output,block_num * sizeof(float), cudaMemcpyDeviceToHost);

    // the final output is array of per block output
    // So compare the element values within each block
    if(check(output,result,block_num))
        printf("The ans is right \n");
    else{
        printf("the ans if wrong\n");
        for (int i=0; i<block_num;i++)
        {
            printf("%lf ", output[i]);
        }
        printf("\n");
    }

    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}