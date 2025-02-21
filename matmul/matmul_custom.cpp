/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.
 */

#include "kernel_operator.h"
#include "lib/matrix/matmul/matmul.h"
using namespace AscendC;
using namespace matmul;


__aicore__ inline void CopyTiling(TCubeTiling* tiling, GM_ADDR tilingGM)
{
    uint32_t* ptr = reinterpret_cast<uint32_t*>(tiling);
    auto tiling32 = reinterpret_cast<__gm__ uint32_t*>(tilingGM);

    for (int i = 0; i < sizeof(TCubeTiling) / sizeof(uint32_t); i++, ptr++) {
        *ptr = *(tiling32 + i);
    }
    return;
}

extern "C" __global__ __aicore__ void matmul_custom(GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR bias, GM_ADDR tilingGm)
{
    using AType = half;  
    using BType = half; 
    using CType = float; 
    using BiasType = float; 
    
    // 初始化tiling结构体，用来存储矩阵分块信息
    TCubeTiling tiling;
    CopyTiling(&tiling, tilingGm);  

    TPipe queue;  // 用于管道操作的队列

    // 定义全局张量对象，分别对应输入矩阵A、B，输出矩阵C和偏置矩阵
    GlobalTensor<AType> aGlobal;
    GlobalTensor<BType> bGlobal;
    GlobalTensor<CType> cGlobal;
    GlobalTensor<BiasType> biasGlobal;

    // 设置全局内存缓冲区，分配矩阵A、B、C和偏置矩阵的内存
    aGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ AType*>(a), tiling.M * tiling.Ka);
    bGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ BType*>(b), tiling.Kb * tiling.N);
    cGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ CType*>(c), tiling.M * tiling.N);
    biasGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ BiasType*>(bias), tiling.M * tiling.N);

    // 初始化偏移量
    int offsetA = 0;
    int offsetB = 0;
    int offsetC = 0;
    int offsetBias = 0;

    // 计算M方向的总块数，ceil函数向上取整
    auto numBlocksM = Ceil(tiling.M, tiling.singleCoreM);

    // 计算当前块的M方向和N方向的索引
    auto coreIdxM = GetBlockIdx() % numBlocksM;
    auto coreIdxN = GetBlockIdx() / numBlocksM;

    // 计算每个矩阵在全局内存中的偏移量
    offsetA = coreIdxM * tiling.Ka * tiling.singleCoreM;
    offsetB = coreIdxN * tiling.singleCoreN;
    offsetC = coreIdxM * tiling.N * tiling.singleCoreM + coreIdxN * tiling.singleCoreN;

    // 根据计算出的偏移量获取全局内存中的相应数据块
    auto gmA = aGlobal[offsetA];
    auto gmB = bGlobal[offsetB];
    auto gmC = cGlobal[offsetC];
    auto gmBias = biasGlobal[offsetBias];

    // 定义矩阵乘法中使用的类型
    typedef MatmulType<AscendC::TPosition::GM, CubeFormat::ND, AType> ATypeMatmul;
    typedef MatmulType<AscendC::TPosition::GM, CubeFormat::ND, BType> BTypeMatmul;
    typedef MatmulType<AscendC::TPosition::GM, CubeFormat::ND, CType> CTypeMatmul;
    typedef MatmulType<AscendC::TPosition::GM, CubeFormat::ND, BiasType> BiasTypeMatmul;

    // 初始化矩阵乘法实现
    MatmulImpl<ATypeMatmul, BTypeMatmul, CTypeMatmul, BiasTypeMatmul> matmulImpl;
    matmulImpl.SetSubBlockIdx(0);  // 设置子块索引
    matmulImpl.Init(&tiling, &queue);  // 初始化矩阵乘法操作

    // 设置矩阵A、B和偏置矩阵
    matmulImpl.SetTensorA(gmA);
    matmulImpl.SetTensorB(gmB);
    matmulImpl.SetBias(gmBias);

    // 执行矩阵乘法操作，将结果存储到矩阵C
    matmulImpl.IterateAll(gmC);
}



#ifndef __CCE_KT_TEST__
// call of kernel function
void matmul_custom_do(uint32_t blockDim, void* l2ctrl, void* stream, uint8_t* a, uint8_t* b, uint8_t* c, uint8_t * bias, uint8_t* tilingGm)
{
    matmul_custom<<<blockDim, l2ctrl, stream>>>(a, b, c, bias, tilingGm);
}
#endif