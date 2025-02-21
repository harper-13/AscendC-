/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.
 * This file constains code of cpu debug and npu code.We read data from bin file
 * and write result to file.
 */
#include "data_utils.h"
#include <chrono>
#ifndef __CCE_KT_TEST__
#include "acl/acl.h"
extern void matmul_custom_do(uint32_t coreDim, void* l2ctrl, void* stream,
    uint8_t *A, uint8_t *B, uint8_t *C, uint8_t *bias, uint8_t *tiling);
#else
#include "tikicpulib.h"
extern "C" void matmul_custom(uint8_t *A, uint8_t *B, uint8_t *C, uint8_t *bias, uint8_t * tiling);
#endif

int32_t main(int32_t argc, char* argv[])
{
    size_t AFileSize = 512 * 512 * sizeof(uint16_t);  // uint16_t represent half
    size_t BFileSize = 512 * 1024 * sizeof(uint16_t);  // uint16_t represent half
    size_t CFileSize = 512 * 1024 * sizeof(float);
    size_t tilingFileSize = 50 * sizeof(uint32_t);
    size_t biasFileSize = 1 * 1024 * sizeof(uint32_t);
    uint32_t blockDim = 1;

#ifdef __CCE_KT_TEST__
    uint8_t *A = (uint8_t *)AscendC::GmAlloc(AFileSize);
    uint8_t *B = (uint8_t *)AscendC::GmAlloc(BFileSize);
    uint8_t *C = (uint8_t *)AscendC::GmAlloc(CFileSize);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingFileSize);
    uint8_t *bias = (uint8_t *) AscendC::GmAlloc(biasFileSize);

    ReadFile("./input/x1_gm.bin", AFileSize, A, AFileSize);
    // PrintData(A, 16, printDataType::HALF);
    ReadFile("./input/x2_gm.bin", BFileSize, B, BFileSize);
    // PrintData(B, 16, printDataType::HALF);
    ReadFile("./input/tiling.bin", tilingFileSize, tiling, tilingFileSize);
    // PrintData(tiling, 16, printDataType::UINT32_T);
    ReadFile("./input/bias_gm.bin", biasFileSize, bias, biasFileSize);

    ICPU_RUN_KF(matmul_custom, blockDim, A, B, C, bias, tiling);

    // PrintData(C, 16, printDataType::FLOAT);
    WriteFile("./output/output.bin", C, CFileSize);

    AscendC::GmFree((void *)A);
    AscendC::GmFree((void *)B);
    AscendC::GmFree((void *)C);
    AscendC::GmFree((void *)tiling);
    AscendC::GmFree((void*)bias);
#else
    CHECK_ACL(aclInit(nullptr));
    aclrtContext context;
    int32_t deviceId = 0;
    CHECK_ACL(aclrtSetDevice(deviceId));
    CHECK_ACL(aclrtCreateContext(&context, deviceId));
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));

    uint8_t *AHost;
    uint8_t *ADevice;
    CHECK_ACL(aclrtMallocHost((void**)(&AHost), AFileSize));
    CHECK_ACL(aclrtMalloc((void**)&ADevice, AFileSize, ACL_MEM_MALLOC_HUGE_FIRST));
    ReadFile("./input/x1_gm.bin", AFileSize, AHost, AFileSize);
    // PrintData(AHost, 16, printDataType::HALF);
    CHECK_ACL(aclrtMemcpy(ADevice, AFileSize, AHost, AFileSize, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *BHost;
    uint8_t *BDevice;
    CHECK_ACL(aclrtMallocHost((void**)(&BHost), BFileSize));
    CHECK_ACL(aclrtMalloc((void**)&BDevice, BFileSize, ACL_MEM_MALLOC_HUGE_FIRST));
    ReadFile("./input/x2_gm.bin", BFileSize, BHost, BFileSize);
    // PrintData(BHost, 16, printDataType::HALF);
    CHECK_ACL(aclrtMemcpy(BDevice, BFileSize, BHost, BFileSize, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *biasHost;
    uint8_t *biasDevice;
    CHECK_ACL(aclrtMallocHost((void**)(&biasHost), biasFileSize));
    CHECK_ACL(aclrtMalloc((void**)&biasDevice, biasFileSize, ACL_MEM_MALLOC_HUGE_FIRST));
    ReadFile("./input/bias_gm.bin", biasFileSize, biasHost, biasFileSize);
    // PrintData(BHost, 16, printDataType::HALF);
    CHECK_ACL(aclrtMemcpy(biasDevice, biasFileSize, biasHost, biasFileSize, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *tilingHost;
    uint8_t *tilingDevice;
    CHECK_ACL(aclrtMallocHost((void**)(&tilingHost), tilingFileSize));
    CHECK_ACL(aclrtMalloc((void**)&tilingDevice, tilingFileSize, ACL_MEM_MALLOC_HUGE_FIRST));
    ReadFile("./input/tiling.bin", tilingFileSize, tilingHost, tilingFileSize);
    // PrintData(tilingHost, 16, printDataType::UINT32_T);
    CHECK_ACL(aclrtMemcpy(tilingDevice, tilingFileSize, tilingHost, tilingFileSize, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *CHost;
    uint8_t *CDevice;
    CHECK_ACL(aclrtMallocHost((void**)(&CHost), CFileSize));
    CHECK_ACL(aclrtMalloc((void**)&CDevice, CFileSize, ACL_MEM_MALLOC_HUGE_FIRST));

    matmul_custom_do(blockDim, nullptr, stream, ADevice, BDevice, CDevice, biasDevice, tilingDevice);
    CHECK_ACL(aclrtSynchronizeStream(stream));

    CHECK_ACL(aclrtMemcpy(CHost, CFileSize, CDevice, CFileSize, ACL_MEMCPY_DEVICE_TO_HOST));
    // PrintData(CHost, 16, printDataType::FLOAT);
    WriteFile("./output/output.bin", CHost, CFileSize);
    
    CHECK_ACL(aclrtFree(ADevice));
    CHECK_ACL(aclrtFreeHost(AHost));
    CHECK_ACL(aclrtFree(BDevice));
    CHECK_ACL(aclrtFreeHost(BHost));
    CHECK_ACL(aclrtFree(tilingDevice));
    CHECK_ACL(aclrtFreeHost(tilingHost));
    CHECK_ACL(aclrtFree(CDevice));
    CHECK_ACL(aclrtFreeHost(CHost));

    CHECK_ACL(aclrtDestroyStream(stream));
    CHECK_ACL(aclrtDestroyContext(context));
    CHECK_ACL(aclrtResetDevice(deviceId));
    CHECK_ACL(aclFinalize());
#endif
    return 0;
}
