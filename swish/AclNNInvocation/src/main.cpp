/**
* @file main.cpp
*
* Copyright (C) 2023. Huawei Technologies Co., Ltd. All rights reserved.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
*/
#include <fstream>
#include <unistd.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "acl/acl_op_compiler.h"
#include "aclnn/acl_meta.h"
#include "aclnn_swish_custom.h"

#define SUCCESS 0
#define FAILED 1

#define INFO_LOG(fmt, args...) fprintf(stdout, "[INFO]  " fmt "\n", ##args)
#define ERROR_LOG(fmt, args...) fprintf(stderr, "[ERROR]  " fmt "\n", ##args)

bool g_isDevice = false;
int deviceId = 0;
aclrtStream stream = nullptr;
size_t inputSize = 0;
size_t outputSize = 0;
void* hostMemX = nullptr;
void* hostMemY = nullptr;
void* devMemX = nullptr;
void* devMemY = nullptr;
aclDataBuffer* inputBuffer; 
aclDataBuffer* outputBuffer;
aclTensor* inputTensor;
aclTensor* outputTensor;

bool ReadFile(const std::string &filePath, size_t fileSize, void *buffer, size_t bufferSize)
{
    struct stat sBuf;
    int fileStatus = stat(filePath.data(), &sBuf);
    if (fileStatus == -1) {
        ERROR_LOG("failed to get file %s", filePath.c_str());
        return false;
    }
    if (S_ISREG(sBuf.st_mode) == 0) {
        ERROR_LOG("%s is not a file, please enter a file", filePath.c_str());
        return false;
    }

    std::ifstream file;
    file.open(filePath, std::ios::binary);
    if (!file.is_open()) {
        ERROR_LOG("Open file failed. path = %s", filePath.c_str());
        return false;
    }

    std::filebuf *buf = file.rdbuf();
    size_t size = buf->pubseekoff(0, std::ios::end, std::ios::in);
    if (size == 0) {
        ERROR_LOG("file size is 0");
        file.close();
        return false;
    }
    if (size > bufferSize) {
        ERROR_LOG("file size is larger than buffer size");
        file.close();
        return false;
    }
    buf->pubseekpos(0, std::ios::in);
    buf->sgetn(static_cast<char *>(buffer), size);
    fileSize = size;
    file.close();
    return true;
}

bool WriteFile(const std::string &filePath, const void *buffer, size_t size)
{
    if (buffer == nullptr) {
        ERROR_LOG("Write file failed. buffer is nullptr");
        return false;
    }

    int fd = open(filePath.c_str(), O_RDWR | O_CREAT | O_TRUNC, S_IRUSR | S_IWRITE);
    if (fd < 0) {
        ERROR_LOG("Open file failed. path = %s", filePath.c_str());
        return false;
    }

    auto writeSize = write(fd, buffer, size);
    (void) close(fd);
    if (writeSize != size) {
        ERROR_LOG("Write file Failed.");
        return false;
    }

    return true;
}

void DestoryResource()
{
    bool flag = false;

    if (aclrtDestroyStream(stream) != ACL_SUCCESS) {
        ERROR_LOG("Reset device %d failed", deviceId);
        flag = true;
    }
    (void)aclrtDestroyStream(stream);

    if (aclrtResetDevice(deviceId) != ACL_SUCCESS) {
        ERROR_LOG("Reset device %d failed", deviceId);
        flag = true;
    }
    INFO_LOG("Reset Device success");
    if (aclFinalize() != ACL_SUCCESS) {
        ERROR_LOG("Finalize acl failed");
        flag = true;
    }
    if (flag) {
        ERROR_LOG("Destory resource failed");
    } else {
        INFO_LOG("Destory resource success");
    }
}

bool InitResource()
{
    // acl.json is dump or profiling config file
    if (aclInit(NULL) != ACL_SUCCESS) {
        ERROR_LOG("acl init failed");
        return false;
    }

    if (aclrtSetDevice(deviceId) != ACL_SUCCESS) {
        ERROR_LOG("Set device failed. deviceId is %d", deviceId);
        (void)aclFinalize();
        return false;
    }
    INFO_LOG("Set device[%d] success", deviceId);

    if (aclrtCreateStream(&stream) != ACL_SUCCESS) {
        ERROR_LOG("Create stream failed");
        return false;
    }
    INFO_LOG("Create stream success");

    aclrtRunMode runMode;
    if (aclrtGetRunMode(&runMode) != ACL_SUCCESS) {
        ERROR_LOG("Get run mode failed");
        DestoryResource();
        return false;
    }
    g_isDevice = (runMode == ACL_DEVICE);
    INFO_LOG("Get RunMode[%d] success", runMode);

    return true;
}

bool ProcessInput()
{
    // define operator
    aclDataType dataType = ACL_FLOAT;
    aclFormat format = ACL_FORMAT_ND;
    std::vector<int64_t> shape { 8, 200, 1024 };
    int64_t shape_size = 1;
    for (auto i : shape) {
        shape_size *= i;
    }
    inputSize = shape_size * aclDataTypeSize(dataType);
    outputSize = shape_size * aclDataTypeSize(dataType);
    if (g_isDevice) {
        if (aclrtMalloc(&hostMemX, inputSize, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
            ERROR_LOG("Malloc device memory for input x failed");
            return false;
        }
        if (aclrtMalloc(&hostMemY, outputSize, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
            ERROR_LOG("Malloc device memory for output y failed");
            return false;
        }
    } else {
        if (aclrtMallocHost(&hostMemX, inputSize) != ACL_SUCCESS) {
            ERROR_LOG("Malloc device memory for input x failed");
            return false;
        }
        if (aclrtMallocHost(&hostMemY, outputSize) != ACL_SUCCESS) {
            ERROR_LOG("Malloc device memory for output y failed");
            return false;
        }
    }
    if (aclrtMalloc(&devMemX, inputSize, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
        ERROR_LOG("Malloc device memory for input x failed");
        return false;
    }
    if (aclrtMalloc(&devMemY, outputSize, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
        ERROR_LOG("Malloc device memory for output y failed");
        return false;
    }
    inputBuffer = aclCreateDataBuffer(devMemX, inputSize);
    outputBuffer = aclCreateDataBuffer(devMemY, outputSize);

    inputTensor = aclCreateTensor(shape.data(), shape.size(), dataType,
        nullptr, 0, format, shape.data(), shape.size(), devMemX);
    if (inputTensor == nullptr) {
        ERROR_LOG("Create Tensor for input x failed");
        return false;
    }    
    outputTensor = aclCreateTensor(shape.data(), shape.size(), dataType,
        nullptr, 0, format, shape.data(), shape.size(), devMemY);
    if (outputTensor == nullptr) {
        ERROR_LOG("Create Tensor for output y failed");
        return false;
    }
    size_t fileSize = 0;
    ReadFile("../input/input_x.bin", fileSize, reinterpret_cast<void *>(hostMemX), inputSize);
    
    aclrtMemcpyKind kind = ACL_MEMCPY_HOST_TO_DEVICE;
    if (g_isDevice) {
        kind = ACL_MEMCPY_DEVICE_TO_DEVICE;
    }
    if (aclrtMemcpy(devMemX, inputSize, hostMemX, inputSize, kind) != ACL_SUCCESS) {
        ERROR_LOG("Copy input x failed");
        return false;
    }
    INFO_LOG("Set input success.");
    return true;
}

bool ExecuteOp()
{
    size_t workspaceSize = 0;
    aclOpExecutor *handle = nullptr;
    double negativeSlope = 0.0;
    auto ret = aclnnSwishCustomGetWorkspaceSize(inputTensor, outputTensor,
                                              &workspaceSize, &handle);
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("Get Operator Workspace failed. error code is %d", static_cast<int32_t>(ret));
        return false;
    }
    INFO_LOG("Execute aclnnSwishCustomGetWorkspaceSize success, workspace size %lu", workspaceSize);
    
    void *workspace = nullptr;
    if (workspaceSize != 0) {
        ret = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) {
            ERROR_LOG("Malloc device memory failed");
        }
    }
    ret = aclnnSwishCustom(workspace, workspaceSize, handle, stream);
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("Execute Operator failed. error code is %d", static_cast<int32_t>(ret));
        return false;
    }
    INFO_LOG("Execute aclnnSwishCustom success");
    ret = aclrtSynchronizeStreamWithTimeout(stream, 5000);
    if (ret != SUCCESS) {
        ERROR_LOG("Synchronize stream failed. error code is %d", static_cast<int32_t>(ret));
        return false;
    }
    INFO_LOG("Synchronize stream success");
    return true;
}

bool GetResult() {
    aclrtMemcpyKind kind = ACL_MEMCPY_DEVICE_TO_HOST;
    if (g_isDevice) {
        kind = ACL_MEMCPY_DEVICE_TO_DEVICE;
    }
    if (aclrtMemcpy(hostMemY, outputSize, devMemY, outputSize, kind) != ACL_SUCCESS) {
        ERROR_LOG("Copy output y failed");
        return false;
    }
    WriteFile("../output/output_y.bin", reinterpret_cast<void *>(hostMemY), outputSize);
    INFO_LOG("Write output success.");
    (void)aclDestroyTensor(inputTensor);
    (void)aclDestroyTensor(outputTensor);
    (void)aclDestroyDataBuffer(inputBuffer);
    (void)aclDestroyDataBuffer(outputBuffer);
    (void)aclrtFree(devMemX);
    (void)aclrtFree(devMemY);
    if (g_isDevice) {
        (void)aclrtFree(hostMemX);
        (void)aclrtFree(hostMemY);
    } else {
        (void)aclrtFreeHost(hostMemX);
        (void)aclrtFreeHost(hostMemY);
    }
    return true;
}

int main(int argc, char **argv)
{
    if (!InitResource()) {
        ERROR_LOG("Init resource failed");
        return FAILED;
    }
    INFO_LOG("Init resource success");

    if (!ProcessInput()) {
        ERROR_LOG("ProcessInput failed");
        DestoryResource();
        return FAILED;
    }
    if (!ExecuteOp()) {
        ERROR_LOG("ExecuteOp failed");
        DestoryResource();
        return FAILED;
    }
    if (!GetResult()) {
        ERROR_LOG("GetResult failed");
        DestoryResource();
        return FAILED;
    }

    DestoryResource();

    return SUCCESS;
}
