#include "kernel_operator.h"

using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;

class KernelSwish {
public:
    __aicore__ inline KernelSwish() {}

    // 初始化函数，设置必要的参数并分配内存
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalLength, uint32_t tileNum)
    {
        ASSERT(GetBlockNum() != 0 && "Block dimension cannot be zero!");
        // 计算每个块的长度
        this->blockLength = totalLength / GetBlockNum();
        this->tileNum = tileNum;
        this->negativeSlope = static_cast<float>(negativeSlope);
        // 确保tile数不为零
        ASSERT(tileNum != 0 && "Tile number cannot be zero!");
        this->tileLength = this->blockLength / tileNum / BUFFER_NUM;
        // 设置全局内存缓冲区，获取当前核心的内存区域
        xGm.SetGlobalBuffer((__gm__ float*)x + this->blockLength * GetBlockIdx(), this->blockLength);
        yGm.SetGlobalBuffer((__gm__ float*)y + this->blockLength * GetBlockIdx(), this->blockLength);
        // 初始化管道缓冲区
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(float));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileLength * sizeof(float));
        pipe.InitBuffer(tmpBuffer, this->tileLength * sizeof(float));
    }

    // 主处理函数
    __aicore__ inline void Process()
    {
        int32_t loopCount = this->tileNum * BUFFER_NUM;
        for (int32_t i = 0; i < loopCount; i++) {
            CopyIn(i);     
            Compute(i);    
            CopyOut(i);    
        }
    }

private:
    // 将数据从输入队列拷贝到局部内存
    __aicore__ inline void CopyIn(int32_t progress)
    {
        LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
        DataCopy(xLocal, xGm[progress * this->tileLength], this->tileLength);
        inQueueX.EnQue(xLocal);
    }

    // 执行Leaky ReLU计算
    __aicore__ inline void Compute(int32_t progress)
    {
        // 从输入队列中取出张量
        LocalTensor<float> xLocal = inQueueX.DeQue<float>();
        LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();
        // Leaky ReLU实现：计算 y=x*max(0,x) ,将值1填充到tmpTensor
        LocalTensor<float> tmpTensor = tmpBuffer.Get<float>();
        float one(1), negOne(-1);
        AscendC::Duplicate<float>(tmpTensor, one, this->tileLength);
        // 计算 y=x*(-1) 并取指数, y=y+1,y=1/y,y=x*y
        Muls(yLocal, xLocal, negOne, this->tileLength);
        Exp(yLocal, yLocal, this->tileLength);
        Adds(yLocal, yLocal, one, this->tileLength);
        Div(yLocal, tmpTensor, yLocal, this->tileLength);
        Mul(yLocal, yLocal, xLocal, this->tileLength);
        // 将结果加入输出队列,释放输入张量，便于重用
        outQueueY.EnQue<float>(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

    // 将计算结果从局部内存拷贝回全局内存
    __aicore__ inline void CopyOut(int32_t progress)
    {
        LocalTensor<float> yLocal = outQueueY.DeQue<float>();
        DataCopy(yGm[progress * this->tileLength], yLocal, this->tileLength);
        outQueueY.FreeTensor(yLocal);
    }

private:
    TPipe pipe;  // 管道，管理输入和输出队列
    TBuf<QuePosition::VECCALC> tmpBuffer;  // 临时缓冲区，用于存储中间计算结果
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;  // 输入队列
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;  // 输出队列
    GlobalTensor<float> xGm, yGm;  // 全局内存中的张量x和y
    float negativeSlope;  // Leaky ReLU的负斜率
    uint32_t blockLength;  // 每个块的长度
    uint32_t tileNum;  // tile的数量
    uint32_t tileLength;  // 每个tile的长度
};


extern "C" __global__ __aicore__ void swish_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    // TODO: user kernel impl
    KernelSwish op;
    op.Init(x, y, tiling_data.totalLength, tiling_data.tileNum);
    op.Process();
}