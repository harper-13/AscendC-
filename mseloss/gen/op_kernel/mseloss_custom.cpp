#include "kernel_operator.h"
using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;

class KernelMseLoss {
 public:
  __aicore__ inline KernelMseLoss() {}

  __aicore__ inline void Init(GM_ADDR ground_truth, GM_ADDR predictions, GM_ADDR mse_output, GM_ADDR tiling_data_addr) {
    GET_TILING_DATA(tiling_data, tiling_data_addr);
    ASSERT(GetBlockNum() != 0 && "Block dimension cannot be zero!");

    this->blockLength = tiling_data.totalLength / GetBlockNum();
    this->numTiles = tiling_data.tileNum;
    this->tileSize = this->blockLength / tiling_data.tileNum / BUFFER_NUM;
    uint32_t block_offset = this->blockLength * GetBlockIdx();

    groundTruthGm.SetGlobalBuffer((__gm__ DTYPE_X*)ground_truth + block_offset, this->blockLength);
    predictionsGm.SetGlobalBuffer((__gm__ DTYPE_Y*)predictions + block_offset, this->blockLength);
    mseOutputGm.SetGlobalBuffer((__gm__ DTYPE_Z*)mse_output + block_offset, this->blockLength);

    pipe.InitBuffer(queueGroundTruth, BUFFER_NUM, this->tileSize * sizeof(DTYPE_X));
    pipe.InitBuffer(queuePredictions, BUFFER_NUM, this->tileSize * sizeof(DTYPE_Y));
    pipe.InitBuffer(queueMseOutput, BUFFER_NUM, this->tileSize * sizeof(DTYPE_Z));
  }

  __aicore__ inline void Process() {
    for (int32_t idx = 0; idx < this->numTiles * BUFFER_NUM; idx++) {
      HandleTile(idx);
    }
  }

 private:
  __aicore__ inline void HandleTile(int32_t idx) {
    CopyInput(idx);
    PerformComputation(idx);
    CopyOutput(idx);
  }

  __aicore__ inline void CopyInput(int32_t idx) {
    LocalTensor<DTYPE_X> groundTruthLocal = queueGroundTruth.AllocTensor<DTYPE_X>();
    LocalTensor<DTYPE_Y> predictionsLocal = queuePredictions.AllocTensor<DTYPE_Y>();
    
    DataCopy(groundTruthLocal, groundTruthGm[idx * this->tileSize], this->tileSize);
    DataCopy(predictionsLocal, predictionsGm[idx * this->tileSize], this->tileSize);
    
    queueGroundTruth.EnQue(groundTruthLocal);
    queuePredictions.EnQue(predictionsLocal);
  }

  __aicore__ inline void PerformComputation(int32_t idx) {
    LocalTensor<DTYPE_X> groundTruthLocal = queueGroundTruth.DeQue<DTYPE_X>();
    LocalTensor<DTYPE_Y> predictionsLocal = queuePredictions.DeQue<DTYPE_Y>();
    LocalTensor<DTYPE_Z> mseLocal = queueMseOutput.AllocTensor<DTYPE_Z>();

    Sub(mseLocal, groundTruthLocal, predictionsLocal, this->tileSize);
    Mul(mseLocal, mseLocal, mseLocal, this->tileSize);

    queueMseOutput.EnQue<DTYPE_Z>(mseLocal);
    queueGroundTruth.FreeTensor(groundTruthLocal);
    queuePredictions.FreeTensor(predictionsLocal);
  }

  __aicore__ inline void CopyOutput(int32_t idx) {
    LocalTensor<DTYPE_Z> mseLocal = queueMseOutput.DeQue<DTYPE_Z>();
    DataCopy(mseOutputGm[idx * this->tileSize], mseLocal, this->tileSize);
    queueMseOutput.FreeTensor(mseLocal);
  }

 private:
  TPipe pipe;
  TBuf<QuePosition::VECCALC> tempBuffer;
  TQue<QuePosition::VECIN, BUFFER_NUM> queueGroundTruth;
  TQue<QuePosition::VECIN, BUFFER_NUM> queuePredictions;
  TQue<QuePosition::VECOUT, BUFFER_NUM> queueMseOutput;
  GlobalTensor<DTYPE_X> groundTruthGm;
  GlobalTensor<DTYPE_Y> predictionsGm;
  GlobalTensor<DTYPE_Z> mseOutputGm;

  uint32_t blockLength;
  uint32_t numTiles;
  uint32_t tileSize;
};


extern "C" __global__ __aicore__ void mseloss_custom(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    // TODO: user kernel impl
    KernelMseLoss op;
    op.Init(x, y, z, tiling);
    op.Process();
}