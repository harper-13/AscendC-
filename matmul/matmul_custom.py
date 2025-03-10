#!/usr/bin/python3
# -*- coding:utf-8 -*-
# Copyright 2022-2023 Huawei Technologies Co., Ltd
import numpy as np
import os

def gen_golden_data():
    x1_gm_type = np.float16
    x2_gm_type = np.float16

    M = 512
    N = 1024
    K = 512

    x1_gm = np.random.randint(1, 10, [M, K]).astype(x1_gm_type)
    x2_gm = np.random.randint(1, 10, [K, N]).astype(x2_gm_type)
    bias_gm = np.random.randint(1, 10, [1, N]).astype(np.float32)

    golden = np.matmul(x1_gm.astype(np.float32), x2_gm.astype(np.float32)).astype(np.float32)
    golden = (golden + bias_gm).astype(np.float32)

    os.system("mkdir -p input")
    os.system("mkdir -p output")
    x1_gm.tofile("./input/x1_gm.bin")
    x2_gm.tofile("./input/x2_gm.bin")
    bias_gm.tofile("./input/bias_gm.bin")
    golden.tofile("./output/golden.bin")


if __name__ == "__main__":
    gen_golden_data()
