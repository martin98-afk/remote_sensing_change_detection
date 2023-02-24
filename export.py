# encoding:utf-8
import gc

import numpy as np

from utils.output_onnx import torch2onnx
from utils.pipeline import RSPipeline

if __name__ == "__main__":
    half = False
    data, model = RSPipeline.load_model("output/ss_eff_b0.yaml",
                                        "pytorch",
                                        'cuda',
                                        half=half)

    sample_data = np.random.randn(1, 3, 512, 512)
    sample_data = sample_data.astype(np.float16) if half else sample_data.astype(
            np.float32)
    torch2onnx(model, "output/ss_eff_b0.onnx",
               sample_data=sample_data)
    del model
    gc.collect()