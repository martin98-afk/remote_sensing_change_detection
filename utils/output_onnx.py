# encoding:utf-8
# 将pytorch模型转换为onnx模型导出
# onnx支持大多数框架下模型的转换，便于整合模型，并且还能加速推理，
# 更可以方便的通过TensorRT或者openvino部署得到进一步提速
import numpy as np
import onnx
import onnxruntime
import torch

from utils.pipeline import RSPipeline


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def softmax(array):
    array = np.exp(array)
    sum_array = np.sum(array, axis=1)
    sum_array = np.expand_dims(sum_array, axis=1)
    sum_array = np.repeat(sum_array, repeats=array.shape[1], axis=1)
    return array / sum_array


if __name__ == "__main__":
    # 查看当前机器
    print(onnxruntime.get_device())
    print(onnxruntime.get_available_providers())
    # 导入pytorch模型
    device = "cuda:0"
    data, model = RSPipeline.load_model("../output/ss_eff_b0.yaml", "pytorch", device=device)
    # 将模型和输入数据转为半精度
    dummy_input = np.load("../sample.npy")
    with torch.no_grad():
        torch_out = model(torch.from_numpy(dummy_input).to(device))
        torch_out = torch.nn.Softmax(dim=1)(torch_out)
    torch_out = torch_out.cpu().numpy()

    # 将输入和输出数据的第一维定义为可以变动的
    dynamic_axes = {
        "input":  {0: "batch"},
        "output": {0: "batch"}
    }
    # 定义输入输出字段
    input_names = ["input"]
    output_names = ["output"]
    # 导出为onnx模型
    torch.onnx.export(
            model=model,
            args=torch.from_numpy(dummy_input).to(device),
            f="../output/ss_eff_b0.onnx",
            input_names=input_names,
            output_names=output_names,
            verbose=False,
            dynamic_axes=dynamic_axes
    )

    # 使用onnx-simplifier对模型进行精简
    import onnxsim

    print("\nStarting to simplify ONNX...")
    # 导入onnx模型
    onnxmodel = onnx.load("../output/ss_eff_b0.onnx")
    onnx.checker.check_model(onnxmodel)
    onnxsim_model, check = onnxsim.simplify(onnxmodel)
    assert check, "assert check failed"

    # 保存精简后的onnx模型
    onnx.save(onnxsim_model, "../output/ss_eff_b0.onnx")
    # 结束模型转换
    print("\nExport complete")

    # 验证模型精度是否有损失
    # pytorch模型输出
    # 使用onnx模型输入测试数据
    session = onnxruntime.InferenceSession("../output/ss_eff_b0.onnx",
                                           providers=['TensorrtExecutionProvider',
                                                      'CUDAExecutionProvider'])
    # onnx模型输出
    onnx_out = session.run(["output"], {"input": dummy_input})[0]
    onnx_out = softmax(onnx_out)
    # 判断输出结果是否一直，小数点后3位一致即可
    print(np.max(torch_out - onnx_out))
    # 画图展示结果
    import matplotlib.pyplot as plt
    torch_out = np.argmax(torch_out, axis=1)
    onnx_out = np.argmax(onnx_out, axis=1)
    fg, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(torch_out[0, ...])
    ax[0].set_title("pytorch model output")
    ax[1].imshow(onnx_out[0, ...])
    ax[1].set_title("onnx model output")
    plt.show()

