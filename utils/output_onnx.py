# encoding:utf-8
# 将pytorch模型转换为onnx模型导出
# onnx支持大多数框架下模型的转换，便于整合模型，并且还能加速推理，
# 更可以方便的通过TensorRT或者openvino部署得到进一步提速
import numpy as np
import onnx
import onnxruntime as ort
import torch


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def softmax(array):
    array = np.exp(array)
    sum_array = np.sum(array, axis=1)
    sum_array = np.expand_dims(sum_array, axis=1)
    sum_array = np.repeat(sum_array, repeats=array.shape[1], axis=1)
    return array / sum_array


def check_models(torch_out, onnx_out):
    """
    验证模型精度是否有损失, 判断输出结果是否一直，小数点后3位一致即可

    :param torch_out:
    :param onnx_out:
    :return:
    """
    #
    # pytorch模型输出
    #
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


def load_onnx_model(model_path, device):
    """


    :param model_path:
    :param device:
    :return:
    """
    providers = [
        ('TensorrtExecutionProvider', {
            'device_id':              1,
            'trt_max_workspace_size': 2147483648,
            'trt_fp16_enable':        True,
        }),
        ('CUDAExecutionProvider', {
            'device_id':                 1,
            'arena_extend_strategy':     'kNextPowerOfTwo',
            'gpu_mem_limit':             6 * 1024 * 1024 * 1024,
            'cudnn_conv_algo_search':    'EXHAUSTIVE',
            'do_copy_in_default_stream': False,
        }),
        'CPUExecutionProvider'
    ]

    ort_session = ort.InferenceSession(model_path, providers=providers)
    return ort_session


def simplify_onnx_model(model_path):
    """
    使用onnxsim工具包将onnx模型进行精简。

    :param model_path: 模型保存地址
    :return:
    """
    # 使用onnx-simplifier对模型进行精简
    import onnxsim

    print("\nStarting to simplify ONNX...")
    # 导入onnx模型
    onnxmodel = onnx.load(model_path)
    onnx.checker.check_model(onnxmodel)
    onnxsim_model, check = onnxsim.simplify(onnxmodel)
    assert check, "assert check failed"

    # 保存精简后的onnx模型
    onnx.save(onnxsim_model, model_path)


def torch2onnx(model,
               output_path,
               sample_data=np.random.randn(1, 3, 512, 512).astype(np.float32),
               simplify=True):
    """
    将pytorch模型转换为onnx模型。

    :param sample_data:
    :param simplify:
    :param model: pytorch模型。
    :param output_path: onnx模型保存地址，
    :return:
    """
    # 查看当前机器
    print(ort.get_device())
    print(ort.get_available_providers())
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
            args=torch.from_numpy(sample_data).to("cuda"),
            f=output_path,
            input_names=input_names,
            output_names=output_names,
            verbose=False,
            dynamic_axes=dynamic_axes
    )
    if simplify:
        simplify_onnx_model(output_path)
    # 结束模型转换
    print("\nExport complete")


if __name__ == "__main__":
    ...
