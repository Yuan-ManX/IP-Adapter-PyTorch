import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image


"""
这是一个全局字典，用于存储不同层的注意力映射（attention maps）。
键（key）为层的名称，值（value）为对应的注意力映射张量。
"""
attn_maps = {}


def hook_fn(name):
    """
    为指定名称的模块注册一个前向钩子（forward hook），用于捕获注意力映射。

    Args:
        name (str): 模块的名称，用于在全局字典中存储对应的注意力映射。

    Returns:
        function: 一个闭包函数，作为前向钩子函数。
    """
    def forward_hook(module, input, output):
        """
        前向钩子函数，在模块前向传播时调用。

        该函数检查模块的处理器（processor）是否具有 `attn_map` 属性，
        如果有，则将其存储到全局的 `attn_maps` 字典中，并删除模块中的 `attn_map` 属性以释放内存。

        Args:
            module (nn.Module): 当前前向传播的模块。
            input (tuple): 模块的输入张量。
            output (torch.Tensor): 模块的输出张量。
        """
        if hasattr(module.processor, "attn_map"):
            # 将注意力映射存储到全局字典中，键为模块的名称
            attn_maps[name] = module.processor.attn_map
            # 删除模块中的 `attn_map` 属性以释放内存
            del module.processor.attn_map

    return forward_hook


def register_cross_attention_hook(unet):
    """
    为U-Net模型中的跨注意力层注册前向钩子，以捕获注意力映射。

    Args:
        unet (nn.Module): U-Net模型实例。

    Returns:
        nn.Module: 注册了钩子后的U-Net模型。
    """
    for name, module in unet.named_modules():
        if name.split('.')[-1].startswith('attn2'):
            module.register_forward_hook(hook_fn(name))

    return unet


def upscale(attn_map, target_size):
    """
    对注意力映射进行上采样，使其与目标图像尺寸匹配。

    Args:
        attn_map (torch.Tensor): 输入的注意力映射张量，形状通常为 (heads, height, width)。
        target_size (tuple): 目标图像的尺寸，格式为 (height, width)。

    Returns:
        torch.Tensor: 上采样后的注意力映射张量，形状为 (1, height, width)。
    """
    # 对注意力映射的注意力头维度取平均，得到形状为 (height, width) 的张量
    attn_map = torch.mean(attn_map, dim=0)
    # 转置维度为 (width, height)
    attn_map = attn_map.permute(1,0)
    # 临时尺寸变量
    temp_size = None

    # 尝试找到合适的缩放因子，使上采样后的尺寸与目标尺寸匹配
    for i in range(0,5):
        scale = 2 ** i
        # 计算缩放后的尺寸是否与目标尺寸匹配
        if ( target_size[0] // scale ) * ( target_size[1] // scale) == attn_map.shape[1]*64:
            temp_size = (target_size[0]//(scale*8), target_size[1]//(scale*8))
            break

    assert temp_size is not None, "temp_size cannot is None"

    # 重塑注意力映射为 (1, temp_height, temp_width)
    attn_map = attn_map.view(attn_map.shape[0], *temp_size)

    # 使用双线性插值进行上采样，输出形状为 (1, target_height, target_width)
    attn_map = F.interpolate(
        attn_map.unsqueeze(0).to(dtype=torch.float32),
        size=target_size,
        mode='bilinear',
        align_corners=False
    )[0]

    # 对上采样后的注意力映射应用softmax函数，使其归一化到 (0,1) 之间
    attn_map = torch.softmax(attn_map, dim=0)
    return attn_map


def get_net_attn_map(image_size, batch_size=2, instance_or_negative=False, detach=True):
    """
    获取网络的跨注意力映射。

    Args:
        image_size (tuple): 目标图像的尺寸，格式为 (height, width)。
        batch_size (int, optional): 批次大小，默认为2。
        instance_or_negative (bool, optional): 如果为True，则选择实例注意力映射；否则选择负面注意力映射，默认为False。
        detach (bool, optional): 是否将注意力映射从计算图中分离，默认为True。

    Returns:
        torch.Tensor: 聚合后的跨注意力映射，形状为 (height, width)。
    """
    # 根据 `instance_or_negative` 参数选择索引，0 表示实例，1 表示负面
    idx = 0 if instance_or_negative else 1
    # 初始化一个空列表，用于存储注意力映射
    net_attn_maps = []

    # 遍历存储在 `attn_maps` 字典中的所有注意力映射
    for name, attn_map in attn_maps.items():
        # 如果 `detach` 为True，则将注意力映射从计算图中分离
        attn_map = attn_map.cpu() if detach else attn_map
        # 将注意力映射按批次大小进行拆分，并选择指定的索引
        attn_map = torch.chunk(attn_map, batch_size)[idx].squeeze()
        # 对注意力映射进行上采样，使其与图像尺寸匹配
        attn_map = upscale(attn_map, image_size) 
        # 将上采样后的注意力映射添加到列表中
        net_attn_maps.append(attn_map) 

    # 对所有注意力映射取平均，得到聚合后的跨注意力映射
    net_attn_maps = torch.mean(torch.stack(net_attn_maps,dim=0),dim=0)

    return net_attn_maps


def attnmaps2images(net_attn_maps):
    """
    将网络生成的注意力映射转换为图像列表。

    Args:
        net_attn_maps (list of torch.Tensor): 网络生成的注意力映射列表，每个张量的形状通常为 (height, width)。

    Returns:
        list of PIL.Image.Image: 转换后的图像列表，每个图像对应一个注意力映射。
    """

    # 初始化一个空列表，用于存储转换后的图像
    images = []

    for attn_map in net_attn_maps:
        # 将注意力映射张量从GPU移到CPU，并转换为NumPy数组
        attn_map = attn_map.cpu().numpy()
        # 计算注意力映射的平均值（可选，用于统计总注意力分数）
        #total_attn_scores += attn_map.mean().item()

        # 归一化注意力映射，使其值范围从0到255
        normalized_attn_map = (attn_map - np.min(attn_map)) / (np.max(attn_map) - np.min(attn_map)) * 255
        # 将归一化后的注意力映射转换为无符号8位整数类型
        normalized_attn_map = normalized_attn_map.astype(np.uint8)
        # 打印归一化后的注意力映射的形状（可选，用于调试）
        #print("norm: ", normalized_attn_map.shape)
        # 将NumPy数组转换为PIL图像
        image = Image.fromarray(normalized_attn_map)

        # 可选：对图像进行进一步处理，例如修复保存格式（此处注释掉）
        #image = fix_save_attn_map(attn_map)

        # 将转换后的图像添加到列表中
        images.append(image)

    #print(total_attn_scores)
    return images


def is_torch2_available():
    """
    检查当前环境中是否可以使用PyTorch 2.0的特性。

    Returns:
        bool: 如果当前PyTorch版本支持 `scaled_dot_product_attention` 函数，则返回True；否则返回False。
    """
    return hasattr(F, "scaled_dot_product_attention")


def get_generator(seed, device):
    """
    获取一个生成器实例，用于设置随机种子。

    Args:
        seed (int 或 list of int, optional): 随机种子。如果为None，则不设置种子。
        device (torch.device): 计算设备（CPU或GPU）。

    Returns:
        torch.Generator 或 list of torch.Generator 或 None: 根据输入参数返回生成器实例或None。
    """

    if seed is not None:
        if isinstance(seed, list):
            # 如果种子是一个列表，则为每个种子创建一个生成器，并返回一个生成器列表
            generator = [torch.Generator(device).manual_seed(seed_item) for seed_item in seed]
        else:
            # 如果种子是一个单独的整数，则创建一个生成器并设置种子
            generator = torch.Generator(device).manual_seed(seed)
    else:
        # 如果没有提供种子，则返回None
        generator = None

    return generator
