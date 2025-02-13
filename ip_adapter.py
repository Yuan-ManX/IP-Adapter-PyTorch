import os
from typing import List

import torch
from diffusers import StableDiffusionPipeline
from diffusers.pipelines.controlnet import MultiControlNetModel
from PIL import Image
from safetensors import safe_open
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

from utils import is_torch2_available, get_generator

if is_torch2_available():
    from .attention_processor import (
        AttnProcessor2_0 as AttnProcessor,
    )
    from .attention_processor import (
        CNAttnProcessor2_0 as CNAttnProcessor,
    )
    from .attention_processor import (
        IPAttnProcessor2_0 as IPAttnProcessor,
    )
else:
    from .attention_processor import AttnProcessor, CNAttnProcessor, IPAttnProcessor
from .resampler import Resampler


# ===========================
# ImageProjModel 类
# ===========================

class ImageProjModel(torch.nn.Module):
    """
    图像投影模型（Projection Model）

    该模型用于将图像嵌入（image embeddings）投影到跨注意力维度（cross-attention dimension），
    并生成额外的上下文标记（extra context tokens），以增强模型在图像相关任务中的表现。

    Args:
        cross_attention_dim (int, optional): 跨注意力机制的维度大小，默认为1024。
        clip_embeddings_dim (int, optional): CLIP图像嵌入的维度大小，默认为1024。
        clip_extra_context_tokens (int, optional): 额外的上下文标记数量，默认为4。
    """

    def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=1024, clip_extra_context_tokens=4):
        super().__init__()

        # 初始化参数
        # 初始化生成器为None（预留接口）
        self.generator = None 
        # 跨注意力维度
        self.cross_attention_dim = cross_attention_dim
        # 额外的上下文标记数量
        self.clip_extra_context_tokens = clip_extra_context_tokens
        # 定义一个线性变换层，用于将CLIP图像嵌入的维度从 clip_embeddings_dim 转换为 clip_extra_context_tokens * cross_attention_dim
        self.proj = torch.nn.Linear(clip_embeddings_dim, self.clip_extra_context_tokens * cross_attention_dim)
        # 定义一个层归一化层，用于对输出进行归一化
        self.norm = torch.nn.LayerNorm(cross_attention_dim)

    def forward(self, image_embeds):
        """
        前向传播方法，执行图像嵌入的投影操作。

        Args:
            image_embeds (torch.Tensor): 输入的图像嵌入，形状为 (batch_size, sequence_length, clip_embeddings_dim)。

        Returns:
            torch.Tensor: 经过投影和归一化后的额外上下文标记，形状为 (batch_size, clip_extra_context_tokens, cross_attention_dim)。
        """
        # 将输入图像嵌入赋值给变量 embeds
        embeds = image_embeds
        # 通过线性变换层将嵌入的维度从 clip_embeddings_dim 转换为 clip_extra_context_tokens * cross_attention_dim
        clip_extra_context_tokens = self.proj(embeds).reshape(
            -1, self.clip_extra_context_tokens, self.cross_attention_dim
        )
        # 对投影后的嵌入进行层归一化
        clip_extra_context_tokens = self.norm(clip_extra_context_tokens)
        # 返回处理后的额外上下文标记
        return clip_extra_context_tokens


# ===========================
# MLPProjModel 类
# ===========================

class MLPProjModel(torch.nn.Module):
    """
    MLP投影模型（MLP Projection Model）

    该模型使用多层感知机（MLP）对图像嵌入进行投影，以适应跨注意力机制的要求。
    它适用于需要将图像提示（image prompt）集成到Stable Diffusion（SD）模型中的场景。

    Args:
        cross_attention_dim (int, optional): 跨注意力机制的维度大小，默认为1024。
        clip_embeddings_dim (int, optional): CLIP图像嵌入的维度大小，默认为1024。
    """

    def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=1024):
        super().__init__()
        
        # 定义一个由多个层组成的MLP，用于投影图像嵌入
        self.proj = torch.nn.Sequential(
            torch.nn.Linear(clip_embeddings_dim, clip_embeddings_dim), # 线性变换，维度保持不变
            torch.nn.GELU(), # 应用GELU激活函数
            torch.nn.Linear(clip_embeddings_dim, cross_attention_dim), # 线性变换，维度转换为跨注意力维度
            torch.nn.LayerNorm(cross_attention_dim) # 对输出进行层归一化
        )
        
    def forward(self, image_embeds):
        """
        前向传播方法，执行图像嵌入的MLP投影操作。

        Args:
            image_embeds (torch.Tensor): 输入的图像嵌入，形状为 (batch_size, sequence_length, clip_embeddings_dim)。

        Returns:
            torch.Tensor: 经过MLP投影后的图像嵌入，形状为 (batch_size, sequence_length, cross_attention_dim)。
        """
        # 通过MLP对图像嵌入进行投影
        clip_extra_context_tokens = self.proj(image_embeds)
        # 返回投影后的图像嵌入
        return clip_extra_context_tokens


# ===========================
# IPAdapter 类
# ===========================

class IPAdapter:
    """
    IP-Adapter 类，用于集成图像提示（Image Prompt）到Stable Diffusion模型中。

    该类通过加载预训练的图像编码器、投影模型和注意力处理器，将图像特征注入到Stable Diffusion的生成过程中，
    从而实现基于图像的生成任务。

    Args:
        sd_pipe: Stable Diffusion的管道（pipeline）对象。
        image_encoder_path (str): 图像编码器的预训练模型路径。
        ip_ckpt (str): IP-Adapter的检查点（checkpoint）文件路径。
        device (torch.device): 计算设备（CPU或GPU）。
        num_tokens (int, optional): 图像特征的上下文长度，默认为4。
    """
    def __init__(self, sd_pipe, image_encoder_path, ip_ckpt, device, num_tokens=4):
        self.device = device
        # 图像编码器的预训练模型路径
        self.image_encoder_path = image_encoder_path
        # IP-Adapter的模型文件路径
        self.ip_ckpt = ip_ckpt
        # 图像特征的上下文长度
        self.num_tokens = num_tokens

        # 将Stable Diffusion管道移动到指定的计算设备
        self.pipe = sd_pipe.to(self.device)
        # 设置IP-Adapter的注意力处理器
        self.set_ip_adapter()

        # 加载图像编码器，并将其移动到计算设备，使用float16精度
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(self.image_encoder_path).to(
            self.device, dtype=torch.float16
        )
        # 初始化CLIP图像处理器，用于预处理图像
        self.clip_image_processor = CLIPImageProcessor()
        # 初始化图像投影模型
        self.image_proj_model = self.init_proj()

        # 加载IP-Adapter的权重
        self.load_ip_adapter()

    def init_proj(self):
        """
        初始化图像投影模型。

        Returns:
            ImageProjModel: 初始化后的图像投影模型实例。
        """
        # 初始化ImageProjModel，设置跨注意力维度和上下文长度
        image_proj_model = ImageProjModel(
            cross_attention_dim=self.pipe.unet.config.cross_attention_dim,
            clip_embeddings_dim=self.image_encoder.config.projection_dim,
            clip_extra_context_tokens=self.num_tokens,
        ).to(self.device, dtype=torch.float16)
        return image_proj_model

    def set_ip_adapter(self):
        """
        设置IP-Adapter的注意力处理器。

        该方法遍历U-Net模型的所有注意力处理器，并根据其位置和类型，替换为自定义的注意力处理器。
        """
        # 获取U-Net模型
        unet = self.pipe.unet
        # 初始化注意力处理器字典
        attn_procs = {}
        for name in unet.attn_processors.keys():
            # 判断是否为跨注意力机制，如果不是，则cross_attention_dim为None
            cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                # 如果是中间块，获取最后一个块的输出通道数作为隐藏层大小
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                # 如果是上采样块，根据块编号获取对应的输出通道数作为隐藏层大小
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                # 如果是下采样块，根据块编号获取对应的输出通道数作为隐藏层大小
                block_id = int(name[len("down_blocks.")])
                hidden_size = unet.config.block_out_channels[block_id]
            if cross_attention_dim is None:
                # 如果不是跨注意力机制，使用默认的注意力处理器
                attn_procs[name] = AttnProcessor()
            else:
                # 否则，使用IPAttnProcessor，并设置相应的参数
                attn_procs[name] = IPAttnProcessor(
                    hidden_size=hidden_size,
                    cross_attention_dim=cross_attention_dim,
                    scale=1.0,
                    num_tokens=self.num_tokens,
                ).to(self.device, dtype=torch.float16)

        # 将自定义的注意力处理器设置到U-Net模型中
        unet.set_attn_processor(attn_procs)
        # 如果管道中包含ControlNet，则设置ControlNet的注意力处理器
        if hasattr(self.pipe, "controlnet"):
            if isinstance(self.pipe.controlnet, MultiControlNetModel):
                for controlnet in self.pipe.controlnet.nets:
                    controlnet.set_attn_processor(CNAttnProcessor(num_tokens=self.num_tokens))
            else:
                self.pipe.controlnet.set_attn_processor(CNAttnProcessor(num_tokens=self.num_tokens))

    def load_ip_adapter(self):
        """
        加载IP-Adapter的权重。

        该方法从指定的检查点文件中加载图像投影模型和注意力处理器的权重。
        """
        if os.path.splitext(self.ip_ckpt)[-1] == ".safetensors":
            # 如果是.safetensors文件，加载状态字典
            state_dict = {"image_proj": {}, "ip_adapter": {}}
            with safe_open(self.ip_ckpt, framework="pt", device="cpu") as f:
                for key in f.keys():
                    if key.startswith("image_proj."):
                        state_dict["image_proj"][key.replace("image_proj.", "")] = f.get_tensor(key)
                    elif key.startswith("ip_adapter."):
                        state_dict["ip_adapter"][key.replace("ip_adapter.", "")] = f.get_tensor(key)
        else:
            # 否则，直接从PyTorch的.pth文件中加载状态字典
            state_dict = torch.load(self.ip_ckpt, map_location="cpu")
        # 加载图像投影模型的权重
        self.image_proj_model.load_state_dict(state_dict["image_proj"])
        # 加载注意力处理器的权重
        ip_layers = torch.nn.ModuleList(self.pipe.unet.attn_processors.values())
        ip_layers.load_state_dict(state_dict["ip_adapter"])

    @torch.inference_mode()
    def get_image_embeds(self, pil_image=None, clip_image_embeds=None):
        """
        获取图像嵌入。

        Args:
            pil_image (PIL.Image.Image or list of PIL.Image.Image, optional): 输入的PIL图像。
            clip_image_embeds (torch.Tensor, optional): 预计算的CLIP图像嵌入。

        Returns:
            tuple: 包含图像提示嵌入和无条件图像提示嵌入的元组。
        """
        if pil_image is not None:
            if isinstance(pil_image, Image.Image):
                # 如果是单个图像，转换为列表
                pil_image = [pil_image]
            # 使用CLIP图像处理器预处理图像，并获取像素值
            clip_image = self.clip_image_processor(images=pil_image, return_tensors="pt").pixel_values
            # 使用图像编码器获取图像嵌入
            clip_image_embeds = self.image_encoder(clip_image.to(self.device, dtype=torch.float16)).image_embeds
        else:
            # 如果提供了预计算的CLIP图像嵌入，则直接使用
            clip_image_embeds = clip_image_embeds.to(self.device, dtype=torch.float16)
        # 通过图像投影模型获取图像提示嵌入
        image_prompt_embeds = self.image_proj_model(clip_image_embeds)
        # 生成无条件图像提示嵌入（使用零张量）
        uncond_image_prompt_embeds = self.image_proj_model(torch.zeros_like(clip_image_embeds))
        return image_prompt_embeds, uncond_image_prompt_embeds

    def set_scale(self, scale):
        """
        设置IP-Adapter的缩放因子。

        Args:
            scale (float): 缩放因子。
        """
        for attn_processor in self.pipe.unet.attn_processors.values():
            if isinstance(attn_processor, IPAttnProcessor):
                attn_processor.scale = scale

    def generate(
        self,
        pil_image=None,
        clip_image_embeds=None,
        prompt=None,
        negative_prompt=None,
        scale=1.0,
        num_samples=4,
        seed=None,
        guidance_scale=7.5,
        num_inference_steps=30,
        **kwargs,
    ):
        """
        生成图像。

        Args:
            pil_image (PIL.Image.Image or list of PIL.Image.Image, optional): 输入的PIL图像。
            clip_image_embeds (torch.Tensor, optional): 预计算的CLIP图像嵌入。
            prompt (str or list of str, optional): 文本提示。
            negative_prompt (str or list of str, optional): 负面文本提示。
            scale (float, optional): 图像提示的缩放因子，默认为1.0。
            num_samples (int, optional): 生成样本的数量，默认为4。
            seed (int, optional): 随机种子。
            guidance_scale (float, optional): 指导比例，默认为7.5。
            num_inference_steps (int, optional): 推理步数，默认为30。

        Returns:
            list of PIL.Image.Image: 生成的图像列表。
        """
        # 设置缩放因子
        self.set_scale(scale)

        if pil_image is not None:
            # 如果提供了PIL图像，确定提示数量
            num_prompts = 1 if isinstance(pil_image, Image.Image) else len(pil_image)
        else:
            # 如果提供了预计算的CLIP图像嵌入，确定提示数量
            num_prompts = clip_image_embeds.size(0)

        if prompt is None:
            # 如果没有提供提示，使用默认提示
            prompt = "best quality, high quality"
        if negative_prompt is None:
            # 如果没有提供负面提示，使用默认负面提示
            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

        if not isinstance(prompt, List):
            # 如果提示不是列表，则转换为列表
            prompt = [prompt] * num_prompts
        if not isinstance(negative_prompt, List):
            # 如果负面提示不是列表，则转换为列表
            negative_prompt = [negative_prompt] * num_prompts

        # 获取图像提示嵌入和无条件图像提示嵌入
        image_prompt_embeds, uncond_image_prompt_embeds = self.get_image_embeds(
            pil_image=pil_image, clip_image_embeds=clip_image_embeds
        )
        bs_embed, seq_len, _ = image_prompt_embeds.shape
        # 重复图像提示嵌入以匹配样本数量
        image_prompt_embeds = image_prompt_embeds.repeat(1, num_samples, 1)
        image_prompt_embeds = image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)
        # 重复无条件图像提示嵌入以匹配样本数量
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.repeat(1, num_samples, 1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)

        with torch.inference_mode():
            # 对提示进行编码
            prompt_embeds_, negative_prompt_embeds_ = self.pipe.encode_prompt(
                prompt,
                device=self.device,
                num_images_per_prompt=num_samples,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )
            # 合并图像提示嵌入和文本提示嵌入
            prompt_embeds = torch.cat([prompt_embeds_, image_prompt_embeds], dim=1)
            # 合并无条件图像提示嵌入和文本负面提示嵌入
            negative_prompt_embeds = torch.cat([negative_prompt_embeds_, uncond_image_prompt_embeds], dim=1)

        # 获取生成器
        generator = get_generator(seed, self.device)

        # 使用Stable Diffusion管道生成图像
        images = self.pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
            **kwargs,
        ).images

        return images


# ===========================
# IPAdapterXL 类（继承自 IPAdapter）
# ===========================

class IPAdapterXL(IPAdapter):
    """
    针对 Stable Diffusion XL (SDXL) 的 IP-Adapter。

    该类扩展了 IPAdapter 类，专门针对 SDXL 模型进行了优化和调整。
    它实现了基于图像提示的生成功能，并支持更复杂的生成流程。

    Args:
        sd_pipe: Stable Diffusion XL 的管道（pipeline）对象。
        image_encoder_path (str): 图像编码器的预训练模型路径。
        ip_ckpt (str): IP-Adapter 的检查点（checkpoint）文件路径。
        device (torch.device): 计算设备（CPU 或 GPU）。
        num_tokens (int, optional): 图像特征的上下文长度，默认为4。
    """

    def generate(
        self,
        pil_image,
        prompt=None,
        negative_prompt=None,
        scale=1.0,
        num_samples=4,
        seed=None,
        num_inference_steps=30,
        **kwargs,
    ):
        """
        生成图像。

        该方法针对 SDXL 模型进行了优化，支持更复杂的生成流程。

        Args:
            pil_image (PIL.Image.Image 或 list of PIL.Image.Image): 输入的 PIL 图像。
            prompt (str 或 list of str, optional): 文本提示。
            negative_prompt (str 或 list of str, optional): 负面文本提示。
            scale (float, optional): 图像提示的缩放因子，默认为1.0。
            num_samples (int, optional): 生成样本的数量，默认为4。
            seed (int, optional): 随机种子。
            num_inference_steps (int, optional): 推理步数，默认为30。

        Returns:
            list of PIL.Image.Image: 生成的图像列表。
        """
        # 设置缩放因子
        self.set_scale(scale)

        # 确定提示数量
        num_prompts = 1 if isinstance(pil_image, Image.Image) else len(pil_image)

        if prompt is None:
            # 如果没有提供提示，使用默认提示
            prompt = "best quality, high quality"
        if negative_prompt is None:
            # 如果没有提供负面提示，使用默认负面提示
            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

        # 如果提示不是列表，则转换为列表
        if not isinstance(prompt, List):
            prompt = [prompt] * num_prompts
        if not isinstance(negative_prompt, List):
            negative_prompt = [negative_prompt] * num_prompts

        # 获取图像提示嵌入和无条件图像提示嵌入
        image_prompt_embeds, uncond_image_prompt_embeds = self.get_image_embeds(pil_image)
        bs_embed, seq_len, _ = image_prompt_embeds.shape
        # 重复图像提示嵌入以匹配样本数量
        image_prompt_embeds = image_prompt_embeds.repeat(1, num_samples, 1)
        image_prompt_embeds = image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)
        # 重复无条件图像提示嵌入以匹配样本数量
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.repeat(1, num_samples, 1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)

        with torch.inference_mode():
            # 对提示进行编码，包括文本提示和图像提示
            (
                prompt_embeds,
                negative_prompt_embeds,
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
            ) = self.pipe.encode_prompt(
                prompt,
                num_images_per_prompt=num_samples,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )
            # 将图像提示嵌入与文本提示嵌入合并
            prompt_embeds = torch.cat([prompt_embeds, image_prompt_embeds], dim=1)
            # 将无条件图像提示嵌入与文本负面提示嵌入合并
            negative_prompt_embeds = torch.cat([negative_prompt_embeds, uncond_image_prompt_embeds], dim=1)

        # 获取生成器
        self.generator = get_generator(seed, self.device)
        
        # 使用 Stable Diffusion XL 管道生成图像
        images = self.pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            num_inference_steps=num_inference_steps,
            generator=self.generator,
            **kwargs,
        ).images

        return images


# ===========================
# IPAdapterPlus 类（继承自 IPAdapter）
# ===========================

class IPAdapterPlus(IPAdapter):
    """
    带有细粒度特征的 IP-Adapter。

    该类扩展了 IPAdapter 类，使用 Resampler 模块来增强图像特征的细粒度处理能力。

    Args:
        sd_pipe: Stable Diffusion 的管道（pipeline）对象。
        image_encoder_path (str): 图像编码器的预训练模型路径。
        ip_ckpt (str): IP-Adapter 的检查点（checkpoint）文件路径。
        device (torch.device): 计算设备（CPU 或 GPU）。
        num_tokens (int, optional): 图像特征的上下文长度，默认为4。
    """

    def init_proj(self):
        """
        初始化投影模型。

        使用 Resampler 模块来增强图像特征的细粒度处理能力。

        Returns:
            Resampler: 初始化后的 Resampler 实例。
        """
        # 初始化 Resampler，设置相关参数，并移动到设备，使用 float16 精度
        image_proj_model = Resampler(
            dim=self.pipe.unet.config.cross_attention_dim,
            depth=4,
            dim_head=64,
            heads=12,
            num_queries=self.num_tokens,
            embedding_dim=self.image_encoder.config.hidden_size,
            output_dim=self.pipe.unet.config.cross_attention_dim,
            ff_mult=4,
        ).to(self.device, dtype=torch.float16)
        return image_proj_model

    @torch.inference_mode()
    def get_image_embeds(self, pil_image=None, clip_image_embeds=None):
        """
        获取图像嵌入。

        使用 Resampler 模块对图像特征进行细粒度处理。

        Args:
            pil_image (PIL.Image.Image 或 list of PIL.Image.Image, optional): 输入的 PIL 图像。
            clip_image_embeds (torch.Tensor, optional): 预计算的 CLIP 图像嵌入。

        Returns:
            tuple: 包含图像提示嵌入和无条件图像提示嵌入的元组。
        """
        if isinstance(pil_image, Image.Image):
            # 如果是单个图像，转换为列表
            pil_image = [pil_image]

        # 使用 CLIP 图像处理器预处理图像，并获取像素值
        clip_image = self.clip_image_processor(images=pil_image, return_tensors="pt").pixel_values
        clip_image = clip_image.to(self.device, dtype=torch.float16)
        # 使用图像编码器获取隐藏状态，并选择倒数第二个隐藏状态作为图像嵌入
        clip_image_embeds = self.image_encoder(clip_image, output_hidden_states=True).hidden_states[-2]
        # 使用 Resampler 对图像嵌入进行细粒度处理
        image_prompt_embeds = self.image_proj_model(clip_image_embeds)
        # 生成无条件图像提示嵌入（使用零张量）
        uncond_clip_image_embeds = self.image_encoder(
            torch.zeros_like(clip_image), output_hidden_states=True
        ).hidden_states[-2]
        uncond_image_prompt_embeds = self.image_proj_model(uncond_clip_image_embeds)
        return image_prompt_embeds, uncond_image_prompt_embeds


class IPAdapterFull(IPAdapterPlus):
    """
    完整功能的 IP-Adapter。

    该类扩展了 IPAdapterPlus 类，旨在提供更全面的功能集，包括使用 MLPProjModel 进行投影处理，
    以实现更复杂或更精细的图像特征处理。

    Args:
        sd_pipe: Stable Diffusion 的管道（pipeline）对象。
        image_encoder_path (str): 图像编码器的预训练模型路径。
        ip_ckpt (str): IP-Adapter 的检查点（checkpoint）文件路径。
        device (torch.device): 计算设备（CPU 或 GPU）。
        num_tokens (int, optional): 图像特征的上下文长度，默认为4。
    """

    def init_proj(self):
        """
        初始化投影模型。

        使用 MLPProjModel 进行投影处理，该模型通过多层感知机（MLP）将图像嵌入投影到跨注意力维度。

        Returns:
            MLPProjModel: 初始化后的 MLPProjModel 实例。
        """
        # 初始化 MLPProjModel，设置跨注意力维度和 CLIP 图像嵌入的维度
        image_proj_model = MLPProjModel(
            cross_attention_dim=self.pipe.unet.config.cross_attention_dim,
            clip_embeddings_dim=self.image_encoder.config.hidden_size,
        ).to(self.device, dtype=torch.float16)
        return image_proj_model


# ===========================
# IPAdapterPlusXL 类（继承自 IPAdapter）
# ===========================

class IPAdapterPlusXL(IPAdapter):
    """
    针对 Stable Diffusion XL (SDXL) 的增强型 IP-Adapter。

    该类扩展了 IPAdapter 类，专为 SDXL 模型设计，采用了更大规模的 Resampler 模型以处理更高维度的图像特征。

    Args:
        sd_pipe: Stable Diffusion XL 的管道（pipeline）对象。
        image_encoder_path (str): 图像编码器的预训练模型路径。
        ip_ckpt (str): IP-Adapter 的检查点（checkpoint）文件路径。
        device (torch.device): 计算设备（CPU 或 GPU）。
        num_tokens (int, optional): 图像特征的上下文长度，默认为4。
    """

    def init_proj(self):
        """
        初始化投影模型。

        使用 Resampler 模型，配置更高维度的参数以适应 SDXL 的需求。

        Returns:
            Resampler: 初始化后的 Resampler 实例。
        """
        # 初始化 Resampler，设置相关参数，并移动到设备，使用 float16 精度
        image_proj_model = Resampler(
            dim=1280, # 输入和输出的维度大小
            depth=4, # 注意力层和前馈层的堆叠深度
            dim_head=64, # 每个注意力头的维度大小
            heads=20, # 多头注意力的头数
            num_queries=self.num_tokens, # 查询（queries）的数量
            embedding_dim=self.image_encoder.config.hidden_size, # 图像编码器的嵌入维度
            output_dim=self.pipe.unet.config.cross_attention_dim, # 输出维度
            ff_mult=4, # 前馈层内部维度相对于输入维度的倍数
        ).to(self.device, dtype=torch.float16)
        return image_proj_model

    @torch.inference_mode()
    def get_image_embeds(self, pil_image):
        """
        获取图像嵌入。

        使用 Resampler 模型对图像特征进行投影处理。

        Args:
            pil_image (PIL.Image.Image 或 list of PIL.Image.Image): 输入的 PIL 图像。

        Returns:
            tuple: 包含图像提示嵌入和无条件图像提示嵌入的元组。
        """
        if isinstance(pil_image, Image.Image):
            # 如果是单个图像，转换为列表
            pil_image = [pil_image]
        # 使用 CLIP 图像处理器预处理图像，并获取像素值
        clip_image = self.clip_image_processor(images=pil_image, return_tensors="pt").pixel_values
        clip_image = clip_image.to(self.device, dtype=torch.float16)
        # 使用图像编码器获取隐藏状态，并选择倒数第二个隐藏状态作为图像嵌入
        clip_image_embeds = self.image_encoder(clip_image, output_hidden_states=True).hidden_states[-2]
        # 使用 Resampler 对图像嵌入进行投影处理
        image_prompt_embeds = self.image_proj_model(clip_image_embeds)
        # 生成无条件图像提示嵌入（使用零张量）
        uncond_clip_image_embeds = self.image_encoder(
            torch.zeros_like(clip_image), output_hidden_states=True
        ).hidden_states[-2]
        uncond_image_prompt_embeds = self.image_proj_model(uncond_clip_image_embeds)
        return image_prompt_embeds, uncond_image_prompt_embeds

    def generate(
        self,
        pil_image,
        prompt=None,
        negative_prompt=None,
        scale=1.0,
        num_samples=4,
        seed=None,
        num_inference_steps=30,
        **kwargs,
    ):
        """
        生成图像。

        该方法针对 SDXL 模型进行了优化，支持更复杂的生成流程。

        Args:
            pil_image (PIL.Image.Image 或 list of PIL.Image.Image): 输入的 PIL 图像。
            prompt (str 或 list of str, optional): 文本提示。
            negative_prompt (str 或 list of str, optional): 负面文本提示。
            scale (float, optional): 图像提示的缩放因子，默认为1.0。
            num_samples (int, optional): 生成样本的数量，默认为4。
            seed (int, optional): 随机种子。
            num_inference_steps (int, optional): 推理步数，默认为30。

        Returns:
            list of PIL.Image.Image: 生成的图像列表。
        """
        # 设置缩放因子
        self.set_scale(scale)

        # 确定提示数量
        num_prompts = 1 if isinstance(pil_image, Image.Image) else len(pil_image)

        if prompt is None:
            # 如果没有提供提示，使用默认提示
            prompt = "best quality, high quality"
        if negative_prompt is None:
            # 如果没有提供负面提示，使用默认负面提示
            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

        # 如果提示不是列表，则转换为列表
        if not isinstance(prompt, List):
            prompt = [prompt] * num_prompts
        if not isinstance(negative_prompt, List):
            negative_prompt = [negative_prompt] * num_prompts

        # 获取图像提示嵌入和无条件图像提示嵌入
        image_prompt_embeds, uncond_image_prompt_embeds = self.get_image_embeds(pil_image)
        bs_embed, seq_len, _ = image_prompt_embeds.shape
        # 重复图像提示嵌入以匹配样本数量
        image_prompt_embeds = image_prompt_embeds.repeat(1, num_samples, 1)
        image_prompt_embeds = image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)
        # 重复无条件图像提示嵌入以匹配样本数量
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.repeat(1, num_samples, 1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)

        with torch.inference_mode():
            # 对提示进行编码，包括文本提示和图像提示
            (
                prompt_embeds,
                negative_prompt_embeds,
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
            ) = self.pipe.encode_prompt(
                prompt,
                num_images_per_prompt=num_samples,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )
            # 将图像提示嵌入与文本提示嵌入合并
            prompt_embeds = torch.cat([prompt_embeds, image_prompt_embeds], dim=1)
            # 将无条件图像提示嵌入与文本负面提示嵌入合并
            negative_prompt_embeds = torch.cat([negative_prompt_embeds, uncond_image_prompt_embeds], dim=1)

        # 获取生成器
        generator = get_generator(seed, self.device)

        # 使用 Stable Diffusion XL 管道生成图像
        images = self.pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            num_inference_steps=num_inference_steps,
            generator=generator,
            **kwargs,
        ).images

        return images
