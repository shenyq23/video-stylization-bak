from causvid.models import (
    get_diffusion_wrapper,
    get_text_encoder_wrapper,
    get_vae_wrapper
)
from typing import List
import torch
import torch.distributed as dist
import time

class CausalStreamInferencePipeline(torch.nn.Module):
    def __init__(self, args, device, text_encoder_on_cpu: bool = False, use_cached_text_embedding: bool = False):
        super().__init__()
        model_type = args.model_type
        self.device = device
        self.text_encoder_on_cpu = text_encoder_on_cpu
        # <--- NEW CODE START --->
        self.use_cached_text_embedding = use_cached_text_embedding
        # <--- NEW CODE END --->
        # Step 1: Initialize all models
        self.generator_model_name = getattr(
            args, "generator_name", args.model_name)
        self.generator = get_diffusion_wrapper(
            model_name=self.generator_model_name)(model_type=model_type)

        self.text_encoder = None
        if not self.use_cached_text_embedding:
            self.text_encoder = get_text_encoder_wrapper(
                model_name=args.model_name)(model_type=model_type)
        else:
            print("Skipping Text Encoder initialization, will use cached embedding.")

        self.vae = get_vae_wrapper(model_name=args.model_name)(model_type=model_type)

        # Step 2: Initialize all causal hyperparmeters
        self._init_denoising_step_list(args, device)

        if model_type == "T2V-1.3B":
            self.num_transformer_blocks = 30
            self.num_heads = 12
        elif model_type == "T2V-14B":
            self.num_transformer_blocks = 40
            self.num_heads = 40
        else:
            raise ValueError(f"Model type {model_type} not supported")
        scale_size = 16
        self.height = args.height//scale_size*2
        self.width = args.width//scale_size*2
        self.frame_seq_length = (args.height//scale_size) * (args.width//scale_size)
        self.kv_cache_length = self.frame_seq_length*args.num_kv_cache
        self.num_sink_tokens = args.num_sink_tokens
        self.adapt_sink_threshold = args.adapt_sink_threshold

        self.conditional_dict = None
        self.kv_cache1 = None
        self.kv_cache2 = None
        self.hidden_states = None
        self.block_x = None
        self.args = args
        self.num_frame_per_block = getattr(
            args, "num_frame_per_block", 1)

        print(f"KV inference with {self.num_frame_per_block} frames per block")

        if self.num_frame_per_block > 1:
            self.generator.model.num_frame_per_block = self.num_frame_per_block

        # self.generator.model.to(self.device)
        self.flow_guidance_cache = None

    def to(self, device=None, dtype=None, non_blocking=False):
        """
        Overrides the default .to() method to handle CPU offloading for the text_encoder.
        """
        # 如果 device 未指定，则使用初始化时的 device
        target_device = device if device is not None else self.device

        # A. 将 generator 和 vae 移动到目标 GPU 设备
        self.generator.to(device=target_device, dtype=dtype, non_blocking=non_blocking)
        self.vae.to(device=target_device, dtype=dtype, non_blocking=non_blocking)

        # B. 根据标志决定 text_encoder 的位置
        if self.text_encoder is not None:
            if self.text_encoder_on_cpu:
                # 确保 text_encoder 在 CPU 上，并使用 float32 (CPU 通常不支持 bfloat16)
                self.text_encoder.to(device='cpu', dtype=torch.float32, non_blocking=non_blocking)
                print("Text encoder is intentionally kept on CPU.")
            else:
                # 正常移动到 GPU
                self.text_encoder.to(device=target_device, dtype=dtype, non_blocking=non_blocking)

        # 更新 pipeline 的主设备属性
        self.device = target_device

        # 返回 self 以支持链式调用
        return self

    def _init_denoising_step_list(self, args, device):
        self.denoising_step_list = torch.tensor(
            args.denoising_step_list, dtype=torch.long, device=device)
        assert self.denoising_step_list[-1] == 0
        # remove the last timestep (which equals zero)
        self.denoising_step_list = self.denoising_step_list[:-1]

        self.scheduler = self.generator.get_scheduler()
        if args.warp_denoising_step:  # Warp the denoising step according to the scheduler time shift
            timesteps = torch.cat((self.scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32))).cuda()
            self.denoising_step_list = timesteps[1000 - self.denoising_step_list]

    def _initialize_kv_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU KV cache for the Wan model.
        """
        kv_cache1 = []

        for i in range(self.num_transformer_blocks):
            cache_length = self.kv_cache_length
            self.generator.model.blocks[i].self_attn.sink_size = self.num_sink_tokens
            self.generator.model.blocks[i].self_attn.adapt_sink_thr = self.adapt_sink_threshold

            kv_cache1.append({
                "k": torch.zeros([batch_size, cache_length, self.num_heads, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, cache_length, self.num_heads, 128], dtype=dtype, device=device),
                # "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                # "local_end_index": torch.tensor([0], dtype=torch.long, device=device),
            })

        self.kv_cache1 = kv_cache1  # always store the clean cache

    def _initialize_crossattn_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU cross-attention cache for the Wan model.
        """
        crossattn_cache = []

        for _ in range(self.num_transformer_blocks):
            crossattn_cache.append({
                "k": torch.zeros([batch_size, 512, self.num_heads, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, 512, self.num_heads, 128], dtype=dtype, device=device),
                "is_init": False,
            })

        self.crossattn_cache = crossattn_cache  # always store the clean cache

    def prepare(
        self,
        text_prompts: List[str],
        device: torch.device,
        dtype: torch.dtype,
        block_mode: str='input',
        noise: torch.Tensor = None,
        current_start: int = 0,
        current_end: int = None,
        block_num: torch.Tensor = None,
        batch_denoise: bool=True,
    ):
        self.device = device
        batch_size = noise.shape[0]

        cached_embedding_path = f"./text_cache/cached_text_embedding_{text_prompts[0][:20]}.pt"

        if self.use_cached_text_embedding:
            print(f"Attempting to load cached text embedding from '{cached_embedding_path}'...")
            self.conditional_dict = torch.load(cached_embedding_path, map_location="cpu")
            # Move the tensor to the correct device and dtype
            self.conditional_dict["prompt_embeds"] = self.conditional_dict["prompt_embeds"].to(device=self.device, dtype=dtype)
            print("Successfully loaded and prepared cached text embedding.")
        else:
            # Original logic to compute and then save the embedding
            if self.text_encoder_on_cpu:
                conditional_dict_cpu = self.text_encoder(text_prompts=text_prompts)
                self.conditional_dict = {
                    "prompt_embeds": conditional_dict_cpu["prompt_embeds"].to(device=self.device, dtype=dtype)
                }
            else:
                self.conditional_dict = self.text_encoder(text_prompts=text_prompts)

            dict_to_save = {"prompt_embeds": self.conditional_dict["prompt_embeds"].cpu()}
            torch.save(dict_to_save, cached_embedding_path)
            print(f"Computed and saved text embedding to '{cached_embedding_path}'.")

        # Step 1: Initialize KV cache
        if self.kv_cache1 is None:
            self._initialize_kv_cache(
                batch_size=batch_size,
                dtype=dtype,
                device=device
            )

            self._initialize_crossattn_cache(
                batch_size=batch_size,
                dtype=dtype,
                device=device
            )
            self.flow_guidance_cache=torch.zeros([self.num_transformer_blocks,len(self.denoising_step_list), self.frame_seq_length, self.num_heads*128], dtype=dtype, device=device)
        else:
            # reset cross attn cache
            for block_index in range(self.num_transformer_blocks):
                self.crossattn_cache[block_index]["is_init"] = False

        current_start = torch.tensor([current_start], dtype=torch.long, device=device)
        # current_end = torch.tensor([current_end], dtype=torch.long, device=device)

        for index, current_timestep in enumerate(self.denoising_step_list):
            # set current timestep
            timestep = torch.ones(
                [batch_size, noise.shape[1]], device=self.device, dtype=torch.int64) * current_timestep

            if index < len(self.denoising_step_list) - 1:
                denoised_pred = self.generator(
                    noisy_image_or_video=noise,
                    conditional_dict=self.conditional_dict,
                    timestep=timestep,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start,
                    # current_end=current_end
                )
                next_timestep = self.denoising_step_list[index + 1]
                noise = self.scheduler.add_noise(
                    denoised_pred.flatten(0, 1),
                    torch.randn_like(denoised_pred.flatten(0, 1)),
                    next_timestep *
                    torch.ones([batch_size], device=device,
                                dtype=torch.long)
                ).unflatten(0, denoised_pred.shape[:2])
            else:
                # for getting real output
                denoised_pred = self.generator(
                    noisy_image_or_video=noise,
                    conditional_dict=self.conditional_dict,
                    timestep=timestep,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start,
                    # current_end=current_end
                )
            # print(noise.shape,current_start)
            # for idx in range (len(noise[0])): print("hidden idx",idx,torch.mean(noise[0][idx]))
            # for idx in range (6): print("idx",idx,torch.mean(
            #     self.kv_cache1[0]['k'][:,idx*self.frame_seq_length:(idx+1)*self.frame_seq_length,0,0],dim=1))

        if not batch_denoise:
            return denoised_pred

        # Pre-allocate hidden_states tensor to avoid memory allocation during inference
        self.batch_size = len(self.denoising_step_list)

        # Determine which blocks to keep based on block_num range
        blocks_to_keep = []
        if block_num is not None:
            start_block, end_block = block_num[0].item(), block_num[1].item()
            blocks_to_keep = list(range(start_block, end_block))
        else:
            blocks_to_keep = list(range(self.num_transformer_blocks))

        # Process only the blocks in the specified range
        for i in range(self.num_transformer_blocks):
            if dist.is_initialized():
                dist.broadcast(self.crossattn_cache[i]['k'], src=0)
                dist.broadcast(self.crossattn_cache[i]['v'], src=0)
                dist.broadcast(self.kv_cache1[i]['k'], src=0)
                dist.broadcast(self.kv_cache1[i]['v'], src=0)

            self.kv_cache1[i]['k'] = self.kv_cache1[i]['k'].repeat(self.batch_size, 1, 1, 1)
            self.kv_cache1[i]['v'] = self.kv_cache1[i]['v'].repeat(self.batch_size, 1, 1, 1)

            # self.kv_cache1[i]['global_end_index'] = self.kv_cache1[i]['global_end_index'].repeat(self.batch_size)
            # self.kv_cache1[i]['local_end_index'] = self.kv_cache1[i]['local_end_index'].repeat(self.batch_size)

            self.crossattn_cache[i]['k'] = self.crossattn_cache[i]['k'].repeat(self.batch_size, 1, 1, 1)
            self.crossattn_cache[i]['v'] = self.crossattn_cache[i]['v'].repeat(self.batch_size, 1, 1, 1)

        # Remove blocks outside the range
        if block_num is not None:
            for i in range(self.num_transformer_blocks):
                if i not in blocks_to_keep:
                    self.kv_cache1[i]['k'] = self.kv_cache1[i]['k'].cpu()
                    self.kv_cache1[i]['v'] = self.kv_cache1[i]['v'].cpu()

        self.hidden_states = torch.zeros(
            (self.batch_size, self.num_frame_per_block, *noise.shape[2:]), dtype=noise.dtype, device=device
        )
        self.latent_flow_data = {}
        self.latent_flow_data['flow']=torch.zeros(
            (self.batch_size, 2, self.height//2,self.width//2), dtype= torch.float32, device=device
        )
        self.latent_flow_data['mask']=torch.zeros(
            (self.batch_size, 1, self.height//2,self.width//2), dtype= torch.bool, device=device
        )

        if block_mode in ['output', 'middle']:
            self.block_x = torch.zeros(
                (self.batch_size, self.frame_seq_length, self.num_heads*128), dtype=noise.dtype, device=device
            )
        else:
            self.block_x = None

        self.kv_cache_starts = torch.ones(self.batch_size, dtype=torch.long, device=device) * current_end
        # self.kv_cache_ends = torch.ones(self.batch_size, dtype=torch.long, device=device) * current_end + self.frame_seq_length

        self.timestep = self.denoising_step_list

        self.conditional_dict['prompt_embeds'] = self.conditional_dict['prompt_embeds'].repeat(self.batch_size, 1, 1)

        return denoised_pred

    def inference_stream(self, noise: torch.Tensor, current_start: int, current_end: int, current_step: int,latent_flow_data=None) -> torch.Tensor:
        # print(self.hidden_states.dtype,self.hidden_states.shape,self.kv_cache_starts.dtype,self.kv_cache_starts.shape,noise.shape)
        #torch.cuda.synchronize(device=self.device)
        inference_stream_start_time= time.time()

        self.hidden_states[1:] = self.hidden_states[:-1].clone()
        self.hidden_states[0] = noise[0]
        self.kv_cache_starts[1:] = self.kv_cache_starts[:-1].clone()
        self.kv_cache_starts[0] = current_start

        if (latent_flow_data!=None):
            # print("test dtype",self.latent_flow_data['flow'].dtype,latent_flow_data[0].dtype,self.latent_flow_data['mask'].dtype,latent_flow_data[1].dtype)
            self.latent_flow_data['flow'][1:] = self.latent_flow_data['flow'][:-1].clone()
            self.latent_flow_data['flow'][0] = latent_flow_data[0].squeeze(0)
            self.latent_flow_data['mask'][1:] = self.latent_flow_data['mask'][:-1].clone()
            self.latent_flow_data['mask'][0] = latent_flow_data[1].squeeze(0)


        # #torch.cuda.synchronize(device=self.device)
        # clone_end_time= time.time()
        # print(f"Clone time: {clone_end_time - inference_stream_start_time} seconds")

        if current_step is not None: self.timestep[0] = current_step

        self.hidden_states = self.generator(
            noisy_image_or_video=self.hidden_states,
            conditional_dict=self.conditional_dict,
            timestep=self.timestep.unsqueeze(1).expand(-1, self.hidden_states.shape[1]),
            kv_cache=self.kv_cache1,
            crossattn_cache=self.crossattn_cache,
            current_start=self.kv_cache_starts,
            flow_guidance_cache=self.flow_guidance_cache,
            latent_flow_data=None if latent_flow_data==None else self.latent_flow_data,
            # current_end=self.kv_cache_ends,
        )
        #torch.cuda.synchronize(device=self.device)
        generator_end_time= time.time()
        # print(f"Generator time: {generator_end_time - inference_stream_start_time} seconds")

        for i in range(len(self.denoising_step_list) - 1):
            self.hidden_states[[i]] = self.scheduler.add_noise(
                self.hidden_states[[i]],
                torch.randn_like(self.hidden_states[[i]]),
                self.denoising_step_list[i + 1] *
                torch.ones([1], device=self.device,
                            dtype=torch.long)
            )
        #torch.cuda.synchronize(device=self.device)
        add_noise_end_time= time.time()
        # print(f"Add noise time: {add_noise_end_time - generator_end_time} seconds")

        # print(self.kv_cache_starts)
        # for idx in range (6): print("kv idx",idx,torch.mean(self.kv_cache1[0]['k'][:,idx*self.frame_seq_length:(idx+1)*self.frame_seq_length,0,0],dim=1))
        # for idx in range (len(self.hidden_states)): print("hidden idx",idx,torch.mean(self.hidden_states[idx]))
        # print_kv_idx_end_time= time.time()
        # print(f"Print kv idx time: {print_kv_idx_end_time - add_noise_end_time} seconds")
        return self.hidden_states

    def inference_wo_batch(self, noise: torch.Tensor, current_start: int, current_end: int, current_step: int) -> torch.Tensor:
        batch_size = noise.shape[0]

        current_start = torch.ones(batch_size, dtype=torch.long, device=self.device) * current_start
        current_end = torch.ones(batch_size, dtype=torch.long, device=self.device) * current_end

        # Step 2.1: Spatial denoising loop
        self.denoising_step_list[0] = current_step
        for index, current_timestep in enumerate(self.denoising_step_list):
            # set current timestep
            timestep = torch.ones(
                [batch_size, noise.shape[1]], device=self.device, dtype=torch.int64) * current_timestep

            if index < len(self.denoising_step_list) - 1:
                denoised_pred = self.generator(
                    noisy_image_or_video=noise,
                    conditional_dict=self.conditional_dict,
                    timestep=timestep,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start,
                    current_end=current_end
                )
                next_timestep = self.denoising_step_list[index + 1]
                noise = self.scheduler.add_noise(
                    denoised_pred.flatten(0, 1),
                    torch.randn_like(denoised_pred.flatten(0, 1)),
                    next_timestep *
                    torch.ones([batch_size], device=self.device,
                                dtype=torch.long)
                ).unflatten(0, denoised_pred.shape[:2])
            else:
                # for getting real output
                denoised_pred = self.generator(
                    noisy_image_or_video=noise,
                    conditional_dict=self.conditional_dict,
                    timestep=timestep,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start,
                    current_end=current_end
                )

        return denoised_pred

    def inference(self, noise: torch.Tensor, current_start: int, current_end: int, \
        current_step: int, block_mode: str='input', block_num=None,\
            patched_x_shape: torch.Tensor=None, block_x: torch.Tensor=None) -> torch.Tensor:

        if block_mode == 'input':
            self.hidden_states[1:] = self.hidden_states[:-1].clone()
            self.hidden_states[0] = noise[0]

            self.kv_cache_starts[1:] = self.kv_cache_starts[:-1].clone()
            self.kv_cache_starts[0] = current_start

            self.kv_cache_ends[1:] = self.kv_cache_ends[:-1].clone()
            self.kv_cache_ends[0] = current_end
        else:
            self.block_x.copy_(block_x)
            self.hidden_states.copy_(noise)
            self.kv_cache_starts.copy_(current_start)
            self.kv_cache_ends.copy_(current_end)

        if current_step is not None:
            self.timestep[0] = current_step

        if block_mode == 'output':
            denoised_pred = self.generator.forward_output(
                noisy_image_or_video=self.hidden_states,
                conditional_dict=self.conditional_dict,
                timestep=self.timestep.unsqueeze(1).expand(-1, self.hidden_states.shape[1]),
                kv_cache=self.kv_cache1,
                crossattn_cache=self.crossattn_cache,
                current_start=self.kv_cache_starts,
                current_end=self.kv_cache_ends,
                block_mode=block_mode,
                block_num=block_num,
                patched_x_shape=patched_x_shape,
                block_x=self.block_x
            )

            for i in range(len(self.denoising_step_list) - 1):
                denoised_pred[[i]] = self.scheduler.add_noise(
                    denoised_pred[[i]],
                    torch.randn_like(denoised_pred[[i]]),
                    self.denoising_step_list[i + 1] *
                    torch.ones([1], device=self.device,
                                dtype=torch.long)
                )
            patched_x_shape = None

        else:
            denoised_pred, patched_x_shape = self.generator.forward_input(
                noisy_image_or_video=self.hidden_states,
                conditional_dict=self.conditional_dict,
                timestep=self.timestep.unsqueeze(1).expand(-1, self.hidden_states.shape[1]),
                kv_cache=self.kv_cache1,
                crossattn_cache=self.crossattn_cache,
                current_start=self.kv_cache_starts,
                current_end=self.kv_cache_ends,
                block_mode=block_mode,
                block_num=block_num,
                patched_x_shape=patched_x_shape,
                block_x=self.block_x,
            )

        return denoised_pred, patched_x_shape
