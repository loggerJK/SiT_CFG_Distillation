import os
import torch
import torch.distributed as dist
from torchvision.utils import save_image
from tqdm import tqdm
from types import MethodType

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from models import SiT_models
from transport import create_transport, Sampler
from diffusers.models import AutoencoderKL

# ------------ DDP 설정 ------------ #
dist.init_process_group("nccl")
rank = dist.get_rank()
world_size = dist.get_world_size()
device = f'cuda:{rank}'
torch.cuda.set_device(device)
torch.manual_seed(0)
torch.set_grad_enabled(False)

# ------------ 기본 설정 ------------ #
output_root = 'results_fid'
if rank == 0:
    os.makedirs(output_root, exist_ok=True)

num_classes = 100
images_per_class = 500
batch_size = 1  # 배치 크기 1로 고정
transport = create_transport(
    'Linear',
    'velocity',
    'velocity',
    None,
    None
)
transport_sampler = Sampler(transport)
sample_fn = transport_sampler.sample_ode(num_steps=20)

# ------------ 모델 준비 ------------ #
latent_size = 256 // 8
model = SiT_models['SiT-XL/2'](
    input_size=latent_size,
    num_classes=1000,
    learn_guidance_embedding=True,
).to(device)
ckpt_path = '/media/dataset2/jiwon/sit_train_cfg_distill/results/SiT_embed_CFG1.09.0_bf16_lr1e-4/epoch0.pt'
model.eval()
model.load_state_dict(torch.load(ckpt_path, map_location='cpu',weights_only=False)['model'])
vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-mse").to(device)

# def forward_with_cfg(self, x, t, y, cfg_scale):
#     half = x[: len(x) // 2]
#     combined = torch.cat([half, half], dim=0)
#     model_out = self.forward(combined, t, y)
#     eps, rest = model_out[:, :4], model_out[:, 4:] 
#     cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
#     half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
#     eps = torch.cat([half_eps, half_eps], dim=0)
#     if rest.shape[1] == 0:
#         return eps
#     else:
#         return torch.cat([eps, rest], dim=1)

# model.forward_with_cfg = MethodType(forward_with_cfg, model)

# ------------ 저장용 함수 ------------ #
def save_single_image(tensor, save_dir, class_label, idx):
    tensor = (tensor.clamp(-1, 1) + 1.0) / 2.0
    path = os.path.join(save_dir, f"img_{class_label:03}_{idx:03}.png")
    save_image(tensor, path)

# ------------ 저장 폴더 생성 ------------ #
if rank == 0:
    for mode in ['TEST_distill_embed']:
        os.makedirs(os.path.join(output_root, mode), exist_ok=True)

# ------------ 인퍼런스 루프 ------------ #
SEED_OFFSET = 10000

total_images = num_classes * images_per_class
image_indices = list(range(total_images))
image_indices_this_rank = image_indices[rank::world_size]

for global_idx in tqdm(image_indices_this_rank, desc=f"Generating on rank {rank}"):
    class_label = global_idx // images_per_class
    img_idx = global_idx % images_per_class

    path_original = os.path.join(output_root, 'TEST_distill_embed', f"img_{class_label:03}_{img_idx:03}.png")

    if os.path.exists(path_original):
        continue
    
    seed = SEED_OFFSET + global_idx
    torch.manual_seed(seed)

    labels = torch.tensor([class_label]).to(device)
    latents = torch.randn(batch_size, 4, 32, 32).to(device)
    
    # Original
    model_kwargs = dict(y=labels, g=torch.Tensor([4.0]).to(device))
    origin = sample_fn(latents, model.forward, **model_kwargs)[-1]
    decoded_origin = vae.decode(origin / 0.18215).sample[0]


    save_single_image(decoded_origin, os.path.join(output_root, 'TEST_distill_embed'), class_label, img_idx)

if rank == 0:
    print("✅ 멀티 GPU 기반 인퍼런스 및 저장 완료")

dist.destroy_process_group()
