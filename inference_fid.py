import os
import torch
from torchvision.utils import save_image
from tqdm import tqdm
from types import MethodType
# from peft import PeftModel
# from pipeline_sit import SiTPipeline
# from SiT.get_sampler import get_sampler
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from models import SiT_models
from download import find_model
from transport import create_transport, Sampler
from diffusers.models import AutoencoderKL
from train_utils import parse_transport_args
import wandb_utils

from diffusers import AutoencoderKL

# ------------ 설정 ------------ #
torch.manual_seed(0)
torch.set_grad_enabled(False)

device = 'cuda:7'
output_root = 'results_fid'
os.makedirs(output_root, exist_ok=True)

num_classes = 100
images_per_class = 500
batch_size = 1  # 고정
# sample_fn = get_sampler('ODE_EULER', 20)
transport = create_transport(
    'Linear',
    'velocity',
    'velocity',
    None,
    None
)  # default: velocity; 
transport_sampler = Sampler(transport)
sample_fn = transport_sampler.sample_ode(num_steps=20)

# ------------ 모델 준비 ------------ #
# pipe = SiTPipeline.from_pretrained('sit_pipeline', torch_dtype=torch.float32).to(device)
# model_orig = pipe.transformer
# vae = pipe.vae
# lora_path = '/ceph/jiwon/nips_sit/results/epoch_7/lora_weights'
# model = PeftModel.from_pretrained(model_orig, lora_path)
latent_size = 256 // 8
model = SiT_models['SiT-XL/2'](
    input_size=latent_size,
    num_classes=1000,
    learn_guidance_embedding=True,
).to(device)
ckpt_path = '/media/dataset2/jiwon/sit_train_cfg_distill/results/000-SiT-XL-2-Linear-velocity-velocity/SiT-XL-2-CFG-Distill-30000.pt'
model.eval()
model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-mse").to(device)

# def forward_with_cfg(self, x, t, y, cfg_scale):
#     half = x[: len(x) // 2]
#     combined = torch.cat([half, half], dim=0)
#     model_out = self.forward(combined, t, y)
#     eps, rest = model_out[:, :3], model_out[:, 3:]
#     cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
#     half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
#     eps = torch.cat([half_eps, half_eps], dim=0)
#     return torch.cat([eps, rest], dim=1)

# model.forward_with_cfg = MethodType(forward_with_cfg, model)

def refine_latents(model, latents, class_labels):
    initial_timestep = torch.Tensor([0.0] * latents.shape[0]).to(latents.device)
    epsilon = model(latents, initial_timestep, class_labels)
    return latents + epsilon

# ------------ 저장용 함수 ------------ #
def save_single_image(tensor, save_dir, class_label, idx):
    tensor = (tensor.clamp(-1, 1) + 1.0) / 2.0
    path = os.path.join(save_dir, f"img_{class_label:03}_{idx:03}.png")
    save_image(tensor, path)

# ------------ 저장 폴더 생성 ------------ #
for mode in ['original', 'distill4.0', 'cfg4.0']:
    os.makedirs(os.path.join(output_root, mode), exist_ok=True)

# ------------ 인퍼런스 루프 ------------ #
SEED_OFFSET = 10000  # 안전한 범위 확보

for class_label in tqdm(range(num_classes), desc="Classes", leave=False):
    for img_idx in tqdm(range(images_per_class), desc=f"Class {class_label}", leave=False):
        seed = SEED_OFFSET + class_label * images_per_class + img_idx  # 안전한 고유 시드
        torch.manual_seed(seed)

        labels = torch.tensor([class_label]).to(device)
        latents = torch.randn(batch_size, 4, 32, 32).to(device)  # 동일 latents로 3개 방식 공유
        
        # Original
        model_kwargs = dict(y=labels, g=torch.Tensor([0.0]).to(device))
        origin = sample_fn(latents, model.forward, **model_kwargs)[-1]
        decoded_origin = vae.decode(origin / 0.18215).sample[0]

        # Distill (cfg_scale=4.0)
        model_kwargs = dict(y=labels, g=torch.Tensor([4.0]).to(device))
        origin = sample_fn(latents, model.forward, **model_kwargs)[-1]
        decoded_distill = vae.decode(origin / 0.18215).sample[0]

        # CFG
        y_null = torch.tensor([1000], device=device)
        combined_latents = torch.cat([latents, latents], dim=0)
        combined_y = torch.cat([labels, y_null], dim=0)
        model_kwargs = dict(y=combined_y, cfg_scale=4.0)
        origin_cfg = sample_fn(combined_latents, model.forward_with_cfg, **model_kwargs)[-1]
        decoded_cfg = vae.decode(origin_cfg / 0.18215).sample[0]

        save_single_image(decoded_origin, os.path.join(output_root, 'original'), class_label, img_idx)
        save_single_image(decoded_distill, os.path.join(output_root, 'distill4.0'), class_label, img_idx)
        save_single_image(decoded_cfg, os.path.join(output_root, 'cfg4.0'), class_label, img_idx)

print("✅ 배치 1 기반 인퍼런스 및 저장 완료")