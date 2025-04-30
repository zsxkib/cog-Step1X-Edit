# predict.py  ──  Cog interface for Step1X-Edit
# ---------------------------------------------------------------------
# Replicates the exact behaviour of app.py (28 steps, CFG 6.0, etc.)
# without any Gradio / Spaces code.  Copy-and-paste ready.

import os, time, random, math, itertools
from pathlib import Path as LocalPath
from typing import Optional, List

import numpy as np
import torch
from cog import BasePredictor, Input, Path                     # Cog helpers
from einops import rearrange, repeat
from huggingface_hub import snapshot_download
from PIL import Image
from safetensors.torch import load_file
from torchvision.transforms import functional as F
from tqdm import tqdm                                         # progress bars

# ────────── CONSTANTS ────────────────────────────────────────────────
MODEL_CACHE       = "model_cache"
MODEL_REPO        = "stepfun-ai/Step1X-Edit"                  # DiT + VAE
QWEN_MODEL_PATH   = "Qwen/Qwen2.5-VL-7B-Instruct"             # Qwen-VL encoder
CUDA_DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(MODEL_CACHE, exist_ok=True)

# ────────── IMPORTS FROM ORIGINAL CODEBASE ───────────────────────────
from modules.autoencoder   import AutoEncoder
from modules.conditioner   import Qwen25VL_7b_Embedder as Qwen2VLEmbedder
from modules.model_edit    import Step1XParams, Step1XEdit
import sampling                                                # provided with repo

# ─────────────────────────────────────────────────────────────────────
def load_state_dict(model, ckpt_path: str, device="cuda", strict=False, assign=True):
    """Load .pt / .safetensors checkpoint into model and return it."""
    if ckpt_path.endswith(".safetensors"):
        state_dict = load_file(ckpt_path, device)
    else:
        state_dict = torch.load(ckpt_path, map_location="cpu")

    model.load_state_dict(state_dict, strict=strict, assign=assign)
    return model


def build_models(dit_path: str, ae_path: str, qwen_path: str,
                 device: str, max_len: int = 640, dtype=torch.bfloat16):
    """Instantiate VAE, DiT, Qwen2-VL encoder with pretrained weights."""
    qwen_enc = Qwen2VLEmbedder(qwen_path, device=device,
                               max_length=max_len, dtype=dtype)

    # Build empty VAE + DiT on meta device then load checkpoints
    with torch.device("meta"):
        ae = AutoEncoder(
            resolution=256, in_channels=3, out_ch=3, ch=128,
            ch_mult=[1, 2, 4, 4], num_res_blocks=2, z_channels=16,
            scale_factor=0.3611, shift_factor=0.1159,
        )
        step1x_cfg = Step1XParams(
            in_channels=64, out_channels=64, vec_in_dim=768,
            context_in_dim=4096, hidden_size=3072, mlp_ratio=4.0,
            num_heads=24, depth=19, depth_single_blocks=38,
            axes_dim=[16, 56, 56], theta=10_000, qkv_bias=True,
        )
        dit = Step1XEdit(step1x_cfg)

    ae  = load_state_dict(ae,  ae_path,  device)
    dit = load_state_dict(dit, dit_path, device)

    ae  = ae.to(device=device, dtype=torch.float32)
    dit = dit.to(device=device, dtype=dtype)
    return ae, dit, qwen_enc


# ──────────  ImageGenerator  (from app.py, unmodified)  ──────────────
class ImageGenerator:
    """End-to-end wrapper that mirrors the one in app.py."""

    def __init__(self, dit_path: str, ae_path: str, qwen_path: str,
                 device="cuda", max_length=640, dtype=torch.bfloat16):
        self.device = torch.device(device)
        self.ae, self.dit, self.llm_encoder = build_models(
            dit_path, ae_path, qwen_path, device, max_length, dtype
        )

    # Convenience for Predictor.setup()
    def to_cuda(self):
        self.ae.to("cuda", dtype=torch.float32)
        self.dit.to("cuda", dtype=torch.bfloat16)
        self.llm_encoder.to("cuda", dtype=torch.bfloat16)

    # ---------------- helpers ported 1-to-1 from app.py --------------
    def prepare(self, prompt, img, ref_image, ref_image_raw):
        bs, _, h, w       = img.shape
        _, _, ref_h, ref_w = ref_image.shape
        assert (h, w) == (ref_h, ref_w)

        # broadcast prompt list
        if bs == 1 and not isinstance(prompt, str):
            bs = len(prompt)
        elif bs >= 1 and isinstance(prompt, str):
            prompt = [prompt] * bs

        img      = rearrange(img,      "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
        ref_img  = rearrange(ref_image,"b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
        if img.shape[0] == 1 and bs > 1:          # replicate for CFG pairs
            img      = repeat(img,     "1 ... -> bs ...", bs=bs)
            ref_img  = repeat(ref_img, "1 ... -> bs ...", bs=bs)

        img_ids = torch.zeros(h//2, w//2, 3)
        img_ids[...,1] += torch.arange(h//2)[:,None]
        img_ids[...,2] += torch.arange(w//2)[None,:]
        img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)

        ref_ids = torch.zeros(h//2, w//2, 3)
        ref_ids[...,1] += torch.arange(h//2)[:,None]
        ref_ids[...,2] += torch.arange(w//2)[None,:]
        ref_ids = repeat(ref_ids, "h w c -> b (h w) c", b=bs)

        # LLM encoding
        if isinstance(prompt, str):
            prompt = [prompt]
        txt, mask = self.llm_encoder(prompt, ref_image_raw)
        txt_ids   = torch.zeros(bs, txt.shape[1], 3)

        img     = torch.cat([img,     ref_img.to(img)],     dim=-2)
        img_ids = torch.cat([img_ids, ref_ids],             dim=-2)

        return dict(img=img, mask=mask, img_ids=img_ids.to(img),
                    llm_embedding=txt.to(img), txt_ids=txt_ids.to(img))

    @staticmethod
    def process_diff_norm(diff_norm, k=0.4):
        powd = torch.pow(diff_norm, k)
        return torch.where(diff_norm > 1.0, powd,
               torch.where(diff_norm < 1.0, torch.ones_like(diff_norm), diff_norm))

    def denoise(self, *, img, img_ids, llm_embedding, txt_ids,
                timesteps: List[float], cfg_guidance=6.0,
                mask=None, show_progress=False, timesteps_truncate=1.0):
        iterator = tqdm(itertools.pairwise(timesteps), desc="denoising…") if show_progress \
                   else itertools.pairwise(timesteps)
        for t_curr, t_prev in iterator:
            if img.shape[0] == 1 and cfg_guidance != -1:
                img = torch.cat([img, img], dim=0)

            t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
            txt, vec = self.dit.connector(llm_embedding, t_vec, mask)
            pred = self.dit(img=img, img_ids=img_ids, txt=txt,
                            txt_ids=txt_ids, y=vec, timesteps=t_vec)

            if cfg_guidance != -1:
                cond, uncond = pred[:pred.shape[0]//2], pred[pred.shape[0]//2:]
                if t_curr > timesteps_truncate:
                    diff      = cond - uncond
                    diff_norm = torch.norm(diff, dim=2, keepdim=True)
                    pred      = uncond + cfg_guidance * diff / self.process_diff_norm(diff_norm)
                else:
                    pred      = uncond + cfg_guidance * (cond - uncond)

            tem     = img[:img.shape[0]//2] + (t_prev - t_curr) * pred
            half    = img.shape[1] // 2
            img     = torch.cat([tem[:, :half], img[:img.shape[0]//2, half:]], dim=1)

        return img[:, : img.shape[1] // 2]

    @staticmethod
    def unpack(x, h, w):
        return rearrange(x, "b (h w) (c ph pw) -> b c (h ph) (w pw)",
                         h=math.ceil(h/16), w=math.ceil(w/16), ph=2, pw=2)

    # --------------------- image I/O helpers -------------------------
    @staticmethod
    def load_image(img):
        if isinstance(img, np.ndarray):
            t = torch.from_numpy(img).permute(2,0,1).float()/255.; return t.unsqueeze(0)
        if isinstance(img, Image.Image):
            return F.to_tensor(img.convert("RGB")).unsqueeze(0)
        if isinstance(img, torch.Tensor):
            return img
        if isinstance(img, str):
            return F.to_tensor(Image.open(img).convert("RGB")).unsqueeze(0)
        raise ValueError(f"Unsupported image type: {type(img)}")

    @staticmethod
    def output_process_image(img_pil, size):
        return img_pil.resize(size)

    @staticmethod
    def input_process_image(img: Image.Image, img_size=512):
        w,h = img.size ; r = w/h
        if w>h:
            w_new = math.ceil(math.sqrt(img_size*img_size*r))
            h_new = math.ceil(w_new/r)
        else:
            h_new = math.ceil(math.sqrt(img_size*img_size/r))
            w_new = math.ceil(h_new*r)
        h_new = math.ceil(h_new) // 16 * 16
        w_new = math.ceil(w_new) // 16 * 16
        return img.resize((w_new,h_new)), img.size

    # -------------------------- main API -----------------------------
    @torch.inference_mode()
    def generate_image(self, *, prompt: str, negative_prompt: str,
                       ref_images: Image.Image, num_steps: int, cfg_guidance: float,
                       seed: int, size_level: int,
                       num_samples: int = 1, init_image=None,
                       image2image_strength: float = 0.0,
                       show_progress: bool = False):

        assert num_samples == 1, "num_samples > 1 not supported."
        ref_raw, original_size = self.input_process_image(ref_images, size_level)
        width, height = ref_raw.width, ref_raw.height

        ref_raw_tensor = self.load_image(ref_raw).to(self.device)
        ref_latent = self.ae.encode(ref_raw_tensor*2 - 1)

        if seed < 0:
            seed = torch.seed()
        
        # Create generator on the correct device
        g = torch.Generator(device=self.device).manual_seed(seed)

        # optional init_image
        if init_image is not None:
            init_tensor = self.load_image(init_image).to(self.device)
            init_tensor = torch.nn.functional.interpolate(init_tensor, (height,width))
            init_latent = self.ae.encode(init_tensor*2 - 1)
        else:
            init_latent = None

        x = torch.randn(num_samples,16,height//8,width//8,
                        dtype=torch.bfloat16, generator=g, device=self.device)

        timesteps = sampling.get_schedule(num_steps, x.shape[-1]*x.shape[-2]//4, shift=True)
        if init_latent is not None:
            t_idx = int((1-image2image_strength)*num_steps)
            t0    = timesteps[t_idx]
            timesteps = timesteps[t_idx:]
            x = t0*x + (1.0-t0)*init_latent.to(x)

        x           = torch.cat([x,x], dim=0)
        ref_latent  = torch.cat([ref_latent, ref_latent], dim=0)
        ref_raw_rep = torch.cat([ref_raw_tensor, ref_raw_tensor], dim=0)
        inputs      = self.prepare([prompt, negative_prompt], x,
                                   ref_image=ref_latent,
                                   ref_image_raw=ref_raw_rep)

        x = self.denoise(**inputs, cfg_guidance=cfg_guidance,
                         timesteps=timesteps, show_progress=show_progress)

        x = self.unpack(x.float(), height, width)
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
            x = self.ae.decode(x).clamp(-1,1).mul_(0.5).add_(0.5)

        return [self.output_process_image(F.to_pil_image(img), original_size)
                for img in x.float()]


# ────────── Cog Predictor ────────────────────────────────────────────
class Predictor(BasePredictor):
    """Re-implements the Gradio demo controls as Cog inputs."""

    def setup(self):
        st = time.time()

        # download checkpoints once
        model_dir = snapshot_download(
            repo_id=MODEL_REPO,
            local_dir=os.path.join(MODEL_CACHE, MODEL_REPO.split("/")[-1]),
            local_dir_use_symlinks=False,
        )
        dit_ckpt = os.path.join(model_dir, "step1x-edit-i1258.safetensors")
        vae_ckpt = os.path.join(model_dir, "vae.safetensors")

        # instantiate pipeline
        self.pipe = ImageGenerator(
            dit_path=dit_ckpt,
            ae_path=vae_ckpt,
            qwen_path=QWEN_MODEL_PATH,
            device=CUDA_DEVICE,
            dtype=torch.bfloat16,
        )
        if CUDA_DEVICE == "cuda":
            self.pipe.to_cuda()

        print(f"Predictor ready in {time.time()-st:.1f}s")

    # ----------------------------- run --------------------------------
    def predict(
        self,
        image: Path = Input(description="Input image"),
        prompt: str  = Input(description="编辑指令 prompt", default="Remove the person from the image."),
        size_level: int = Input(description="Internal resolution",
                                default=512, choices=[512,768,1024]),
        random_seed: int = Input(description="Random seed (-1=random)", default=-1),
    ) -> Path:

        seed = random.randint(0, 2**32-1) if random_seed == -1 else random_seed
        print("Using seed:", seed)

        img_pil = Image.open(str(image)).convert("RGB")

        result_pil = self.pipe.generate_image(
            prompt=prompt, negative_prompt="",
            ref_images=img_pil, num_steps=28, cfg_guidance=6.0,
            seed=seed, size_level=size_level, show_progress=True
        )[0]

        out_path = "/tmp/step1x_edit_output.png"
        result_pil.save(out_path)
        print("Saved to", out_path)
        return Path(out_path)