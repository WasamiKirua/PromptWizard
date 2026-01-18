from __future__ import annotations

from typing import Any, Dict

GEMINI_MODEL = "gemini-2.5-flash"
OPENAI_MODEL = "gpt-4o-mini"
GROK_MODEL = "grok-2-vision-latest"

PROVIDER_LABELS = {
    "gemini": "Gemini",
    "openai": "OpenAI",
    "grok": "Grok",
}

MODEL_DATA: Dict[str, Any] = {
    "model_families": [
        {
            "id": "sd15",
            "label": "Stable Diffusion 1.5 Family",
            "architecture": "stable-diffusion-v1",
            "type": "image",
            "default_resolution": 512,
            "loader_node": "CheckpointLoaderSimple",
            "notes": "All SD 1.x finetunes (Anything, DreamShaper, etc.) are compatible.",
            "checkpoints": [
                {
                    "id": "sd15_base",
                    "label": "Stable Diffusion 1.5",
                    "path": "models/checkpoints/sd15.safetensors",
                },
                {
                    "id": "sd15_inpaint",
                    "label": "SD 1.5 Inpainting",
                    "path": "models/checkpoints/sd15_inpainting.safetensors",
                },
            ],
        },
        {
            "id": "sd21",
            "label": "Stable Diffusion 2.x Family",
            "architecture": "stable-diffusion-v2",
            "type": "image",
            "default_resolution": 768,
            "loader_node": "CheckpointLoaderSimple",
            "notes": "Includes 2.0 / 2.1 512px and 768px variants.",
            "checkpoints": [
                {
                    "id": "sd21_base_512",
                    "label": "Stable Diffusion 2.1 (512)",
                    "path": "models/checkpoints/sd21_512.safetensors",
                },
                {
                    "id": "sd21_base_768",
                    "label": "Stable Diffusion 2.1 (768)",
                    "path": "models/checkpoints/sd21_768.safetensors",
                },
            ],
        },
        {
            "id": "sdxl",
            "label": "SDXL 1.0 Family",
            "architecture": "sdxl-1.0",
            "type": "image",
            "default_resolution": 1024,
            "loader_node": "SDXLCheckpointLoader",
            "notes": "SDXL base + refiner + SDXL finetunes.",
            "checkpoints": [
                {
                    "id": "sdxl_base",
                    "label": "SDXL 1.0 Base",
                    "path": "models/checkpoints/sdxl_base.safetensors",
                },
                {
                    "id": "sdxl_refiner",
                    "label": "SDXL 1.0 Refiner",
                    "path": "models/checkpoints/sdxl_refiner.safetensors",
                },
                {
                    "id": "sdxl_juggernaut",
                    "label": "Juggernaut XL (example finetune)",
                    "path": "models/checkpoints/juggernaut_xl.safetensors",
                },
            ],
        },
        {
            "id": "sd3",
            "label": "Stable Diffusion 3 / 3.5",
            "architecture": "stable-diffusion-3",
            "type": "image",
            "default_resolution": 1024,
            "loader_node": "SD3CheckpointLoader",
            "notes": "Requires SD3-specific loader/workflow, separate text encoders.",
            "checkpoints": [
                {
                    "id": "sd3_medium",
                    "label": "Stable Diffusion 3 Medium",
                    "path": "models/checkpoints/sd3_medium.safetensors",
                },
                {
                    "id": "sd3_large",
                    "label": "Stable Diffusion 3 Large",
                    "path": "models/checkpoints/sd3_large.safetensors",
                },
                {
                    "id": "sd35_large",
                    "label": "Stable Diffusion 3.5 Large",
                    "path": "models/checkpoints/sd35_large.safetensors",
                },
                {
                    "id": "sd35_turbo",
                    "label": "Stable Diffusion 3.5 Turbo",
                    "path": "models/checkpoints/sd35_turbo.safetensors",
                },
            ],
        },
        {
            "id": "flux1",
            "label": "FLUX 1.x Family",
            "architecture": "flux-1",
            "type": "image",
            "default_resolution": 1024,
            "loader_node": "FLUXLoader",
            "notes": "Needs FLUX-specific custom nodes and workflows.",
            "checkpoints": [
                {
                    "id": "flux1_dev",
                    "label": "FLUX.1 dev",
                    "path": "models/checkpoints/flux1_dev.safetensors",
                },
                {
                    "id": "flux1_schnell",
                    "label": "FLUX.1 schnell",
                    "path": "models/checkpoints/flux1_schnell.safetensors",
                },
            ],
        },
        {
            "id": "flux2",
            "label": "FLUX.2 Family",
            "architecture": "flux-2",
            "type": "image",
            "default_resolution": 1024,
            "loader_node": "FLUX2Loader",
            "notes": "FLUX.2 dev/flex/pro variants via updated FLUX nodes.",
            "checkpoints": [
                {
                    "id": "flux2_dev",
                    "label": "FLUX.2 dev",
                    "path": "models/checkpoints/flux2_dev.safetensors",
                },
                {
                    "id": "flux2_flex",
                    "label": "FLUX.2 flex",
                    "path": "models/checkpoints/flux2_flex.safetensors",
                },
                {
                    "id": "flux2_pro",
                    "label": "FLUX.2 pro",
                    "path": "models/checkpoints/flux2_pro.safetensors",
                },
            ],
        },
        {
            "id": "z_image",
            "label": "Z-Image Turbo (AuraFlow)",
            "architecture": "auraflow",
            "type": "image",
            "default_resolution": 1024,
            "loader_node": "UNETLoader (AuraFlow)",
            "notes": "Uses Qwen 3.4B LLM as text encoder. High adherence to natural language.",
            "checkpoints": [
                {
                    "id": "z_image_turbo",
                    "label": "Z-Image Turbo BF16",
                    "path": "z_image_turbo_bf16.safetensors",
                }
            ],
        },
        {
            "id": "wan22",
            "label": "WAN 2.2 Family",
            "architecture": "wan-2.2",
            "type": "image+video",
            "default_resolution": 720,
            "loader_node": "WAN22Loader",
            "notes": "Requires WAN 2.2 custom nodes/workflows; supports T2I and T2V.",
            "checkpoints": [
                {
                    "id": "wan22_5b_ti2v",
                    "label": "Wan 2.2 5B (T2I/T2V)",
                    "path": "models/checkpoints/wan2.2_ti2v_5b_fp16.safetensors",
                },
                {
                    "id": "wan22_14b_ti2v",
                    "label": "Wan 2.2 14B (T2I/T2V)",
                    "path": "models/checkpoints/wan2.2_ti2v_14b_fp16.safetensors",
                },
            ],
        },
        {
            "id": "svd",
            "label": "Stable Video Diffusion",
            "architecture": "stable-video-diffusion",
            "type": "video",
            "default_resolution": 576,
            "loader_node": "VideoDiffusionLoader",
            "notes": "Includes SVD and similar open video models via video nodes.",
            "checkpoints": [
                {
                    "id": "svd_base",
                    "label": "Stable Video Diffusion Base",
                    "path": "models/checkpoints/svd_base.safetensors",
                },
                {
                    "id": "svd_xt",
                    "label": "Stable Video Diffusion XT",
                    "path": "models/checkpoints/svd_xt.safetensors",
                },
            ],
        },
        {
            "id": "cascade",
            "label": "Stable Cascade / Other",
            "architecture": "stable-cascade",
            "type": "image",
            "default_resolution": 1024,
            "loader_node": "CascadeCheckpointLoader",
            "notes": "Includes Stable Cascade, Kandinsky, Playground v2, PixArt, etc.",
            "checkpoints": [
                {
                    "id": "stable_cascade",
                    "label": "Stable Cascade",
                    "path": "models/checkpoints/stable_cascade.safetensors",
                },
                {
                    "id": "kandinsky_2",
                    "label": "Kandinsky 2.x",
                    "path": "models/checkpoints/kandinsky2.safetensors",
                },
                {
                    "id": "playground_v2",
                    "label": "Playground v2",
                    "path": "models/checkpoints/playground_v2.safetensors",
                },
                {
                    "id": "pixart_sigma",
                    "label": "PixArt-Σ (Sigma)",
                    "path": "models/checkpoints/pixart_sigma.safetensors",
                },
            ],
        },
    ],
    "auxiliary_models": {
        "vaes": [
            {
                "id": "vae_sd15",
                "label": "VAE for SD 1.5",
                "path": "models/vae/vae_sd15.safetensors",
            },
            {
                "id": "vae_sdxl",
                "label": "VAE for SDXL",
                "path": "models/vae/vae_sdxl.safetensors",
            },
            {
                "id": "vae_wan22",
                "label": "VAE for Wan 2.2",
                "path": "models/vae/vae_wan22.safetensors",
            },
        ],
        "upscalers": [
            {
                "id": "4x_ultrasharp",
                "label": "4x-UltraSharp",
                "path": "models/upscale/4x_ultrasharp.safetensors",
            },
            {
                "id": "swinir_4x",
                "label": "SwinIR 4x",
                "path": "models/upscale/swinir_4x.safetensors",
            },
        ],
        "face_fixers": [
            {
                "id": "gfpgan",
                "label": "GFPGAN",
                "path": "models/facefix/gfpgan.pth",
            },
            {
                "id": "codeformer",
                "label": "CodeFormer",
                "path": "models/facefix/codeformer.pth",
            },
        ],
        "control_models": [
            {
                "id": "controlnet_canny",
                "label": "ControlNet Canny",
                "path": "models/controlnet/controlnet_canny.safetensors",
            },
            {
                "id": "controlnet_depth",
                "label": "ControlNet Depth",
                "path": "models/controlnet/controlnet_depth.safetensors",
            },
            {
                "id": "ip_adapter_face",
                "label": "IP-Adapter Face",
                "path": "models/ipadapter/ipadapter_face.safetensors",
            },
        ],
    },
}

FOCUS_ASPECTS_LIST = [
    "Subject Identity",
    "Outfit & Fashion",
    "Pose & Angle",
    "Art Style & Aesthetic",
    "Lighting & Atmosphere",
    "Background & Setting",
    "Motion & Camera Movement",
]
