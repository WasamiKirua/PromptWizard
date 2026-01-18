from __future__ import annotations

import base64
import json
import os
from typing import Any, Dict, List, Optional

import httpx
from fastapi import UploadFile
from google import genai
from google.genai import types
from openai import OpenAI

from config import GEMINI_MODEL, GROK_MODEL, MODEL_DATA, OPENAI_MODEL, PROVIDER_LABELS


def resolve_api_key(provider: str, provided_key: Optional[str], env_path: str) -> Optional[str]:
    if provided_key:
        return provided_key

    env: Dict[str, str] = {}
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as handle:
            for line in handle.read().splitlines():
                stripped = line.strip()
                if not stripped or stripped.startswith("#") or "=" not in stripped:
                    continue
                key, value = stripped.split("=", 1)
                env[key.strip()] = value.strip()

    env.update({k: v for k, v in os.environ.items()})

    if provider == "gemini":
        return env.get("GEMINI_API_KEY") or env.get("API_KEY")
    if provider == "openai":
        return env.get("OPENAI_API_KEY")
    if provider == "grok":
        return env.get("GROK_API_KEY") or env.get("XAI_API_KEY")
    return None


def ensure_negative_prompt(family_id: str, provided: Optional[str]) -> str:
    if family_id == "z_image":
        return provided or "blurry ugly bad"
    return provided or ""


def build_prompt_text(
    config: Dict[str, Any],
    family: Dict[str, Any],
    checkpoint: Dict[str, Any],
) -> str:
    prompt_guidance = ""
    family_id = family["id"]

    if family_id in {"flux1", "flux2"}:
        prompt_guidance = (
            "Output a rich, descriptive natural language paragraph. Focus on textures, "
            "lighting, and physical details. Do NOT use comma-separated tags unless "
            "specifying a specific trigger word. Describe the scene as if writing a story. "
            "Flux handles complex instruction well."
        )
    elif family_id == "sdxl":
        prompt_guidance = (
            "Output a hybrid format: start with a strong subject description in natural language, "
            "followed by comma-separated quality tags (e.g., 'masterpiece, best quality, 8k, "
            "ultra-detailed'). Include specific camera settings if photorealistic."
        )
    elif family_id == "sd15":
        prompt_guidance = (
            "Heavily reliant on tags. Use comma-separated keywords. Focus on 'best quality', "
            "'masterpiece', and specific art station tags. Keep sentences short or broken into tokens."
        )
    elif family_id == "sd3":
        prompt_guidance = (
            "Use natural language with high attention to detail. SD3 adheres strictly to the prompt, "
            "so include every visible element, color, and relationship between objects. It understands "
            "spatial relationships well."
        )
    elif family_id == "z_image":
        prompt_guidance = (
            "This model (AuraFlow architecture) uses the Qwen 3.4B LLM as a text encoder. Write a "
            "long, flowing, multi-sentence caption (120-180 words) that reads like vivid alt-text: "
            "cover subject identity, wardrobe, props, background, lighting, composition, camera/lens "
            "info, and microtextures. Avoid tag soup and keep it natural. CRITICAL: The Negative Prompt "
            "must be exactly 'blurry ugly bad'."
        )
    elif family_id in {"wan22", "svd"}:
        prompt_guidance = (
            "This is for a VIDEO generation model. Describe the MOTION, camera movement (pan, zoom, "
            "tilt), and duration flow. Start with the scene description, then describe how the subject "
            "moves or how the camera moves. E.g., 'The girl smiles as the camera slowly zooms in'."
        )
    elif family_id in {"cascade", "sd21"}:
        prompt_guidance = "Standard Stable Diffusion prompting. Use a mix of description and quality tags."
    else:
        prompt_guidance = "Generate a high-quality, detailed prompt suitable for diffusion models."

    aux_notes: List[str] = []
    auxiliary = config.get("auxiliary", {})
    if auxiliary.get("controlModel"):
        control = next(
            (c for c in MODEL_DATA["auxiliary_models"]["control_models"] if c["id"] == auxiliary["controlModel"]),
            None,
        )
        aux_notes.append(
            f"User is using ControlNet ({control['label'] if control else 'control model'}). Ensure the prompt "
            "describes pose/structure clearly so it aligns with the control signal."
        )
    if auxiliary.get("upscaler"):
        aux_notes.append(
            "User is using an Upscaler. Emphasize high-frequency details (texture, fabric threads, pores) "
            "to justify the resolution."
        )
    if auxiliary.get("faceFixer"):
        aux_notes.append("User is using a face-fixing stage. Keep facial identity consistent and clearly described.")

    focus_aspects = config.get("focusAspects", [])
    focus_instruction = (
        f"Pay special attention to these aspects: {', '.join(focus_aspects)}. Ensure these are described in high detail."
        if focus_aspects
        else "Capture the overall vibe and subject of the images."
    )

    negative_prompt_guidance = (
        'Negative Prompt MUST be exactly "blurry ugly bad".'
        if family_id == "z_image"
        else "Include a concise Negative Prompt that filters common artifacts (blur, distortion, duplicate limbs, bad anatomy)."
    )

    aux_instruction_block = " ".join(aux_notes) if aux_notes else "No auxiliary models selected."

    return (
        "You are an expert AI Prompt Engineer.\n"
        "Analyze the provided reference images and generate a SINGLE, perfect prompt that would recreate a similar output "
        f"using '{family['label']}' (Checkpoint: {checkpoint['label']}).\n\n"
        f"Target Model Architecture: {family['architecture']}\n"
        f"Model Type: {family['type']}\n"
        f"Focus Areas: {focus_instruction}\n"
        f"User Context/Notes: {config.get('additionalContext') or 'None provided'}\n"
        f"{aux_instruction_block}\n\n"
        "Guidance:\n"
        f"- {prompt_guidance}\n"
        "- If multiple images are provided, synthesize their common elements into one cohesive character or style definition.\n"
        f"- {negative_prompt_guidance}\n\n"
        "Return only valid JSON:\n"
        "{\n"
        "  \"prompt\": \"THE GENERATED PROMPT STRING\",\n"
        "  \"negativePrompt\": \"THE NEGATIVE PROMPT STRING (OR EMPTY)\"\n"
        "}"
    )


def get_family(family_id: str) -> Optional[Dict[str, Any]]:
    return next((f for f in MODEL_DATA["model_families"] if f["id"] == family_id), None)


def get_checkpoint(family: Dict[str, Any], checkpoint_id: str) -> Optional[Dict[str, Any]]:
    return next((c for c in family.get("checkpoints", []) if c["id"] == checkpoint_id), None)


def _extract_output_text(response: Any) -> str:
    output_text = getattr(response, "output_text", None)
    if output_text:
        return str(output_text).strip()

    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            if getattr(content, "type", "") in {"output_text", "text"}:
                text = getattr(content, "text", "")
                if text:
                    return str(text).strip()
    return ""


def call_gemini(
    images: List[UploadFile],
    prompt_text: str,
    temperature: float,
    api_key: str,
    family_id: str,
) -> Dict[str, str]:
    parts: List[types.Part] = []
    for img in images:
        raw = img.file.read()
        parts.append(
            types.Part.from_bytes(
                data=raw,
                mime_type=img.content_type or "image/png",
            )
        )

    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[*parts, prompt_text],
            config=types.GenerateContentConfig(
                temperature=temperature,
                response_mime_type="application/json",
            ),
        )
    except Exception as exc:
        raise RuntimeError(f"Gemini API error: {exc}") from exc

    text = getattr(response, "text", None)
    if not text:
        raise RuntimeError("Gemini response was empty.")

    parsed = json.loads(text)
    return {
        "prompt": parsed.get("prompt") or "Failed to generate prompt.",
        "negativePrompt": ensure_negative_prompt(family_id, parsed.get("negativePrompt")),
    }


def call_openai_compatible(
    base_url: str,
    model: str,
    api_key: str,
    prompt_text: str,
    images: List[UploadFile],
    temperature: float,
    provider: str,
    family_id: str,
) -> Dict[str, str]:
    user_content: List[Dict[str, Any]] = []
    for img in images:
        raw = img.file.read()
        encoded = base64.b64encode(raw).decode("ascii")
        user_content.append(
            {
                "type": "input_image",
                "image_url": f"data:{img.content_type or 'image/png'};base64,{encoded}",
                "detail": "high",
            }
        )
    user_content.append(
        {
            "type": "input_text",
            "text": "Follow the system instructions and return the JSON object.",
        }
    )

    messages = [
        {"role": "system", "content": prompt_text},
        {"role": "user", "content": user_content},
    ]

    try:
        client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=httpx.Timeout(60.0),
        )
        response = client.responses.create(
            model=model,
            input=messages,
            temperature=temperature,
            store=False,
        )
    except Exception as exc:
        error_body = ""
        status_code = getattr(getattr(exc, "response", None), "status_code", None)
        if getattr(exc, "response", None) is not None:
            try:
                error_body = exc.response.text
            except Exception:
                error_body = ""
        status_line = f" {status_code}" if status_code else ""
        detail_line = f": {error_body}" if error_body else ""
        raise RuntimeError(f"{PROVIDER_LABELS[provider]} API HTTP{status_line}{detail_line} {exc}") from exc

    content_text = _extract_output_text(response)
    if not content_text:
        raise RuntimeError(f"No response from {PROVIDER_LABELS[provider]} model.")

    parsed = json.loads(content_text)
    return {
        "prompt": parsed.get("prompt") or "Failed to generate prompt.",
        "negativePrompt": ensure_negative_prompt(family_id, parsed.get("negativePrompt")),
    }


def generate_prompt(
    *,
    images: List[UploadFile],
    config: Dict[str, Any],
    provider: str,
    api_key: str,
    family: Dict[str, Any],
    checkpoint: Dict[str, Any],
) -> Dict[str, str]:
    prompt_text = build_prompt_text(config, family, checkpoint)
    print("[prompt-alchemy] build_prompt_text", prompt_text)
    temperature = 0.2 + (config.get("creativityLevel", 0.5) * 0.6)
    print(
        "[prompt-alchemy] generate_prompt",
        {
            "provider": provider,
            "model": family.get("id"),
            "checkpoint": checkpoint.get("id"),
            "temperature": temperature,
            "image_count": len(images),
        },
    )

    if provider == "gemini":
        result = call_gemini(images, prompt_text, temperature, api_key, family["id"])
    elif provider == "openai":
        result = call_openai_compatible(
            "https://api.openai.com/v1",
            OPENAI_MODEL,
            api_key,
            prompt_text,
            images,
            temperature,
            provider,
            family["id"],
        )
    elif provider == "grok":
        result = call_openai_compatible(
            "https://api.x.ai/v1",
            GROK_MODEL,
            api_key,
            prompt_text,
            images,
            temperature,
            provider,
            family["id"],
        )
    else:
        raise ValueError("Unsupported provider selected.")
    print("[prompt-alchemy] generate_prompt result", {**result, "temperature": temperature})
    return result
