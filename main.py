import uuid
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from config import FOCUS_ASPECTS_LIST, MODEL_DATA, PROVIDER_LABELS
from prompting import generate_prompt, get_checkpoint, get_family, resolve_api_key

APP_INSTANCE_ID = uuid.uuid4().hex
BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"

app = FastAPI()
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")

templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


def update_env_file(key: str, value: str) -> None:
    lines: List[str] = []
    if ENV_PATH.exists():
        lines = ENV_PATH.read_text(encoding="utf-8").splitlines()

    updated = False
    new_lines: List[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            new_lines.append(line)
            continue
        existing_key, _ = stripped.split("=", 1)
        if existing_key.strip() == key:
            new_lines.append(f"{key}={value}")
            updated = True
        else:
            new_lines.append(line)

    if not updated:
        if new_lines and new_lines[-1] != "":
            new_lines.append("")
        new_lines.append(f"{key}={value}")

    ENV_PATH.write_text("\n".join(new_lines) + "\n", encoding="utf-8")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    default_family = MODEL_DATA["model_families"][0]
    default_checkpoint = default_family["checkpoints"][0]
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "model_data": MODEL_DATA,
            "focus_aspects": FOCUS_ASPECTS_LIST,
            "default_family": default_family,
            "default_checkpoint": default_checkpoint,
            "app_instance_id": APP_INSTANCE_ID,
        },
    )


@app.get("/partials/checkpoints", response_class=HTMLResponse)
async def checkpoints_partial(request: Request, family_id: str) -> HTMLResponse:
    family = get_family(family_id) or MODEL_DATA["model_families"][0]
    checkpoint = family["checkpoints"][0]
    return templates.TemplateResponse(
        "partials/checkpoints.html",
        {
            "request": request,
            "family": family,
            "selected_checkpoint": checkpoint,
            "include_oob": True,
        },
    )


@app.post("/api/keys")
async def update_api_key(
    provider: str = Form(...),
    api_key: str = Form(""),
) -> Dict[str, str]:
    env_key_map = {
        "gemini": "GEMINI_API_KEY",
        "openai": "OPENAI_API_KEY",
        "grok": "GROK_API_KEY",
    }
    key_name = env_key_map.get(provider)
    if key_name and api_key:
        update_env_file(key_name, api_key)
    return {"status": "ok"}


@app.get("/api/keys")
async def load_api_key(provider: str) -> Dict[str, str]:
    env_key_map = {
        "gemini": "GEMINI_API_KEY",
        "openai": "OPENAI_API_KEY",
        "grok": "GROK_API_KEY",
    }
    key_name = env_key_map.get(provider)
    if not key_name:
        return {"api_key": ""}
    if not ENV_PATH.exists():
        return {"api_key": ""}
    env_data = ENV_PATH.read_text(encoding="utf-8").splitlines()
    for line in env_data:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        env_key, value = stripped.split("=", 1)
        if env_key.strip() == key_name:
            print(f"[prompt-alchemy] selected key {key_name} {value.strip()}")
            return {"api_key": value.strip()}
    return {"api_key": ""}


@app.post("/generate", response_class=HTMLResponse)
async def generate_prompt_handler(
    request: Request,
    provider: str = Form("gemini"),
    api_key: str = Form(""),
    model_family_id: str = Form(...),
    checkpoint_id: str = Form(...),
    focus_aspects: List[str] = Form([]),
    creativity_level: float = Form(0.5),
    additional_context: str = Form(""),
    auxiliary_upscaler: Optional[str] = Form(None),
    auxiliary_face_fixer: Optional[str] = Form(None),
    auxiliary_control_model: Optional[str] = Form(None),
    images: List[UploadFile] = File([]),
) -> HTMLResponse:
    if not images:
        return templates.TemplateResponse(
            "partials/prompt_result.html",
            {
                "request": request,
                "error": "Please upload at least one reference image.",
                "result": None,
            },
        )

    family = get_family(model_family_id)
    if not family:
        return templates.TemplateResponse(
            "partials/prompt_result.html",
            {
                "request": request,
                "error": "Invalid model configuration.",
                "result": None,
            },
        )
    checkpoint = get_checkpoint(family, checkpoint_id)
    if not checkpoint:
        checkpoint = family["checkpoints"][0]

    key = resolve_api_key(provider, api_key, str(ENV_PATH))
    if not key:
        return templates.TemplateResponse(
            "partials/prompt_result.html",
            {
                "request": request,
                "error": f"API Key is missing. Please enter your {PROVIDER_LABELS.get(provider, 'selected')} API Key.",
                "result": None,
            },
        )

    config = {
        "modelFamilyId": family["id"],
        "checkpointId": checkpoint["id"],
        "focusAspects": focus_aspects,
        "creativityLevel": creativity_level,
        "additionalContext": additional_context,
        "auxiliary": {
            "upscaler": auxiliary_upscaler or None,
            "faceFixer": auxiliary_face_fixer or None,
            "controlModel": auxiliary_control_model or None,
        },
    }

    print(
        "[prompt-alchemy] generate request",
        {
            "provider": provider,
            "model_family_id": model_family_id,
            "checkpoint_id": checkpoint_id,
            "focus_aspects": focus_aspects,
            "creativity_level": creativity_level,
            "additional_context": additional_context,
            "auxiliary_upscaler": auxiliary_upscaler,
            "auxiliary_face_fixer": auxiliary_face_fixer,
            "auxiliary_control_model": auxiliary_control_model,
            "image_count": len(images),
        },
    )

    try:
        result = generate_prompt(
            images=images,
            config=config,
            provider=provider,
            api_key=key,
            family=family,
            checkpoint=checkpoint,
        )
    except Exception:
        return templates.TemplateResponse(
            "partials/prompt_result.html",
            {
                "request": request,
                "error": "Failed to generate prompt. Please try again.",
                "result": None,
            },
        )

    prompt_result = {
        "prompt": result["prompt"],
        "negativePrompt": result.get("negativePrompt") or "",
        "modelFamily": family["label"],
        "checkpoint": checkpoint["label"],
    }

    return templates.TemplateResponse(
        "partials/prompt_result.html",
        {
            "request": request,
            "error": None,
            "result": prompt_result,
        },
    )
