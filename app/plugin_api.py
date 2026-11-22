# ─────────────────────────────────────────────────────────────
#  Bölüm:    Eklenti Köprü Uçları (public /v1)
#  Dosya:    app/plugin_api.py
#  Amaç:     Kurulu ve çalışan eklentilerin HTTP servislerini public
#            /v1/* uçlarına köprüler. Eklentiler ayrı alt süreç olarak
#            çalışır; bu modül yalnızca doğru portu bulup isteği yönlendirir.
#  Uçlar:    POST /v1/images/upscale   -> {"image": <b64 png>}  (category=upscale)
#            POST /v1/videos/restore   -> {"video": <b64 mp4>}  (category=restore)
#            POST /v1/videos/lipsync   -> {"image"/"video", "audio"} -> {"video": <b64>}
#  Not:      Yanıt biçimi eklentinin döndüğü b64 JSON'dur (OpenAI-tarzı değil).
#            İstekler accelerator kuyruğundan geçer (VProvider seviyesinde
#            serialize); eklenti alt süreçleri ayrı süreç olduğundan GPU
#            üzerindeki gerçek eşzamanlılık motorlara bağlıdır.
#  Kullanım: app/main.py içinde app.include_router(plugin_router)
# ─────────────────────────────────────────────────────────────

from typing import Optional

import httpx
from fastapi import APIRouter, Depends, HTTPException
from fastapi.routing import APIRouter as _Router
from pydantic import BaseModel, Field

from app import plugins
from app.gateway import ApiContext, api_auth
from app.openai_api import _openai_error
from app.runtime_coord import accelerator

router: _Router = APIRouter(prefix="/v1", dependencies=[Depends(api_auth)])

_PROXY_TIMEOUT = 900.0   # ağır çıkarımlar (restore/lipsync) için cömert zaman aşımı


class UpscaleRequest(BaseModel):
    image: str = Field(..., description="PNG görseli base64")


class RestoreRequest(BaseModel):
    video: str = Field(..., description="MP4 video base64")


class LipsyncRequest(BaseModel):
    audio: str = Field(..., description="Ses (wav/mp3) base64")
    video: Optional[str] = Field(None, description="Video base64 (opsiyonel)")
    image: Optional[str] = Field(None, description="Görsel base64 (opsiyonel)")


def _require_running_plugin(category: str, prefer: tuple[str, ...] = ()) -> tuple[str, int]:
    """Kategoride çalışan bir eklenti portu ister; yoksa OpenAI hatası fırlatır."""
    found = plugins.running_port_for_category(category, prefer=prefer)
    if found is None:
        raise _openai_error(
            503,
            f"'{category}' kategorisinde çalışan bir eklenti yok. "
            "Panel > Eklentiler'den kurup başlatın.",
            code="plugin_not_running",
        )
    return found


async def _http_post_json(url: str, payload: dict) -> "httpx.Response":
    """Eklenti servisine JSON POST atar (test için ayrı tutulur)."""
    async with httpx.AsyncClient(timeout=_PROXY_TIMEOUT) as client:
        return await client.post(url, json=payload)


async def _proxy_to_plugin(port: int, path: str, payload: dict) -> dict:
    """İsteği çalışan eklenti servisine yönlendirir (GPU kuyruğu altında)."""
    url = f"http://127.0.0.1:{port}{path}"
    async with accelerator.acquire(f"plugin:{path}"):
        try:
            resp = await _http_post_json(url, payload)
        except httpx.HTTPError as exc:
            raise _openai_error(
                502, f"Eklenti servisine ulaşılamadı: {exc}", code="plugin_unreachable"
            ) from exc

    if resp.status_code != 200:
        try:
            detail = resp.json().get("error", resp.text)
        except ValueError:
            detail = resp.text
        raise _openai_error(
            502, f"Eklenti hatası: {detail}", code="plugin_error"
        )
    try:
        return resp.json()
    except ValueError as exc:
        raise _openai_error(
            502, "Eklenti geçersiz yanıt döndürdü.", code="plugin_bad_response"
        ) from exc


@router.post("/images/upscale")
async def upscale_image(req: UpscaleRequest, ctx: ApiContext = Depends(api_auth)):
    """Görseli büyütür (Real-ESRGAN). Yanıt: {"image": <b64 png>}."""
    plugin_id, port = _require_running_plugin("upscale")
    ctx.enforce_model(plugin_id)
    result = await _proxy_to_plugin(port, "/upscale", {"image": req.image})
    ctx.charge("upscale", plugin_id, images=1)
    return result


@router.post("/videos/restore")
async def restore_video(req: RestoreRequest, ctx: ApiContext = Depends(api_auth)):
    """Videoyu onarır/yükseltir (SeedVR2). Yanıt: {"video": <b64 mp4>}."""
    plugin_id, port = _require_running_plugin("restore")
    ctx.enforce_model(plugin_id)
    result = await _proxy_to_plugin(port, "/restore", {"video": req.video})
    ctx.charge("restore", plugin_id, videos=1)
    return result


@router.post("/videos/lipsync")
async def lipsync_video(req: LipsyncRequest, ctx: ApiContext = Depends(api_auth)):
    """Ses ile video/görseli senkronlar (MuseTalk/LatentSync). Yanıt: {"video": <b64>}."""
    if not req.video and not req.image:
        raise _openai_error(
            400, "'video' ya da 'image' alanlarından biri gerekli.",
            param="video", code="missing_input",
        )
    plugin_id, port = _require_running_plugin("lipsync", prefer=("latent_sync", "muse_talk"))
    ctx.enforce_model(plugin_id)
    payload = {"audio": req.audio}
    if req.video:
        payload["video"] = req.video
    if req.image:
        payload["image"] = req.image
    result = await _proxy_to_plugin(port, "/lipsync", payload)
    ctx.charge("lipsync", plugin_id, videos=1)
    return result
