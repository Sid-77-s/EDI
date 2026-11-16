"""
integration/generation/providers/sdxl_local.py — ComfyUI Local SDXL Provider Adapter.

Features:
- Communicates with an external ComfyUI inference service via REST API.
- Never embeds inference inside camera or perception processes.
- Defaults to Stable Diffusion XL Base 1.0 (FP16) at 768x768.
- Evaluates SDXL Lightning as an accelerated 4-step/8-step checkpoint.
- Low-VRAM optimizations for 8GB GPU memory constraints.
- Text-to-image and img2img image editing support.
- Configurable host, ports, timeouts, and checkpoint identifiers.
"""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, Union
import cv2
import numpy as np

from integration.generation.providers.base import BaseImageProvider
from integration.generation.schemas import (
    GenerationMode,
    GenerationRequest,
    GenerationResult,
    JobState,
)

logger = logging.getLogger(__name__)

# Defaults
DEFAULT_COMFYUI_HOST = "http://127.0.0.1:8188"
DEFAULT_SDXL_CHECKPOINT = "sd_xl_base_1.0.safetensors"
DEFAULT_LIGHTNING_CHECKPOINT = "sdxl_lightning_4step.safetensors"
DEFAULT_TIMEOUT_S = 60.0
DEFAULT_POLL_INTERVAL_S = 0.5


class ComfyUIProvider(BaseImageProvider):
    """Adapter for separate local ComfyUI inference service."""

    def __init__(
        self,
        host: Optional[str] = None,
        checkpoint: Optional[str] = None,
        use_lightning: bool = False,
        timeout: float = DEFAULT_TIMEOUT_S,
        poll_interval: float = DEFAULT_POLL_INTERVAL_S,
        asset_dir: Union[str, Path] = "assets",
        http_client: Optional[Callable[[urllib.request.Request, float], bytes]] = None,
    ) -> None:
        super().__init__(name="sdxl-local")
        self.host = (host or os.environ.get("COMFYUI_HOST") or DEFAULT_COMFYUI_HOST).rstrip("/")
        self.use_lightning = use_lightning or (os.environ.get("COMFYUI_LIGHTNING", "").lower() in ("1", "true"))
        
        default_ckpt = DEFAULT_LIGHTNING_CHECKPOINT if self.use_lightning else DEFAULT_SDXL_CHECKPOINT
        self.checkpoint = checkpoint or os.environ.get("COMFYUI_CHECKPOINT") or default_ckpt
        
        self.timeout = timeout
        self.poll_interval = poll_interval
        self.asset_dir = Path(asset_dir)
        self.client_id = str(uuid.uuid4())
        self._http_client = http_client or self._default_http_request

    def _default_http_request(self, req: urllib.request.Request, timeout: float) -> bytes:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return response.read()

    def is_available(self) -> bool:
        """Checks if local ComfyUI instance is responding."""
        try:
            req = urllib.request.Request(f"{self.host}/system_stats", method="GET")
            self._http_client(req, 1.5)
            return True
        except Exception:
            return False

    def build_sdxl_workflow(
        self,
        prompt: str,
        negative_prompt: str = "ugly, blurry, low quality, distorted, extra limbs",
        width: int = 768,
        height: int = 768,
        seed: Optional[int] = None,
        steps: Optional[int] = None,
        cfg: Optional[float] = None,
        sampler_name: Optional[str] = None,
        scheduler: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Builds standard ComfyUI SDXL execution graph with 768x768 low-VRAM sizing.
        """
        actual_seed = seed if seed is not None else int(time.time() * 1000) % (2**31)

        # Lightning evaluation mode
        is_lightning = self.use_lightning or "lightning" in self.checkpoint.lower()
        actual_steps = steps if steps is not None else (4 if is_lightning else 20)
        actual_cfg = cfg if cfg is not None else (1.5 if is_lightning else 7.0)
        actual_sampler = sampler_name or ("euler" if is_lightning else "euler")
        actual_scheduler = scheduler or ("sgm_uniform" if is_lightning else "normal")

        workflow: Dict[str, Any] = {
            "3": {
                "class_type": "KSampler",
                "inputs": {
                    "cfg": actual_cfg,
                    "denoise": 1.0,
                    "latent_image": ["5", 0],
                    "model": ["4", 0],
                    "negative": ["7", 0],
                    "positive": ["6", 0],
                    "sampler_name": actual_sampler,
                    "scheduler": actual_scheduler,
                    "seed": actual_seed,
                    "steps": actual_steps,
                },
            },
            "4": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {"ckpt_name": self.checkpoint},
            },
            "5": {
                "class_type": "EmptyLatentImage",
                "inputs": {
                    "batch_size": 1,
                    # Low-VRAM 768x768 target
                    "height": height,
                    "width": width,
                },
            },
            "6": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "clip": ["4", 1],
                    "text": prompt,
                },
            },
            "7": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "clip": ["4", 1],
                    "text": negative_prompt,
                },
            },
            "8": {
                "class_type": "VAEDecode",
                "inputs": {
                    "samples": ["3", 0],
                    "vae": ["4", 2],
                },
            },
            "9": {
                "class_type": "SaveImage",
                "inputs": {
                    "filename_prefix": "MagicMirror",
                    "images": ["8", 0],
                },
            },
        }
        return workflow

    def generate(self, request: GenerationRequest) -> GenerationResult:
        """Executes text-to-image on local ComfyUI service."""
        start_time = time.monotonic()
        prompt_text = request.effective_prompt()

        width = request.width or 768
        height = request.height or 768

        workflow = self.build_sdxl_workflow(
            prompt=prompt_text,
            width=width,
            height=height,
            seed=request.seed,
            steps=request.steps,
        )

        try:
            prompt_id = self._submit_workflow(workflow)
            image_data = self._poll_and_fetch_output(prompt_id, start_time)

            np_arr = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)
            if img is None:
                raise RuntimeError("Failed to decode generated ComfyUI image bytes")

            file_path = self.save_image_to_disk(img, request.concept, self.asset_dir)
            latency = (time.monotonic() - start_time) * 1000.0

            return GenerationResult(
                concept=request.concept,
                visual_id=request.visual_id,
                image=img,
                image_path=file_path,
                status=JobState.READY,
                provider=self.name,
                model=self.checkpoint,
                latency_ms=latency,
                metadata={"prompt_id": prompt_id, "checkpoint": self.checkpoint, "lightning": self.use_lightning},
            )
        except Exception as e:
            latency = (time.monotonic() - start_time) * 1000.0
            logger.error("ComfyUI generation failed for '%s': %s", request.concept, e)
            return GenerationResult(
                concept=request.concept,
                visual_id=request.visual_id,
                status=JobState.FAILED,
                provider=self.name,
                model=self.checkpoint,
                latency_ms=latency,
                error_message=f"ComfyUI SDXL error: {e}",
            )

    def edit(self, request: GenerationRequest) -> GenerationResult:
        """Executes image edit / refinement via ComfyUI."""
        start_time = time.monotonic()
        if request.input_image is None:
            return GenerationResult(
                concept=request.concept,
                visual_id=request.visual_id,
                status=JobState.FAILED,
                provider=self.name,
                model=self.checkpoint,
                error_message="Image editing requested but no input_image provided",
            )

        try:
            # Upload image or pass in workflow
            upload_name = self._upload_image(request.input_image)
            
            # Build img2img workflow
            actual_seed = request.seed if request.seed is not None else int(time.time() * 1000) % (2**31)
            is_lightning = self.use_lightning or "lightning" in self.checkpoint.lower()
            actual_steps = request.steps if request.steps is not None else (4 if is_lightning else 20)
            actual_cfg = 1.5 if is_lightning else 7.0

            workflow = {
                "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": self.checkpoint}},
                "2": {"class_type": "LoadImage", "inputs": {"image": upload_name}},
                "3": {"class_type": "VAEEncode", "inputs": {"pixels": ["2", 0], "vae": ["1", 2]}},
                "4": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1], "text": request.effective_prompt()}},
                "5": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1], "text": "ugly, blurry, low quality"}},
                "6": {
                    "class_type": "KSampler",
                    "inputs": {
                        "cfg": actual_cfg,
                        "denoise": 0.65,
                        "latent_image": ["3", 0],
                        "model": ["1", 0],
                        "negative": ["5", 0],
                        "positive": ["4", 0],
                        "sampler_name": "euler",
                        "scheduler": "sgm_uniform" if is_lightning else "normal",
                        "seed": actual_seed,
                        "steps": actual_steps,
                    },
                },
                "7": {"class_type": "VAEDecode", "inputs": {"samples": ["6", 0], "vae": ["1", 2]}},
                "8": {"class_type": "SaveImage", "inputs": {"filename_prefix": "MagicMirror_Edit", "images": ["7", 0]}},
            }

            prompt_id = self._submit_workflow(workflow)
            image_data = self._poll_and_fetch_output(prompt_id, start_time)

            np_arr = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)
            if img is None:
                raise RuntimeError("Failed to decode edited ComfyUI image bytes")

            file_path = self.save_image_to_disk(img, request.concept, self.asset_dir, prefix="edit_")
            latency = (time.monotonic() - start_time) * 1000.0

            return GenerationResult(
                concept=request.concept,
                visual_id=request.visual_id,
                image=img,
                image_path=file_path,
                status=JobState.READY,
                provider=self.name,
                model=self.checkpoint,
                latency_ms=latency,
                metadata={"prompt_id": prompt_id, "mode": "edit"},
            )
        except Exception as e:
            latency = (time.monotonic() - start_time) * 1000.0
            return GenerationResult(
                concept=request.concept,
                visual_id=request.visual_id,
                status=JobState.FAILED,
                provider=self.name,
                model=self.checkpoint,
                latency_ms=latency,
                error_message=f"ComfyUI SDXL edit error: {e}",
            )

    def _submit_workflow(self, workflow: Dict[str, Any]) -> str:
        """Sends workflow to ComfyUI /prompt endpoint."""
        payload = json.dumps({"prompt": workflow, "client_id": self.client_id}).encode("utf-8")
        req = urllib.request.Request(
            f"{self.host}/prompt",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        resp = self._http_client(req, timeout=10.0)
        data = json.loads(resp.decode("utf-8"))
        prompt_id = data.get("prompt_id")
        if not prompt_id:
            raise RuntimeError(f"ComfyUI did not return prompt_id: {data}")
        return str(prompt_id)

    def _poll_and_fetch_output(self, prompt_id: str, start_time: float) -> bytes:
        """Polls /history/{prompt_id} until image is generated, then downloads image bytes."""
        history_url = f"{self.host}/history/{prompt_id}"
        
        while (time.monotonic() - start_time) < self.timeout:
            try:
                req = urllib.request.Request(history_url, method="GET")
                resp = self._http_client(req, timeout=5.0)
                data = json.loads(resp.decode("utf-8"))
                
                if prompt_id in data:
                    item = data[prompt_id]
                    outputs = item.get("outputs", {})
                    for node_id, node_out in outputs.items():
                        images = node_out.get("images", [])
                        if images:
                            img_info = images[0]
                            fname = img_info.get("filename")
                            subfolder = img_info.get("subfolder", "")
                            img_type = img_info.get("type", "output")
                            return self._fetch_image(fname, subfolder, img_type)
            except Exception as e:
                logger.debug("ComfyUI history poll transient error: %s", e)

            time.sleep(self.poll_interval)

        raise TimeoutError(f"ComfyUI job {prompt_id} timed out after {self.timeout:.1f}s")

    def _fetch_image(self, filename: str, subfolder: str, img_type: str) -> bytes:
        """Downloads image bytes from ComfyUI /view endpoint."""
        params = urllib.parse.urlencode({
            "filename": filename,
            "subfolder": subfolder,
            "type": img_type,
        })
        url = f"{self.host}/view?{params}"
        req = urllib.request.Request(url, method="GET")
        return self._http_client(req, timeout=15.0)

    def _upload_image(self, image_input: Union[np.ndarray, str, Path]) -> str:
        """Uploads image to ComfyUI /upload/image endpoint (multipart/form-data)."""
        filename = f"upload_{uuid.uuid4().hex[:8]}.png"
        
        if isinstance(image_input, (str, Path)):
            with open(str(image_input), "rb") as f:
                img_bytes = f.read()
        elif isinstance(image_input, np.ndarray):
            success, enc = cv2.imencode(".png", image_input)
            if not success:
                raise ValueError("Could not encode image array to PNG")
            img_bytes = enc.tobytes()
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

        boundary = f"----WebKitFormBoundary{uuid.uuid4().hex}"
        body = bytearray()
        body.extend(f"--{boundary}\r\n".encode("utf-8"))
        body.extend(f'Content-Disposition: form-data; name="image"; filename="{filename}"\r\n'.encode("utf-8"))
        body.extend(b"Content-Type: image/png\r\n\r\n")
        body.extend(img_bytes)
        body.extend(b"\r\n")
        body.extend(f"--{boundary}--\r\n".encode("utf-8"))

        req = urllib.request.Request(
            f"{self.host}/upload/image",
            data=bytes(body),
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
            method="POST",
        )
        resp = self._http_client(req, timeout=15.0)
        data = json.loads(resp.decode("utf-8"))
        return data.get("name", filename)

