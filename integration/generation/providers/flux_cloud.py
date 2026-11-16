"""
integration/generation/providers/flux_cloud.py — Black Forest Labs (FLUX) Cloud Provider.

Supports:
- Configurable model identifier (defaults to FLUX.2 Klein 4B: 'flux.2-klein-4b', supports 'flux-pro', etc.)
- Text-to-image and Image editing
- Credentials loaded from environment variables (BFL_API_KEY, BLACK_FOREST_LABS_API_KEY)
- Request timeouts, retry logic with exponential backoff
- Async job status tracking (polling get_result)
- Clean error reporting and automatic asset saving
"""

from __future__ import annotations

import base64
import json
import logging
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union
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
DEFAULT_BFL_MODEL = "flux.2-klein-4b"
DEFAULT_BFL_EDIT_MODEL = "flux-pro-1.0-fill"
DEFAULT_BFL_BASE_URL = "https://api.bfl.ml/v1"
DEFAULT_TIMEOUT_S = 30.0
DEFAULT_POLL_INTERVAL_S = 1.0
DEFAULT_MAX_RETRIES = 3


class FluxCloudProvider(BaseImageProvider):
    """Client adapter for Black Forest Labs (FLUX) API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        edit_model_name: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = DEFAULT_TIMEOUT_S,
        poll_interval: float = DEFAULT_POLL_INTERVAL_S,
        max_retries: int = DEFAULT_MAX_RETRIES,
        asset_dir: Union[str, Path] = "assets",
        http_client: Optional[Callable[[urllib.request.Request, float], bytes]] = None,
    ) -> None:
        super().__init__(name="flux-cloud")
        # Load API key strictly from param or environment, NEVER hardcode
        self._api_key = (
            api_key
            or os.environ.get("BFL_API_KEY")
            or os.environ.get("BLACK_FOREST_LABS_API_KEY")
        )
        self.model_name = model_name or os.environ.get("BFL_MODEL") or DEFAULT_BFL_MODEL
        self.edit_model_name = (
            edit_model_name
            or os.environ.get("BFL_EDIT_MODEL")
            or DEFAULT_BFL_EDIT_MODEL
        )
        self.base_url = (base_url or os.environ.get("BFL_BASE_URL") or DEFAULT_BFL_BASE_URL).rstrip("/")
        self.timeout = timeout
        self.poll_interval = poll_interval
        self.max_retries = max_retries
        self.asset_dir = Path(asset_dir)
        self._http_client = http_client or self._default_http_request

    @property
    def api_key(self) -> Optional[str]:
        return self._api_key

    def set_api_key(self, key: str) -> None:
        self._api_key = key

    def is_available(self) -> bool:
        """Available if API key is present."""
        return bool(self._api_key and self._api_key.strip())

    def _default_http_request(self, req: urllib.request.Request, timeout: float) -> bytes:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return response.read()

    def _execute_with_retry(
        self,
        make_req: Callable[[], urllib.request.Request],
        timeout: float,
    ) -> bytes:
        """Executes HTTP request with exponential backoff for 429/5xx errors."""
        attempt = 0
        backoff = 1.0
        last_exc: Optional[Exception] = None

        while attempt < self.max_retries:
            attempt += 1
            try:
                req = make_req()
                return self._http_client(req, timeout)
            except urllib.error.HTTPError as e:
                last_exc = e
                # Retry on rate limit (429) or transient server errors (500, 502, 503, 504)
                if e.code in (429, 500, 502, 503, 504) and attempt < self.max_retries:
                    logger.warning("BFL HTTP %d, retrying attempt %d in %.1fs: %s", e.code, attempt, backoff, e.reason)
                    time.sleep(backoff)
                    backoff *= 2.0
                    continue
                err_body = ""
                try:
                    err_body = e.read().decode("utf-8", errors="replace")
                except Exception:
                    pass
                raise RuntimeError(f"BFL HTTP {e.code}: {e.reason} ({err_body})") from e
            except urllib.error.URLError as e:
                last_exc = e
                if attempt < self.max_retries:
                    logger.warning("BFL URLError, retrying attempt %d in %.1fs: %s", attempt, backoff, e.reason)
                    time.sleep(backoff)
                    backoff *= 2.0
                    continue
                raise RuntimeError(f"BFL connection failed: {e.reason}") from e
            except Exception as e:
                raise RuntimeError(f"BFL unexpected request error: {e}") from e

        raise RuntimeError(f"BFL request failed after {self.max_retries} attempts: {last_exc}")

    def generate(self, request: GenerationRequest) -> GenerationResult:
        """Text-to-image generation via FLUX cloud."""
        start_time = time.monotonic()
        if not self.is_available():
            return GenerationResult(
                concept=request.concept,
                visual_id=request.visual_id,
                status=JobState.FAILED,
                provider=self.name,
                model=self.model_name,
                error_message="BFL_API_KEY environment variable is not configured",
            )

        endpoint_url = f"{self.base_url}/{self.model_name}"
        payload = {
            "prompt": request.effective_prompt(),
            "width": max(256, min(1536, request.width)),
            "height": max(256, min(1536, request.height)),
        }
        if request.seed is not None:
            payload["seed"] = request.seed
        if request.steps is not None:
            payload["steps"] = request.steps

        try:
            job_id = self._submit_task(endpoint_url, payload)
            image, sample_url = self._poll_job(job_id, start_time)
            
            # Save asset to local disk
            file_path = self.save_image_to_disk(image, request.concept, self.asset_dir)
            latency = (time.monotonic() - start_time) * 1000.0

            return GenerationResult(
                concept=request.concept,
                visual_id=request.visual_id,
                image=image,
                image_path=file_path,
                status=JobState.READY,
                provider=self.name,
                model=self.model_name,
                latency_ms=latency,
                metadata={"job_id": job_id, "sample_url": sample_url, "prompt": payload["prompt"]},
            )
        except Exception as e:
            latency = (time.monotonic() - start_time) * 1000.0
            logger.error("FLUX generation failed for '%s': %s", request.concept, e)
            return GenerationResult(
                concept=request.concept,
                visual_id=request.visual_id,
                status=JobState.FAILED,
                provider=self.name,
                model=self.model_name,
                latency_ms=latency,
                error_message=str(e),
            )

    def edit(self, request: GenerationRequest) -> GenerationResult:
        """Image editing via FLUX cloud fill/edit endpoint."""
        start_time = time.monotonic()
        if not self.is_available():
            return GenerationResult(
                concept=request.concept,
                visual_id=request.visual_id,
                status=JobState.FAILED,
                provider=self.name,
                model=self.edit_model_name,
                error_message="BFL_API_KEY environment variable is not configured",
            )

        if request.input_image is None:
            return GenerationResult(
                concept=request.concept,
                visual_id=request.visual_id,
                status=JobState.FAILED,
                provider=self.name,
                model=self.edit_model_name,
                error_message="Image editing requested but no input_image provided",
            )

        # Encode input image as base64
        base64_img = self._encode_image_b64(request.input_image)
        endpoint_url = f"{self.base_url}/{self.edit_model_name}"
        payload = {
            "prompt": request.effective_prompt(),
            "image": base64_img,
        }

        try:
            job_id = self._submit_task(endpoint_url, payload)
            image, sample_url = self._poll_job(job_id, start_time)
            
            file_path = self.save_image_to_disk(image, request.concept, self.asset_dir, prefix="edit_")
            latency = (time.monotonic() - start_time) * 1000.0

            return GenerationResult(
                concept=request.concept,
                visual_id=request.visual_id,
                image=image,
                image_path=file_path,
                status=JobState.READY,
                provider=self.name,
                model=self.edit_model_name,
                latency_ms=latency,
                metadata={"job_id": job_id, "sample_url": sample_url, "mode": "edit"},
            )
        except Exception as e:
            latency = (time.monotonic() - start_time) * 1000.0
            logger.error("FLUX image editing failed for '%s': %s", request.concept, e)
            return GenerationResult(
                concept=request.concept,
                visual_id=request.visual_id,
                status=JobState.FAILED,
                provider=self.name,
                model=self.edit_model_name,
                latency_ms=latency,
                error_message=str(e),
            )

    def _submit_task(self, url: str, payload: dict) -> str:
        """Submits prompt task to BFL endpoint and retrieves task id."""
        body_bytes = json.dumps(payload).encode("utf-8")

        def _make_req():
            req = urllib.request.Request(
                url,
                data=body_bytes,
                headers={
                    "Content-Type": "application/json",
                    "x-key": self._api_key or "",
                    "User-Agent": "MagicMirror-FluxClient/1.0",
                },
                method="POST",
            )
            return req

        resp_bytes = self._execute_with_retry(_make_req, timeout=15.0)
        resp_data = json.loads(resp_bytes.decode("utf-8"))
        job_id = resp_data.get("id")
        if not job_id:
            raise RuntimeError(f"BFL response missing 'id': {resp_data}")
        return str(job_id)

    def _poll_job(self, job_id: str, start_time: float) -> Tuple[np.ndarray, str]:
        """Polls get_result endpoint until job is Ready or timed out."""
        poll_url = f"{self.base_url}/get_result?id={urllib.parse.quote(job_id)}"

        def _make_req():
            return urllib.request.Request(
                poll_url,
                headers={
                    "x-key": self._api_key or "",
                    "User-Agent": "MagicMirror-FluxClient/1.0",
                },
                method="GET",
            )

        while (time.monotonic() - start_time) < self.timeout:
            resp_bytes = self._execute_with_retry(_make_req, timeout=10.0)
            data = json.loads(resp_bytes.decode("utf-8"))
            status = data.get("status")

            if status == "Ready":
                result_obj = data.get("result", {})
                sample_url = result_obj.get("sample")
                if not sample_url:
                    raise RuntimeError(f"BFL job ready but missing result.sample URL: {data}")
                
                # Download sample image
                img_data = self._download_image(sample_url)
                np_arr = np.frombuffer(img_data, np.uint8)
                img = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)
                if img is None:
                    raise RuntimeError("Failed to decode downloaded BFL image bytes")
                return img, sample_url

            elif status in ("Error", "Request Moderated", "Content Moderated"):
                detail = data.get("detail") or data.get("error") or status
                raise RuntimeError(f"BFL job {job_id} failed with status '{status}': {detail}")

            time.sleep(self.poll_interval)

        raise TimeoutError(f"FLUX job {job_id} timed out after {self.timeout:.1f}s")

    def _download_image(self, url: str) -> bytes:
        """Download raw image bytes from result URL."""
        def _make_req():
            return urllib.request.Request(
                url,
                headers={"User-Agent": "MagicMirror-FluxClient/1.0"},
                method="GET",
            )
        return self._execute_with_retry(_make_req, timeout=20.0)

    def _encode_image_b64(self, img_input: Union[np.ndarray, str, Path]) -> str:
        """Convert image ndarray or path to base64 string."""
        if isinstance(img_input, (str, Path)):
            with open(str(img_input), "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        elif isinstance(img_input, np.ndarray):
            success, enc = cv2.imencode(".png", img_input)
            if not success:
                raise ValueError("Could not encode image array to PNG")
            return base64.b64encode(enc.tobytes()).decode("utf-8")
        raise TypeError(f"Unsupported image type: {type(img_input)}")

