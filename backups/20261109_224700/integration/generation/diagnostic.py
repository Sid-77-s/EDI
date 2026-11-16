"""
integration/generation/diagnostic.py — Hardware, VRAM, and Live Provider Connectivity Diagnostic.

Reports:
1. GPU hardware specs, Total VRAM, Free/Used VRAM.
2. Status of real external providers (FLUX Cloud, ComfyUI SDXL Local).
3. Real latency and VRAM footprint of locally executed generation.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import time
import urllib.request
from typing import Any, Dict

from integration.generation.engine import GenerationEngine
from integration.generation.providers.flux_cloud import FluxCloudProvider
from integration.generation.providers.procedural import ProceduralVisualProvider
from integration.generation.providers.sdxl_local import ComfyUIProvider
from integration.generation.schemas import GenerationRequest

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("generation_diagnostic")


def get_gpu_telemetry() -> Dict[str, Any]:
    """Queries NVIDIA-SMI for precise GPU and VRAM metrics."""
    telemetry: Dict[str, Any] = {
        "gpu_available": False,
        "name": "Unknown",
        "driver_version": "Unknown",
        "cuda_version": "Unknown",
        "total_vram_mb": 0,
        "used_vram_mb": 0,
        "free_vram_mb": 0,
    }
    try:
        cmd = [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total,memory.used,memory.free",
            "--format=csv,noheader,nounits",
        ]
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL)
        parts = [p.strip() for p in out.strip().split(",")]
        if len(parts) >= 5:
            telemetry["gpu_available"] = True
            telemetry["name"] = parts[0]
            telemetry["driver_version"] = parts[1]
            telemetry["total_vram_mb"] = float(parts[2])
            telemetry["used_vram_mb"] = float(parts[3])
            telemetry["free_vram_mb"] = float(parts[4])
    except Exception as e:
        logger.warning("NVIDIA-SMI query failed: %s", e)

    return telemetry


def check_live_providers() -> Dict[str, Any]:
    """Checks actual live connectivity for both real external providers."""
    results: Dict[str, Any] = {}

    # 1. FLUX Cloud (Black Forest Labs)
    bfl_key = os.environ.get("BFL_API_KEY") or os.environ.get("BLACK_FOREST_LABS_API_KEY")
    flux_prov = FluxCloudProvider(api_key=bfl_key)
    results["flux_cloud"] = {
        "api_key_configured": bool(bfl_key and bfl_key.strip()),
        "model": flux_prov.model_name,
        "base_url": flux_prov.base_url,
        "is_available": flux_prov.is_available(),
        "live_connection": False,
        "notes": "",
    }
    if not results["flux_cloud"]["api_key_configured"]:
        results["flux_cloud"]["notes"] = "BFL_API_KEY environment variable is not set. Cloud FLUX requires an API key."

    # 2. ComfyUI Local SDXL Service
    comfy_host = os.environ.get("COMFYUI_HOST", "http://127.0.0.1:8188")
    comfy_prov = ComfyUIProvider(host=comfy_host)
    comfy_is_up = comfy_prov.is_available()
    results["comfyui_local"] = {
        "host": comfy_host,
        "is_available": comfy_is_up,
        "live_connection": comfy_is_up,
        "checkpoint": comfy_prov.checkpoint,
        "notes": "ComfyUI service responded on /system_stats" if comfy_is_up else f"No ComfyUI service running at {comfy_host}",
    }

    # 3. Procedural Provider
    proc_prov = ProceduralVisualProvider()
    results["procedural"] = {
        "is_available": proc_prov.is_available(),
        "live_connection": True,
        "notes": "Always available locally (zero VRAM overhead)",
    }

    return results


def run_real_hardware_benchmark() -> Dict[str, Any]:
    """Measures real latency on the local hardware for available local generation."""
    telemetry_before = get_gpu_telemetry()
    proc = ProceduralVisualProvider()

    req = GenerationRequest(concept="water molecule", width=768, height=768)
    t0 = time.perf_counter()
    res = proc.generate(req)
    t1 = time.perf_counter()

    telemetry_after = get_gpu_telemetry()

    return {
        "concept": req.concept,
        "generation_success": res.is_success,
        "image_path": res.image_path,
        "measured_latency_ms": round((t1 - t0) * 1000.0, 2),
        "vram_delta_mb": round(telemetry_after.get("used_vram_mb", 0) - telemetry_before.get("used_vram_mb", 0), 2),
        "gpu_telemetry": telemetry_after,
    }


def run_diagnostic() -> Dict[str, Any]:
    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "hardware": get_gpu_telemetry(),
        "providers": check_live_providers(),
        "benchmark": run_real_hardware_benchmark(),
    }
    return report


if __name__ == "__main__":
    report = run_diagnostic()
    print("\n" + "=" * 60)
    print(" MAGIC MIRROR GENERATION ENGINE DIAGNOSTIC REPORT")
    print("=" * 60)
    print(f"GPU: {report['hardware']['name']}")
    print(f"Total VRAM: {report['hardware']['total_vram_mb']} MiB")
    print(f"Used VRAM:  {report['hardware']['used_vram_mb']} MiB")
    print(f"Free VRAM:  {report['hardware']['free_vram_mb']} MiB")
    print("-" * 60)
    print("LIVE PROVIDER STATUS:")
    for prov_name, details in report["providers"].items():
        print(f"  • {prov_name}:")
        print(f"      Available: {details['is_available']}")
        print(f"      Live Connection: {details['live_connection']}")
        print(f"      Notes: {details['notes']}")
    print("-" * 60)
    print("HARDWARE BENCHMARK (Procedural):")
    bm = report["benchmark"]
    print(f"  Concept: {bm['concept']}")
    print(f"  Success: {bm['generation_success']}")
    print(f"  Measured Latency: {bm['measured_latency_ms']} ms")
    print(f"  VRAM Delta: {bm['vram_delta_mb']} MiB")
    print(f"  Saved Asset: {bm['image_path']}")
    print("=" * 60 + "\n")

