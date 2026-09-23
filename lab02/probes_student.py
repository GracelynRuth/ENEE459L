from __future__ import annotations

import json
import re
from typing import Any
import sys

from env import Env, ModuleNotAvailable, getattr_path, read_text, unknown, major_minor

# The NVIDIA-built PyTorch wheels for Jetson carry a local version segment —
# the part after "+" — that names the NVIDIA container release. A wheel from
# plain PyPI has no such segment. This is a hint, not a proof, which is why the
# probe reports the tag itself alongside the interpretation.
_NV_LOCAL_TAG = re.compile(r"(?:^|\.)nv\d", re.IGNORECASE)

# `# R36 (release), REVISION: 5.0, GCID: ...`
_L4T_RELEASE = re.compile(r"R(\d+)\s*\(release\)", re.IGNORECASE)
_L4T_REVISION = re.compile(r"REVISION:\s*([\d.]+)")

# hepler function
def _split_local_version(raw: str) -> dict[str, Any]:
    if not raw:
        return {"raw": raw, "public": None, "local": None, "nvidia_build": False}
    public, sep, local = raw.partition("+")
    local = local if sep else None
    return {
        "raw": raw,
        "public": public or None,
        "local": local,
        "nvidia_build": bool(local and _NV_LOCAL_TAG.search(local)),
    }

# ---------------------------------------------------------------------------
# The probes.
# ---------------------------------------------------------------------------

def probe_torch(env: Env) -> dict[str, Any]:
    #write your code here
    # step 1: safely importing the module
    src = "import torch"
    try:
        torch = env.importer("torch")
    except ModuleNotAvailable as e:
        return unknown(env, f"torch is not importable: {e}")

    # step 2: Extract and Parse Version information
    raw = getattr_path(torch, "__version__")
    
    if raw:
        version = _split_local_version(str(raw))
    else:
        version = None

    # step 3 query CUDA hardware avialability
    attr = getattr_path(torch, "cuda.is_available")
    cuda_callable = callable(attr)
    cuda_available = False
    if cuda_callable:
        cuda_available = bool(attr())
    else:
        cuda_available = None

    # step 4 query the primary device name
    device = None
    if cuda_available:
        method = getattr_path(torch, "cuda.get_device_name")
        if method and callable(method):
            device = method(0)
    
    out = {
        "value": raw,
        "source": src,
        "status": "ok" if raw else unknown(src, "issue"),
        "version": version,
        "cuda_available": cuda_available,
        "cuda_version": getattr_path(torch, "version.cuda"),
        "device_name": device
    }

    if not raw:
        out["detail"] = "torch imported but exposes no __version__"
        return out
    
    # step 6 determine and attached env diagnosis
    nv = version["nvidia_build"] if version else False

    if cuda_available:
        out["diagnosis"] = "torch is installed and sees the GPU"
    elif not cuda_available:
        out["diagnosis"] = "torch is installed but does not epxose torch.cuda.is_available"
    elif nv:
        out["diagnosis"] = "his is an NVIDIA build but it cannot see the GPU — the wheel is right, so look at the driver stack, the container, or the user’s groups, not at pip"
    else:
        out["diagnosis"] = "this wheel has no NVIDIA local version tag and cannot see the GPU — it is almost certainly a stock PyPI wheel and must be replaced from the Jetson index"
    
    return out
    pass


def probe_cuda(env: Env) -> dict[str, Any]:
    #write your code here
    # step 1 define src manifest path
    src = "/usr/local/cuda/version.json"

    #step 2 read the raw file safelt
    raw = read_text(env.root, src)

    if not raw:
        return unknown(src, "CUDA toolkit manifest absent - no toolkit installed at /usr/local/cuda")

    # step 3 parse json manifest data
    try:
        data = json.loads(raw)
    except ValueError:
        return unknown(src, "CUDA toolkit manifest is present but not valid JSON")
    
    # extract and validate version string
    version = data.get("cuda", {}).get("version")

    if not version:
        return unknown(src, "manifest present but names no cuda version")
    
    return {
        "value": str(version),
        "source": str(src),
        "status": "ok",
        "line": major_minor(version)
    }
    pass


def probe_opencv(env: Env) -> dict[str, Any]:
    #write your code here
    # step 1 set the opencv src
    src = "import cv2"

    # step 2 safely import opencv
    try:
        cv2 = env.importer("cv2")
    except Exception as e:
        return unknown(src,f"cv2 is not importable: {e}")
    
    #step 3 extract the installed version
    raw = getattr_path(cv2, "__version__")

    # step 4 
    cuda_devices = None
    counter = getattr_path(cv2, "cuda.getCudaEnabledDeviceCount")
    detail = ""
    if callable(counter):
        devices = int(counter())
        cuda_devices = devices
        if devices:
            detail = f"built with CUDA, {devices} device(s) visible"
        else:
            detail = "the cv2.cuda namespace exists but reports no devices this is a non-CUDA build"
    else:
        detail = "no cv2.cuda namespace — a non-CUDA build, which is what JetPack ships"
    
    return {
        "value": raw,
        "source": src,
        "status": "ok" if raw else unknown(src, "issue"),
        "cuda_devices": cuda_devices,
        "cuda_enabled":bool(cuda_devices),
        "detail": detail
    }
    pass


def probe_tensorrt(env: Env) -> dict[str, Any]:
    # write your code here
    src = "import tensorrt"

    try:
        trt = env.importer("tensorrt")
    except ModuleNotAvailable as e:
        hint = ""
        if env.python.prefix and env.python.prefix != env.python.base_prefix:
            hint = " — you are inside a virtual environment, and TensorRT is a system package that a venv made without –system-site-packages cannot see"
            return unknown(src, f"tensorrt is not importable: {e}{hint}")
    
    raw = getattr_path(trt, "__version__")
    
    if not raw:
        return unknown(src, "tensorrt imported but exposes no __version__")
    
    return {
        "value": raw,
        "source": src,
        "status": "ok",
        "line": major_minor(str(raw))
    }
    pass


def probe_l4t(env: Env) -> dict[str, Any]:
    # write your code here
    src = "etc/nv_tegra_release"
    raw = read_text(env.root, src)

    if not raw:
        return unknown(src, "not a Jetson, or the L4T release file is absent")

    release = _L4T_RELEASE.search(raw)
    revision = _L4T_REVISION.search(raw)

    if not release or not revision:
        return unknown(src, f"release file present but unparseable: {raw.splitlines()[0][:80]}")
    
    version = f"{release.group(1)}.{revision.group(1)}"

    return {
        "value": version,
        "source": src,
        "status": "ok",
        "line": major_minor(version),
        "raw": raw.splitlines()[0]
    }
    
    pass

## for debugging - uncomment the following lines for debugging.
# if __name__ == "__main__":
#     env = Env.real()
#     out = probe_tensorrt(env)
#     print(out)

# for generating system_report.json
if __name__ == "__main__":
    # calling base environment
    env = Env.real()

    # testing probes
    report = {
        "probe_torch": probe_torch(env),
        "probe_cuda": probe_cuda(env),
        "probe_opencv": probe_opencv(env),
        "probe_tensorrt": probe_tensorrt(env),
        "probe_l4t": probe_l4t(env),
    }
    
    path = "system_report.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

