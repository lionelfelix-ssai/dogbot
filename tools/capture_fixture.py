#!/usr/bin/env python3
"""Capture raw YOLO26 Hailo outputs for one image into an .npz test fixture.

Runs on the Pi with system python3 (hailo_platform + numpy + cv2):

    python3 capture_fixture.py IMAGE [--hef /home/admin/models/yolo26s.hef]
                               [--out yolo26_outputs.npz] [--float16]

The image is letterboxed to 640x640 RGB (gray 114 padding), fed as UINT8 NHWC,
and every output vstream is read back as FLOAT32 (dequantized). The .npz holds
one array per output layer (keyed by layer name, batch dim dropped), plus
`_meta_scale`, `_meta_pad` (letterbox params) and `_meta_image` (source name).
It also prints per-layer shape and value range so the head layout can be read.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def letterbox(img: np.ndarray, size: int = 640) -> tuple[np.ndarray, float, tuple[int, int]]:
    h, w = img.shape[:2]
    scale = min(size / w, size / h)
    nw, nh = round(w * scale), round(h * scale)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    out = np.full((size, size, 3), 114, dtype=np.uint8)
    px, py = (size - nw) // 2, (size - nh) // 2
    out[py:py + nh, px:px + nw] = resized
    return out, scale, (px, py)


def run(hef_path: str, frame: np.ndarray) -> dict[str, np.ndarray]:
    from hailo_platform import (HEF, ConfigureParams, FormatType, HailoStreamInterface,
                                InferVStreams, InputVStreamParams, OutputVStreamParams,
                                VDevice)

    hef = HEF(hef_path)
    with VDevice() as target:
        cfg = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
        ng = target.configure(hef, cfg)[0]
        in_params = InputVStreamParams.make(ng, format_type=FormatType.UINT8)
        out_params = OutputVStreamParams.make(ng, format_type=FormatType.FLOAT32)
        in_name = hef.get_input_vstream_infos()[0].name
        with InferVStreams(ng, in_params, out_params) as pipe:
            with ng.activate(ng.create_params()):
                res = pipe.infer({in_name: frame[None, ...]})
    return {k: np.asarray(v)[0] for k, v in res.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--hef", default="/home/admin/models/yolo26s.hef")
    ap.add_argument("--out", default="yolo26_outputs.npz")
    ap.add_argument("--float16", action="store_true", help="store arrays as float16")
    args = ap.parse_args()

    bgr = cv2.imread(args.image)
    if bgr is None:
        raise SystemExit(f"cannot read {args.image}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    frame, scale, pad = letterbox(rgb)
    outputs = run(args.hef, np.ascontiguousarray(frame))

    for name, arr in sorted(outputs.items()):
        print(f"{name:28s} shape={arr.shape} dtype={arr.dtype} "
              f"min={arr.min():.4f} max={arr.max():.4f} mean={arr.mean():.4f}")
        if arr.shape[-1] == 80:
            flat = arr.reshape(-1, 80)
            best = flat.max(axis=0)
            top = np.argsort(best)[::-1][:5]
            print("    top classes:", ", ".join(f"{c}:{best[c]:.3f}" for c in top))

    dtype = np.float16 if args.float16 else np.float32
    data = {k: v.astype(dtype) for k, v in outputs.items()}
    data["_meta_scale"] = np.array(scale, dtype=np.float32)
    data["_meta_pad"] = np.array(pad, dtype=np.int32)
    data["_meta_image"] = np.array(Path(args.image).name)
    np.savez_compressed(args.out, **data)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
