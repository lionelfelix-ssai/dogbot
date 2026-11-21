"""Hailo-8L inference for the YOLO26 HEF.

Same hailo_platform calls as tools/capture_fixture.py (HailoRT 4.20,
InferVStreams), but the device, network group and vstreams are opened once
and kept open, so each infer() is one blocking round trip.

Input: UINT8 NHWC 640x640x3 RGB. Outputs: dict layer name -> FLOAT32 array
(dequantized, batch dim dropped), ready for dogbot.yolo26.decode_yolo26.
"""
from __future__ import annotations

import contextlib

import numpy as np


class HailoDetector:
    def __init__(self, hef_path: str):
        from hailo_platform import (HEF, ConfigureParams, FormatType, HailoStreamInterface,
                                    InferVStreams, InputVStreamParams, OutputVStreamParams,
                                    VDevice)

        self._stack = contextlib.ExitStack()
        try:
            hef = HEF(hef_path)
            target = self._stack.enter_context(VDevice())
            cfg = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
            ng = target.configure(hef, cfg)[0]
            in_params = InputVStreamParams.make(ng, format_type=FormatType.UINT8)
            out_params = OutputVStreamParams.make(ng, format_type=FormatType.FLOAT32)
            info = hef.get_input_vstream_infos()[0]
            self._in_name = info.name
            self.input_shape = tuple(info.shape)  # (640, 640, 3)
            self._pipe = self._stack.enter_context(InferVStreams(ng, in_params, out_params))
            self._stack.enter_context(ng.activate(ng.create_params()))
        except BaseException:
            self._stack.close()
            raise

    def infer(self, rgb640: np.ndarray) -> dict[str, np.ndarray]:
        if rgb640.shape != self.input_shape or rgb640.dtype != np.uint8:
            raise ValueError(f"expected uint8 {self.input_shape}, got "
                             f"{rgb640.dtype} {rgb640.shape}")
        frame = np.ascontiguousarray(rgb640)[None, ...]
        res = self._pipe.infer({self._in_name: frame})
        return {k: np.asarray(v)[0] for k, v in res.items()}

    def close(self) -> None:
        self._stack.close()
