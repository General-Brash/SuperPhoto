import numpy as np
import openvino as ov
import torch

from realesrgan import RealESRGANer


class _CompiledModelAdapter:
    def __init__(self, model_path, threads):
        core = ov.Core()
        model = core.read_model(model_path)
        config = {
            'INFERENCE_NUM_THREADS': str(threads),
            'NUM_STREAMS': '1',
            'PERFORMANCE_HINT': 'LATENCY',
        }
        self.compiled_model = core.compile_model(model, 'CPU', config)
        self.output = self.compiled_model.output(0)

    def __call__(self, tensor):
        array = np.ascontiguousarray(tensor.detach().cpu().numpy(), dtype=np.float32)
        result = self.compiled_model([array])[self.output]
        return torch.from_numpy(np.array(result, copy=True))


class OpenVINORealESRGANer(RealESRGANer):
    def __init__(self, scale, model_path, tile=0, tile_pad=10, pre_pad=0, threads=4):
        self.scale = scale
        self.tile_size = tile
        self.tile_pad = tile_pad
        self.pre_pad = pre_pad
        self.mod_scale = None
        self.half = False
        self.device = torch.device('cpu')
        self.model = _CompiledModelAdapter(model_path, threads)
