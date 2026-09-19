import argparse
import numpy as np
import openvino as ov
import torch
from basicsr.archs.rrdbnet_arch import RRDBNet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--sizes', nargs='+', type=int, default=[64, 256, 276])
    parser.add_argument('--architecture', choices=('rrdb', 'rrdb6'), default='rrdb')
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    weights = checkpoint.get('params_ema', checkpoint.get('params'))
    model = RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=6 if args.architecture == 'rrdb6' else 23,
        num_grow_ch=32,
        scale=4,
    )
    model.load_state_dict(weights, strict=True)
    model.eval()

    core = ov.Core()
    compiled = core.compile_model(
        args.model,
        'CPU',
        {
            'INFERENCE_NUM_THREADS': '4',
            'NUM_STREAMS': '1',
            'PERFORMANCE_HINT': 'LATENCY'
        },
    )
    output_port = compiled.output(0)

    torch.manual_seed(0)
    for size in args.sizes:
        input_tensor = torch.rand((1, 3, size, size), dtype=torch.float32)
        with torch.no_grad():
            torch_output = model(input_tensor).numpy()
        openvino_output = compiled([input_tensor.numpy()])[output_port]
        difference = np.abs(torch_output - openvino_output)
        print(f'size={size} mean_abs={difference.mean():.8g} '
              f'max_abs={difference.max():.8g}')


if __name__ == '__main__':
    main()
