import argparse
import openvino as ov
import torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from pathlib import Path

from realesrgan.archs.srvgg_arch import SRVGGNetCompact


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--architecture', choices=('rrdb', 'rrdb6', 'srvgg'), default='rrdb')
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    weights = checkpoint.get('params_ema', checkpoint.get('params'))
    if weights is None:
        raise ValueError('Checkpoint does not contain params_ema or params')

    if args.architecture in ('rrdb', 'rrdb6'):
        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=6 if args.architecture == 'rrdb6' else 23,
            num_grow_ch=32,
            scale=4,
        )
    else:
        model = SRVGGNetCompact(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_conv=32,
            upscale=4,
            act_type='prelu',
        )
    model.load_state_dict(weights, strict=True)
    model.eval()

    example = torch.zeros((1, 3, 64, 64), dtype=torch.float32)
    ov_model = ov.convert_model(
        model,
        example_input=example,
        input=ov.PartialShape([1, 3, -1, -1]),
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    ov.save_model(ov_model, output, compress_to_fp16=False)
    print(f'Saved OpenVINO IR to {output}')


if __name__ == '__main__':
    main()
