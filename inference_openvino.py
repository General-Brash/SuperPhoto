import argparse
import glob
import os

import cv2

from app.inference import OpenVINORealESRGANer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', default='inputs')
    parser.add_argument('-o', '--output', default='results')
    parser.add_argument('--model-path', default='weights/RealESRGAN_x4plus.xml')
    parser.add_argument('-s', '--outscale', type=float, default=4)
    parser.add_argument('-t', '--tile', type=int, default=256)
    parser.add_argument('--tile-pad', type=int, default=10)
    parser.add_argument('--pre-pad', type=int, default=0)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--suffix', default='out')
    parser.add_argument('--ext', choices=('auto', 'jpg', 'png'), default='auto')
    parser.add_argument('--alpha-upsampler', choices=('realesrgan', 'bicubic'), default='realesrgan')
    args = parser.parse_args()

    upsampler = OpenVINORealESRGANer(
        scale=4,
        model_path=args.model_path,
        tile=args.tile,
        tile_pad=args.tile_pad,
        pre_pad=args.pre_pad,
        threads=args.threads,
    )

    paths = [args.input] if os.path.isfile(args.input) else sorted(glob.glob(os.path.join(args.input, '*')))
    if not paths:
        raise FileNotFoundError(f'No input images found at {args.input}')

    os.makedirs(args.output, exist_ok=True)
    for index, path in enumerate(paths):
        image_name, extension = os.path.splitext(os.path.basename(path))
        image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if image is None:
            print(f'Skipping unreadable image: {path}')
            continue

        print(f'Testing {index} {image_name}')
        output, image_mode = upsampler.enhance(
            image,
            outscale=args.outscale,
            alpha_upsampler=args.alpha_upsampler,
        )

        output_extension = extension[1:] if args.ext == 'auto' else args.ext
        if image_mode == 'RGBA':
            output_extension = 'png'
        output_name = image_name if not args.suffix else f'{image_name}_{args.suffix}'
        output_path = os.path.join(args.output, f'{output_name}.{output_extension}')
        if not cv2.imwrite(output_path, output):
            raise RuntimeError(f'Failed to write output image: {output_path}')
        print(f'Saved {output_path}')


if __name__ == '__main__':
    main()
