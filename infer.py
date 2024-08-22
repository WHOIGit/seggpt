import torch
import argparse

import src.models as models
from src.engine import infer


def get_args_parser():
    """
    Parse arguments for running SegGPT inference.
    """
    parser = argparse.ArgumentParser('SegGPT inference', add_help=False)
    parser.add_argument('--ckpt_path', type=str, help='path to ckpt',
                        default='seggpt_vit_large.pth')
    parser.add_argument('--model', type=str, help='specific model name',
                        default='seggpt_vit_large_patch16_input896x448')
    parser.add_argument('--seg_type', type=str, help='embedding for segmentation types', 
                        choices=['instance', 'semantic'], default='instance')
    parser.add_argument('--device', type=str, help='cpu, cuda, or mps (for silicon macs)',
                        default='cpu')
    parser.add_argument('--output_dir', type=str, help='path to output',
                        default='./')
    parser.add_argument('--input_dir', type=str, required=True, help='directory of input images to be segmented')
    parser.add_argument('--prompt_dir', type=str, required=True, help='directory of prompt images to use as context')
    parser.add_argument('--target_dir', type=str, required=True, help='directory of target images, i.e., binary mask images of the prompt images')
    parser.add_argument('--patch_images', action='store_true', help='divide images into 448x448 patches')
    parser.add_argument('--num_prompts', type=int, default=8, help='The number of prompt/targets to use if patching images')
    return parser.parse_args()


def prepare_model(chkpt_dir, arch='seggpt_vit_large_patch16_input896x448', seg_type='instance'):
    """
    Load the specified configuration type and checkpoint.
    """
    # build model
    model = getattr(models, arch)()
    model.seg_type = seg_type
    # load model
    checkpoint = torch.load(chkpt_dir, map_location='cpu', weights_only=True)
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    model.eval()
    return model


if __name__ == '__main__':
    args = get_args_parser()

    device = torch.device(args.device)
    model = prepare_model(args.ckpt_path, args.model, args.seg_type).to(device)
    print('Model loaded.')

    # Perform inference on a directory of input images, directory of prompt images, and directory of target images
    infer(model, device, args.input_dir, args.prompt_dir, args.target_dir, args.output_dir, args.patch_images, args.num_prompts, True)