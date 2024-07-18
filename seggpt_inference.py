import os
import torch
import argparse

from seggpt_engine import inference_image, inference_image_dir, inference_video
import seggpt_models


def get_args_parser():
    parser = argparse.ArgumentParser('SegGPT inference', add_help=False)
    parser.add_argument('--ckpt_path', type=str, help='path to ckpt',
                        default='seggpt_vit_large.pth')
    parser.add_argument('--model', type=str, help='specific model name',
                        default='seggpt_vit_large_patch16_input896x448')
    parser.add_argument('--input_image', type=str, help='path to input image to be tested',
                        default=None)
    parser.add_argument('--input_video', type=str, help='path to input video to be tested',
                        default=None)
    parser.add_argument('--num_frames', type=int, help='number of prompt frames in video',
                        default=0)
    parser.add_argument('--prompt_image', type=str, nargs='+', help='path to prompt image',
                        default=None)
    parser.add_argument('--prompt_target', type=str, nargs='+', help='path to prompt target',
                        default=None)
    parser.add_argument('--seg_type', type=str, help='embedding for segmentation types', 
                        choices=['instance', 'semantic'], default='instance')
    parser.add_argument('--device', type=str, help='cpu, cuda, or mps (for silicon macs)',
                        default='cpu')
    parser.add_argument('--output_dir', type=str, help='path to output',
                        default='./')
    parser.add_argument('--input_dir', type=str, help='directory of input images to be segmented')
    parser.add_argument('--prompt_dir', type=str, help='directory of prompt images to use as context')
    parser.add_argument('--target_dir', type=str, help='directory of target images, i.e., binary mask images of the prompt images')
    parser.add_argument('--patch_images', action='store_true', help='divide images into 448x448 patches')
    return parser.parse_args()


def prepare_model(chkpt_dir, arch='seggpt_vit_large_patch16_input896x448', seg_type='instance'):
    # build model
    model = getattr(seggpt_models, arch)()
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

    # Directory of input images, directory of prompt images, directory of target images
    if args.input_dir is not None:
        assert args.prompt_dir is not None and args.target_dir is not None

        inference_image_dir(model, device, args.input_dir, args.prompt_dir, args.target_dir, args.output_dir, args.patch_images)

    # Input image, prompt image, target image
    if args.input_image is not None:
        assert args.prompt_image is not None and args.prompt_target is not None

        img_name = os.path.basename(args.input_image)
        out_path = os.path.join(args.output_dir, "output_" + '.'.join(img_name.split('.')[:-1]) + '.png')

        inference_image(model, device, args.input_image, args.prompt_image, args.prompt_target, out_path)
    
    # Input image, prompt image, target image
    if args.input_video is not None:
        assert args.prompt_target is not None
        vid_name = os.path.basename(args.input_video)
        out_path = os.path.join(args.output_dir, "output_" + '.'.join(vid_name.split('.')[:-1]) + '.mp4')
        img_path = os.path.join(args.output_dir, "output_" + '.'.join(vid_name.split('.')[:-1]) + '_%d.png')

        inference_video(model, device, args.input_video, args.num_frames, args.prompt_image, args.prompt_target, out_path, img_path)

    print('Finished.')
