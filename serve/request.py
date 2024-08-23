import os
import sys
import base64
import argparse
import requests
import json
from typing import Union


def prepare_images(image_dir: str):
    """
    Load a directory of images into binary format. Returns the list of binary images, where each entry in the 
    list is a tuple of the form (binarized image, image name).
    """
    image_names = os.listdir(image_dir)
    images = [[base64.b64encode(open(os.path.join(image_dir, image), 'rb').read()).decode('utf-8'), image] for image in image_names]
    return images


def request(input_dir: str, prompt_dir: str, target_dir: str, output_dir: str, patch_images: bool, num_prompts: Union[str, int]):
    """
    
    """
    input_imgs = prepare_images(input_dir)
    prompt_imgs = prepare_images(prompt_dir)
    target_imgs = prepare_images(target_dir)

    if num_prompts == 'all':
        num_prompts_for_request = len(prompt_imgs)
    
    data = {
        'input': input_imgs,
        'prompts': prompt_imgs,
        'targets': target_imgs,
        'output_dir': output_dir,
        'patch_images': patch_images,
        'num_prompts': num_prompts_for_request
    }
    json_data = json.dumps(data)
    size_in_bytes = sys.getsizeof(json_data.encode('utf-8'))
    size_in_mb = size_in_bytes / (1024 * 1024)
    print(f"The size of the JSON request is: {size_in_mb:.6f} MB")
    response = requests.post('http://localhost:8080/predictions/seggpt', json=data)
    print(response.status_code)
    print(response.text)


def get_args_parser():
    """
    Parse arguments for running SegGPT inference through TorchServe.
    """
    parser = argparse.ArgumentParser('TorchServe SegGPT inference', add_help=False)
    
    # I/O directories
    parser.add_argument('--input_dir', type=str, required=True, help='directory of input images to be segmented')
    parser.add_argument('--prompt_dir', type=str, required=True, help='directory of prompt images to use as context')
    parser.add_argument('--target_dir', type=str, required=True, help='directory of target images, i.e., binary mask images of the prompt images')
    parser.add_argument('--output_dir', type=str, required=True, help='path to output')
    
    # Inference options
    parser.add_argument('--patch_images', action='store_true', help='divide images into 448x448 patches')
    parser.add_argument('--num_prompts', default='all', help='The number of prompt/targets to use if patching images')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args_parser()
    request(args.input_dir, args.prompt_dir, args.target_dir, args.output_dir, args.patch_images, args.num_prompts)
