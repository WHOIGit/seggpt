"""
Functions for sending a request to the SegGPT TorchServe implementation and processing the results.
"""

import io
import os
import base64
import argparse
from typing import Union

import requests
from PIL import Image


def prepare_images(image_dir: str):
    """
    Loads a directory of images into binary format. Returns the list of binary images, where each 
    entry in the list is a tuple of the form (binarized image, image name).
    """
    image_names = os.listdir(image_dir)
    images = [
        [
            base64.b64encode(open(os.path.join(image_dir, image), "rb").read()).decode(
                "utf-8"
            ),
            image,
        ]
        for image in image_names
    ]
    return images


def request(
    input_dir: str,
    prompt_dir: str,
    target_dir: str,
    output_dir: str,
    patch_images: bool,
    num_prompts: Union[str, int],
):
    """ 
    Sends a request to SegGPT and saves the results.
    """
    input_imgs = prepare_images(input_dir)
    prompt_imgs = prepare_images(prompt_dir)
    target_imgs = prepare_images(target_dir)

    if num_prompts == "all":
        num_prompts_for_request = len(prompt_imgs)
    else:
        num_prompts_for_request = num_prompts

    data = {
        "input": input_imgs,
        "prompts": prompt_imgs,
        "targets": target_imgs,
        "output_dir": output_dir,
        "patch_images": patch_images,
        "num_prompts": num_prompts_for_request,
    }
    response = requests.post("http://localhost:8080/predictions/seggpt", json=data)
    print(response.json())

    for mask_index, mask in enumerate(response.json()):
        mask_data = base64.b64decode(mask)
        mask_image = Image.open(io.BytesIO(mask_data))
        original_name, ext = os.path.splitext(input_imgs[mask_index][1])
        out_path = os.path.join(output_dir, f'{original_name}_mask{ext}')
        mask_image.save(out_path)


def get_args_parser():
    """
    Parse arguments for running SegGPT inference through TorchServe.
    """
    parser = argparse.ArgumentParser("TorchServe SegGPT inference", add_help=False)

    # I/O directories
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="directory of input images to be segmented",
    )
    parser.add_argument(
        "--prompt_dir",
        type=str,
        required=True,
        help="directory of prompt images to use as context",
    )
    parser.add_argument(
        "--target_dir",
        type=str,
        required=True,
        help="directory of target images, i.e., binary mask images of the prompt images",
    )
    parser.add_argument("--output_dir", type=str, required=True, help="path to output")

    # Inference options
    parser.add_argument(
        "--patch_images", action="store_true", help="divide images into 448x448 patches"
    )
    parser.add_argument(
        "--num_prompts",
        default="all",
        help="The number of prompt/targets to use if patching images",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args_parser()
    request(
        args.input_dir,
        args.prompt_dir,
        args.target_dir,
        args.output_dir,
        args.patch_images,
        args.num_prompts,
    )
