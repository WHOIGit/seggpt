import os
import torch
import torch.nn.functional as F
import numpy as np
import random
from tqdm import tqdm

from PIL import Image


imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])


class Cache(list):
    """
    A class for a FIFO cache with a specified maximum size.
    """
    def __init__(self, max_size=0):
        super().__init__()
        self.max_size = max_size

    def append(self, x):
        if self.max_size <= 0:
            return
        super().append(x)
        if len(self) > self.max_size:
            self.pop(0)


@torch.no_grad()
def run_one_image(img, tgt, model, device):
    """
    Run inference with the given input image and target image. 
    """
    x = torch.tensor(img)
    # make it a batch-like
    x = torch.einsum('nhwc->nchw', x)

    tgt = torch.tensor(tgt)
    # make it a batch-like
    tgt = torch.einsum('nhwc->nchw', tgt)

    bool_masked_pos = torch.zeros(model.patch_embed.num_patches)
    bool_masked_pos[model.patch_embed.num_patches//2:] = 1
    bool_masked_pos = bool_masked_pos.unsqueeze(dim=0)
    valid = torch.ones_like(tgt)

    if model.seg_type == 'instance':
        seg_type = torch.ones([valid.shape[0], 1])
    else:
        seg_type = torch.zeros([valid.shape[0], 1])
    
    feat_ensemble = 0 if len(x) > 1 else -1
    _, y, mask = model(x.float().to(device), tgt.float().to(device), bool_masked_pos.to(device), valid.float().to(device), seg_type.to(device), feat_ensemble)
    y = model.unpatchify(y)
    y = torch.einsum('nchw->nhwc', y).detach().cpu()

    output = y[0, y.shape[1]//2:, :, :]
    output = torch.clip((output * imagenet_std + imagenet_mean) * 255, 0, 255)
    return output


def to_patches(image: Image.Image, patch_size: int):
    """
    Convert the given PIL image into patches of the specified size.
    """
    # Get the original dimensions
    img_width, img_height = image.size
    
    # Calculate the new dimensions
    new_width = max((img_width // patch_size), 1) * patch_size
    new_height = max((img_height // patch_size), 1) * patch_size

    # Resize the image
    resized_image = image.resize((new_width, new_height))

    patches = []
    # Calculate the number of patches in both dimensions
    num_patches_x = resized_image.width // patch_size
    num_patches_y = resized_image.height // patch_size

    for i in range(num_patches_x):
        for j in range(num_patches_y):
            # Calculate the bounding box of the current patch
            left = i * patch_size
            upper = j * patch_size
            right = left + patch_size
            lower = upper + patch_size

            # Crop the patch from the image
            patches.append(resized_image.crop((left, upper, right, lower)))
    return patches


def generate_prompt_target_images(prompt_dir: list, target_dir: list, patchify: bool = False, patch_size: int = 448, num_patches: int = 8):
    """
    Generate prompt and target PIL Images from the given prompt and target directories. Split them 
    into patches if patchify is True.
    """
    prompt_names = os.listdir(prompt_dir)
    prompt_paths = [os.path.join(prompt_dir, prompt_img_name) for prompt_img_name in prompt_names]

    target_paths = [os.path.join(target_dir, prompt_img_name) for prompt_img_name in prompt_names]
    target_paths = [os.path.splitext(target)[0] + '_target.png' for target in target_paths]

    prompt_images = []
    target_images = []
    if patchify:
        for prompt_path in prompt_paths:
           prompt_image = Image.open(prompt_path).convert("RGB")
           prompt_images.extend(to_patches(prompt_image, patch_size))

        for target_path in target_paths:
           target_image = Image.open(target_path).convert("RGB")
           target_images.extend(to_patches(target_image, patch_size))

        paired_list = list(zip(prompt_images, target_images))
        samples = random.sample(paired_list, num_patches)
        prompt_images, target_images = zip(*samples)
    else:
        prompt_images = [Image.open(img).convert("RGB") for img in prompt_paths]
        target_images = [Image.open(img).convert("RGB") for img in target_paths]

    return prompt_images, target_images


def inference_single_image(model, device, input_image: Image, prompt_dir, target_dir, out_dir, patchify, num_patches, output_image_name: str = 'output', save_image=True):
    """
    Run inference on the given PIL image, using the prompts and targets from the given prompt and target directories, 
    respectively. If patchify is specified, divides the prompt and target images into patches, and randomly selects 
    the specified number of patches to use as model context.
    """
    print('patchify: ', patchify)
    patch_size = 448
    prompt_images, target_images = generate_prompt_target_images(prompt_dir, target_dir, patchify, patch_size, num_patches)

    img_path_no_ext = os.path.splitext(os.path.join(out_dir, output_image_name))[0]
    output_merged_path = img_path_no_ext + '_merged' + '.png'
    output_mask_path =img_path_no_ext + '_mask' + '.png'
    output_merged_img, output_mask_img = inference_and_save_image(model, device, input_image, prompt_images, target_images, output_merged_path, output_mask_path, patchify, save_image)
    return output_merged_img, output_mask_img



def inference_image_dir(model, device, input_dir, prompt_dir, target_dir, out_dir, patchify, num_patches, save_images=True):
    """
    Run inference on the images in the given input directory, using prompts and targets from the given prompt and
    target directories, respectively. If patchify is specified, divides the prompt and target images into patches, 
    and randomly selects the specified number of patches to use as model context.
    """
    print('patchify: ', patchify)
    patch_size = 448
    prompt_images, target_images = generate_prompt_target_images(prompt_dir, target_dir, patchify, patch_size, num_patches)

    output_merged_imgs = []
    output_masks = []
    for img_name in tqdm(sorted(os.listdir(input_dir))):
        img_path = os.path.join(input_dir, img_name)
        img_path_no_ext = os.path.splitext(os.path.join(out_dir, img_name))[0]
        output_merged_path = img_path_no_ext + '_merged' + '.png'
        output_mask_path =img_path_no_ext + '_mask' + '.png'
        output_merged_img, output_mask_img = inference_and_save_image(model, device, img_path, prompt_images, target_images, output_merged_path, output_mask_path, patchify, save_images)
        output_merged_imgs.append(output_merged_img)
        output_masks.append(output_mask_img)

    return output_merged_imgs, output_masks


def inference_and_save_image(model, device, input_image, prompt_images, target_images, out_merged_path, out_mask_path, patchify, save_image=True):
    """
    Run inference on the image at the given img_path, using the given prompt and target images. If patchify is 
    specified, divides the images into separate patches to reduce data loss from resizing.
    """
    patch_size = 448

    # If input_image is a path, convert it to a PIL Image
    if (isinstance(input_image, str)):
        input_image = Image.open(input_image).convert("RGB")
    else:
        if not isinstance(input_image, Image.Image):
            raise TypeError(f'Input image must be either a string path or a PIL Image. Received {type(input_image)}.')

    if patchify:
        original_width, original_height = input_image.size
        patches = to_patches(input_image, patch_size)
        annotated_patches = []
        num_patches_x = input_image.width // patch_size
        num_patches_y = input_image.height // patch_size
        for patch in tqdm(patches, position=1, leave=False):
            annotated_patches.append(inference_image(model, device, patch, prompt_images, target_images))

        reconstructed_image = Image.new('RGB', (num_patches_x * patch_size, num_patches_y * patch_size))
        patch_id = 0
        for i in range(num_patches_x):
            for j in range(num_patches_y):
                # Calculate the position where the patch should be placed
                left = i * patch_size
                upper = j * patch_size

                # Paste the patch into the reconstructed image
                annotated_patch = Image.fromarray(annotated_patches[patch_id].astype(np.uint8))
                reconstructed_image.paste(annotated_patch, box=(left, upper, left + annotated_patch.width, upper + annotated_patch.height))
                patch_id += 1
        # Resize reconstructed patched image to original size
        output_mask_img = reconstructed_image.resize((original_width, original_height))
        output_merged = Image.fromarray((np.array(input_image) * (0.6 * np.array(output_mask_img) / 255 + 0.4)).astype(np.uint8))
    else:
        output = inference_image(model, device, input_image, prompt_images, target_images)

        # Create PIL Images for merged output (original image + mask) and mask
        output_mask_img = Image.fromarray((output).astype(np.uint8))
        output_merged = Image.fromarray((np.array(input_image) * (0.6 * output / 255 + 0.4)).astype(np.uint8))

        output_mask_img = Image.fromarray((output).astype(np.uint8))

    if save_image:
        output_mask_img.save(out_mask_path)
        output_merged.save(out_merged_path)

    return output_merged, output_mask_img


def inference_image(model, device, input_image, prompt_images, target_images):
    """
    Run inference on the given input image, using the given prompt images and target images.
    """
    res, hres = 448, 448

    # Open the input image and resize it
    size = input_image.size
    input_image = np.array(input_image.resize((res, hres))) / 255.

    image_batch, target_batch = [], []
    # For each prompt-target pair, add them to the list of 
    for prompt_image, target_image in zip(prompt_images, target_images):
        prompt_image = prompt_image.resize((res, hres))
        prompt_image = np.array(prompt_image) / 255.

        target_image = target_image.resize((res, hres), Image.NEAREST)
        target_image = np.array(target_image) / 255.

        target_to_fill = target_image  # target_to_fill is not available
        # stack target image over the target that will be filled
        stacked_targets = np.concatenate((target_image, target_to_fill), axis=0)
        # stack prompt image over the input image
        stacked_images = np.concatenate((prompt_image, input_image), axis=0)

        assert stacked_images.shape == (2*res, hres, 3), f'{stacked_images.shape}'
        # normalize by ImageNet mean and std
        stacked_images = stacked_images - imagenet_mean
        stacked_images = stacked_images / imagenet_std

        assert stacked_targets.shape == (2*res, hres, 3), f'{stacked_targets.shape}'
        # normalize by ImageNet mean and std
        stacked_targets = stacked_targets - imagenet_mean
        stacked_targets = stacked_targets / imagenet_std

        image_batch.append(stacked_images)
        target_batch.append(stacked_targets)

    # convert list of images/targets into a batched tensor of shape (n, h, w, c)
    image_batch_array = np.stack(image_batch, axis=0)
    target_batch_array = np.stack(target_batch, axis=0)

    """### Run SegGPT on the image"""
    # make random mask reproducible (comment out to make it change)
    torch.manual_seed(2)
    output = run_one_image(image_batch_array, target_batch_array, model, device)
    output = F.interpolate(
        output[None, ...].permute(0, 3, 1, 2), 
        size=[size[1], size[0]], 
        mode='nearest',
    ).permute(0, 2, 3, 1)[0].numpy()

    return output