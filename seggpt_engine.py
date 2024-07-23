import os
import torch
import torch.nn.functional as F
import numpy as np
import random
import cv2
from tqdm import tqdm

from PIL import Image


imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])


class Cache(list):
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


def to_patches(image, patch_size):
    """
    Convert the given PIL image into patches of the specified size.
    """
    # Get the original dimensions
    img_width, img_height = image.size
    
    # Calculate the new dimensions
    new_width = (img_width // patch_size) * patch_size
    new_height = (img_height // patch_size) * patch_size

    # Resize the image
    resized_image = image.resize((new_width, new_height))

    patches = []
    # Calculate the number of patches in both dimensions
    num_patches_x = resized_image.width // patch_size
    num_patches_y = resized_image.height // patch_size

    patch_id = 0
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


def inference_image_dir(model, device, input_dir, prompt_dir, target_dir, out_dir, patchify):
    print('patchify: ', patchify)
    prompt_names = os.listdir(prompt_dir)
    prompt_paths = [os.path.join(prompt_dir, prompt_img_name) for prompt_img_name in prompt_names]

    target_paths = [os.path.join(target_dir, prompt_img_name) for prompt_img_name in prompt_names]
    target_paths = [os.path.splitext(target)[0] + '_target.png' for target in target_paths]

    # If using patches, convert all prompt and target images to patches
    prompt_images = []
    target_images = []
    if patchify:
        patch_size = 448
        for prompt_path in prompt_paths:
           prompt_image = Image.open(prompt_path).convert("RGB")
           prompt_images.extend(to_patches(prompt_image, patch_size))
        
        for target_path in target_paths:
           target_image = Image.open(target_path).convert("RGB")
           target_images.extend(to_patches(target_image, patch_size))

        paired_list = list(zip(prompt_images, target_images))
        samples = random.sample(paired_list, 8)
        prompt_images, target_images = zip(*samples)
        for np, p in enumerate(prompt_images):
            p.save(f'prompt_{np}.png')
        for nt, t in enumerate(target_images):
            t.save(f'target_{nt}.png')
    else:
        prompt_images = [Image.open(img).convert("RGB") for img in prompt_paths]
        target_images = [Image.open(img).convert("RGB") for img in target_paths]

    for img_name in tqdm(sorted(os.listdir(input_dir))):
        img_path = os.path.join(input_dir, img_name)
        output_path = os.path.splitext(os.path.join(out_dir, img_name))[0] + '_annotated' + '.png'
        inference_and_save_image(model, device, img_path, prompt_images, target_images, output_path, patchify)


def inference_and_save_image(model, device, img_path, prompt_images, target_images, out_path, patchify):
    patch_size = 448
    input_image = Image.open(img_path).convert("RGB")
    if patchify:
        patches = to_patches(input_image, patch_size)
        annotated_patches = []
        num_patches_x = input_image.width // patch_size
        num_patches_y = input_image.height // patch_size
        pn = 0
        for patch in tqdm(patches):
            patch.save(f'patch_{pn}.png')
            annotated_patches.append(inference_image(model, device, patch, prompt_images, target_images))
            Image.fromarray(annotated_patches[-1].astype(np.uint8)).save(f'test{len(annotated_patches)}.png')
            pn += 1

        reconstructed_image = Image.new('RGB', (num_patches_x * patch_size, num_patches_y * patch_size))
        patch_id = 0
        for i in range(num_patches_x):
            for j in range(num_patches_y):
                # Calculate the position where the patch should be placed
                left = i * patch_size
                upper = j * patch_size

                # Paste the patch into the reconstructed image
                merged_patch = Image.fromarray((np.array(patches[patch_id]) * (0.6 * annotated_patches[patch_id] / 255 + 0.4)).astype(np.uint8))
                reconstructed_image.paste(merged_patch, (left, upper))
                patch_id += 1
        reconstructed_image.save(out_path)
    else:
        output = inference_image(model, device, input_image, prompt_images, target_images)
        output = Image.fromarray((np.array(input_image) * (0.6 * output / 255 + 0.4)).astype(np.uint8))
        output.save(out_path)


def inference_image(model, device, input_image, prompt_images, target_images):
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


def inference_video(model, device, vid_path, num_frames, img2_paths, tgt2_paths, out_path, img_path, anno_color=''):
    # currently only checks for mp4 file
    using_video = vid_path[-3:] == 'mp4'
    
    res, hres = 448, 448

    if using_video:
        cap = cv2.VideoCapture(vid_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height), True)
    else:
        frames = [os.path.join(vid_path, file) for file in sorted(os.listdir(vid_path))]
        ff = 0
    
    if img2_paths is None:
        if using_video:
            _, frame = cap.read()
        else:
            frame = cv2.imread(frames[ff])
            ff += 1

        img2 = [np.array(Image.fromarray(frame[:, :, ::-1]).convert('RGB').resize((res, hres))) / 255.]
    else:
        img2 = [np.array(Image.open(img2_path).convert("RGB").resize((res, hres))) / 255. for img2_path in img2_paths]

    tgt2 = [np.array(Image.open(tgt2_path).convert("RGB").resize((res, hres), Image.NEAREST)) / 255. for tgt2_path in tgt2_paths]

    frames_cache, target_cache = Cache(num_frames), Cache(num_frames)

    i = 0
    while True:
        if using_video:
            ret, frame = cap.read()
        else:
            ret = ff < len(frames)
            if ret:
                frame_name = frames[ff]
                frame = cv2.imread(frame_name)
                ff += 1

        if not ret:
            break

        image_batch, target_batch = [], []
        image = Image.fromarray(frame[:, :, ::-1]).convert('RGB')
        input_image = np.array(image)
        size = image.size
        image = np.array(image.resize((res, hres))) / 255.

        for prompt, target in zip(img2 + frames_cache, tgt2 + target_cache):
            tgt = target  # tgt is not available
            tgt = np.concatenate((target, tgt), axis=0)
            img = np.concatenate((prompt, image), axis=0)

            assert img.shape == (2*res, res, 3), f'{img.shape}'
            # normalize by ImageNet mean and std
            img = img - imagenet_mean
            img = img / imagenet_std

            assert tgt.shape == (2*res, res, 3), f'{img.shape}'
            # normalize by ImageNet mean and std
            tgt = tgt - imagenet_mean
            tgt = tgt / imagenet_std

            image_batch.append(img)
            target_batch.append(tgt)

        img = np.stack(image_batch, axis=0)
        tgt = np.stack(target_batch, axis=0)

        torch.manual_seed(2)
        output = run_one_image(img, tgt, model, device)

        frames_cache.append(image)
        target_cache.append(
            output.mean(-1) \
                .gt(128).float() \
                .unsqueeze(-1).expand(-1, -1, 3) \
                .numpy()
        )

        output = F.interpolate(
            output[None, ...].permute(0, 3, 1, 2), 
            size=[size[1], size[0]], 
            mode='nearest',
        ).permute(0, 2, 3, 1)[0].numpy()
        if using_video:
            img_name = img_path % i
        else:
            img_name = os.path.join(img_path, os.path.splitext(os.path.basename(frame_name))[0] + '.png')
        img_to_write = np.tile((output.mean(-1) > 128).astype(np.float32)[:, :, np.newaxis], (1, 1, 3)).astype(np.uint8)
        img_to_write[:] = anno_color.split()
        cv2.imwrite(img_name, img_to_write)
        if using_video:
            output = input_image * (0.6 * output / 255 + 0.4)
            video_writer.write(np.ascontiguousarray(output.astype(np.uint8)[:, :, ::-1]))
        i += 1

    if using_video:
        video_writer.release()
