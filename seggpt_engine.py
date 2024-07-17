import os
import torch
import torch.nn.functional as F
import numpy as np
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


def inference_image_dir(model, device, input_dir, prompt_dir, target_dir, out_dir):
    prompt_names = os.listdir(prompt_dir)
    prompt_paths = [os.path.join(prompt_dir, prompt_img_name) for prompt_img_name in prompt_names]

    target_paths = [os.path.join(target_dir, prompt_img_name) for prompt_img_name in prompt_names]
    target_paths = [os.path.splitext(target)[0] + '_target.png' for target in target_paths]

    for img_name in tqdm(sorted(os.listdir(input_dir))):
        img_path = os.path.join(input_dir, img_name)
        output_path = os.path.splitext(os.path.join(out_dir, img_name))[0] + '_annotated' + '.png'
        inference_image(model, device, img_path, prompt_paths, target_paths, output_path)


def inference_image(model, device, img_path, prompt_paths, target_paths, out_path):
    res, hres = 448, 448

    image = Image.open(img_path).convert("RGB")
    input_image = np.array(image)
    size = image.size
    image = np.array(image.resize((res, hres))) / 255.

    image_batch, target_batch = [], []
    for img2_path, tgt2_path in zip(prompt_paths, target_paths):
        img2 = Image.open(img2_path).convert("RGB")
        img2 = img2.resize((res, hres))
        img2 = np.array(img2) / 255.

        tgt2 = Image.open(tgt2_path).convert("RGB")
        tgt2 = tgt2.resize((res, hres), Image.NEAREST)
        tgt2 = np.array(tgt2) / 255.

        tgt = tgt2  # tgt is not available
        tgt = np.concatenate((tgt2, tgt), axis=0)
        img = np.concatenate((img2, image), axis=0)
    
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

    """### Run SegGPT on the image"""
    # make random mask reproducible (comment out to make it change)
    torch.manual_seed(2)
    output = run_one_image(img, tgt, model, device)
    output = F.interpolate(
        output[None, ...].permute(0, 3, 1, 2), 
        size=[size[1], size[0]], 
        mode='nearest',
    ).permute(0, 2, 3, 1)[0].numpy()
    output = Image.fromarray((input_image * (0.6 * output / 255 + 0.4)).astype(np.uint8))
    output.save(out_path)


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
            print(img_name)
        img_to_write = np.tile((output.mean(-1) > 128).astype(np.float32)[:, :, np.newaxis], (1, 1, 3)).astype(np.uint8)
        img_to_write[:] = anno_color.split()
        cv2.imwrite(img_name, img_to_write)
        if using_video:
            output = input_image * (0.6 * output / 255 + 0.4)
            video_writer.write(np.ascontiguousarray(output.astype(np.uint8)[:, :, ::-1]))
        i += 1

    if using_video:
        video_writer.release()
