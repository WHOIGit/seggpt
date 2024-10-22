<div align="center">
<h1>SegGPT: Segmenting Everything In Context </h1>

</div>

<br>

   This repository contains inference functions for SegGPT, a generalist model for segmenting everything in context. With only one single model, SegGPT can perform arbitrary segmentation tasks in images or videos via in-context inference, such as object instance, stuff, part, contour, and text. 

  This repository is forked from [BAAI's Painter repository](https://github.com/baaivision/Painter).

[[Paper]](https://arxiv.org/abs/2304.03284)
[[Demo]](https://huggingface.co/spaces/BAAI/SegGPT)


## **Usage**

SegGPT takes in three types of data: input images, prompt images, and target images. The input images are the images to be annotated. The prompt images are images that have already been annotated, and the target images are the annotations of those images, in the form of masks. 

## **Local Inference**

Create a virtual environment
```
python -m venv .venv
source .venv/bin/activate
```

Install dependencies
```
pip install -r requirements.txt
pip install git+https://github.com/facebookresearch/detectron2.git --no-build-isolation
```

Download the model checkpoint
```
wget https://huggingface.co/BAAI/SegGPT/resolve/main/seggpt_vit_large.pth
```

Run inference on a directory of input images, using a directory of prompt images and a corresponding directory of target images
```
python infer.py --input_dir {directory of input images} --prompt_dir {directory of prompt images} --target_dir {directory of target images} --output_dir {desired output directory}
```


## **TorchServe Inference**

TODO

## **Slurm Inference**

To run with Slurm, follow the [Slurm README](slurm/README.md)
