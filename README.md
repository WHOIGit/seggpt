<div align="center">
<h1>SegGPT: Segmenting Everything In Context </h1>

</div>

<br>

   This repository contains inference functions for SegGPT, a generalist model for segmenting everything in context. With only one single model, SegGPT can perform arbitrary segmentation tasks in images or videos via in-context inference, such as object instance, stuff, part, contour, and text. 
   SegGPT is evaluated on a broad range of tasks, including few-shot semantic segmentation, video object segmentation, semantic segmentation, and panoptic segmentation. 
   Our results show strong capabilities in segmenting in-domain and out-of-domain targets, either qualitatively or quantitatively. 

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
python seggpt_inference.py --input_dir {directory of input images} --prompt_dir {directory of prompt images} --target_dir {directory of target images} --output_dir {desired output directory}
```


## **TorchServe Inference**

TODO


## Citation

```
@article{SegGPT,
  title={SegGPT: Segmenting Everything In Context},
  author={Wang, Xinlong and Zhang, Xiaosong and Cao, Yue and Wang, Wen and Shen, Chunhua and Huang, Tiejun},
  journal={arXiv preprint arXiv:2304.03284},
  year={2023}
}
```
