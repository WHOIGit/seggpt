import io
import os
import base64
import logging
import zipfile
from collections import defaultdict
import torch
from PIL import Image


from ts.torch_handler.base_handler import BaseHandler


logger = logging.getLogger(__name__)


class SegGPTHandler(BaseHandler):
    """
    Handler for SegGPT
    """

    def initialize(self, context):
        """Load the model and set it to eval mode"""

        self.manifest = context.manifest
        serialized_file = self.manifest["model"]["serializedFile"]

        properties = context.system_properties
        model_dir = properties.get("model_dir")

        model_pt_path = os.path.join(model_dir, serialized_file)

        if not os.path.isfile(model_pt_path):
            raise RuntimeError("Missing the model.pt file")

        extras_zip_path = os.path.join(model_dir, "seggpt_extras.zip")
        if not os.path.exists(
            os.path.join(os.path.join(model_dir, "src"), "models.py")
        ):
            with zipfile.ZipFile(extras_zip_path, "r") as zip_file:
                zip_file.extractall(model_dir)

        from src.models import seggpt_vit_large_patch16_input896x448

        # Determine device
        if torch.cuda.is_available():
            logger.info("Using CUDA")
            self.map_location = "cuda"
            self.device = torch.device("cuda:0")

        # MPS does not support all of PyTorch's functions used in SegGPT yet. The operator aten::upsample_bicubic2d.out
        # is not implemented for MPS
        # elif torch.backends.mps.is_available():
        #     logger.info("Using MPS")
        #     self.map_location = 'mps'
        #     self.device = torch.device('mps')
        #     os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
        else:
            logger.info("Using CPU")
            self.map_location = "cpu"
            self.device = torch.device("cpu")

        # Load model
        self.model = seggpt_vit_large_patch16_input896x448()
        self.model.seg_type = "instance"
        seggpt_chkpt = torch.load(
            os.path.join(model_dir, "seggpt_vit_large.pth"),
            map_location=self.map_location,
            weights_only=True,
        )
        self.model.load_state_dict(seggpt_chkpt["model"], strict=False)

        self.model.to(self.device)
        self.model.eval()

        logger.debug("Model file %s loaded successfully", model_pt_path)

        self.initialized = True

    def preprocess(self, data):
        """Preprocess the data into a form usable by SegGPT"""

        preprocessed_data = defaultdict(list)
        for row in data:
            request = row.get("body")

            input_imgs = request.get("input")
            for input_img in input_imgs:
                preprocessed_data["input"].append(
                    (
                        Image.open(io.BytesIO(base64.b64decode(input_img[0]))).convert(
                            "RGB"
                        ),
                        input_img[1],
                    )
                )

            prompt_imgs = request.get("prompts")
            for prompt_img in prompt_imgs:
                preprocessed_data["prompts"].append(
                    Image.open(io.BytesIO(base64.b64decode(prompt_img[0]))).convert(
                        "RGB"
                    )
                )

            target_imgs = request.get("targets")
            for target_img in target_imgs:
                preprocessed_data["targets"].append(
                    Image.open(io.BytesIO(base64.b64decode(target_img[0]))).convert(
                        "RGB"
                    )
                )

            preprocessed_data["output_dir"] = request.get("output_dir")
            preprocessed_data["patch_images"] = request.get("patch_images")
            preprocessed_data["num_prompts"] = request.get("num_prompts")

        return preprocessed_data

    def inference(self, data):
        """Perform inference on the input data"""
        from src.engine import infer

        with torch.no_grad():
            output_merged_imgs, output_masks = infer(
                self.model,
                self.device,
                data["input"],
                data["prompts"],
                data["targets"],
                data["output_dir"],
                data["patch_images"],
                data["num_prompts"],
                save_images=False,
            )
        return output_merged_imgs, output_masks

    def _encode_image_to_base64(self, image):
        """
        Encode the given PIL Image as a base64 string.
        """
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_data = buffered.getvalue()
        img_str = base64.b64encode(img_data).decode("utf-8")
        return img_str

    def postprocess(self, data):
        """
        Convert the output PIL images into a binary form.
        """
        output_merged_imgs, output_masks = data

        # binarized_merged = [
        #     self._encode_image_to_base64(img) for img in output_merged_imgs
        # ]
        binarized_masks = [self._encode_image_to_base64(img) for img in output_masks]
        
        torch.cuda.empty_cache()

        return [binarized_masks]
