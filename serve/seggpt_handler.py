import io
import os
import torch
import base64
import logging
import zipfile
from PIL import Image

from ts.torch_handler.base_handler import BaseHandler


logger = logging.getLogger(__name__)

class SegGPTHandler(BaseHandler):
    """
    Handler for SegGPT
    """

    def initialize(self, context):
        """ Load the model and set it to eval mode """

        self.manifest = context.manifest
        serialized_file = self.manifest["model"]["serializedFile"]

        properties = context.system_properties
        model_dir = properties.get("model_dir")

        model_pt_path = os.path.join(model_dir, serialized_file)

        if not os.path.isfile(model_pt_path):
            raise RuntimeError("Missing the model.pt file")

        extras_zip_path = os.path.join(model_dir, "seggpt_extras.zip")
        if not os.path.exists(os.path.join(model_dir, "seggpt_models.py")):
            with zipfile.ZipFile(extras_zip_path, "r") as zip_file:
                zip_file.extractall(model_dir)

        import seggpt_models

        # Load model
        arch = 'seggpt_vit_large_patch16_input896x448'
        self.model = getattr(seggpt_models, arch)()
        self.model.seg_type = 'instance'
        seggpt_chkpt = torch.load(
            os.path.join(model_dir, "seggpt_vit_large.pth"), map_location='cpu', weights_only=True
        )
        self.model.load_state_dict(seggpt_chkpt['model'], strict=False)

        self.model.to(self.device)
        self.model.eval()

        logger.debug("Model file %s loaded successfully", model_pt_path)

        self.initialized = True

    def preprocess(self, data):
        """ Preprocess the data into a form usable by SegGPT """
        images = []

        for row in data:
            image = row.get("data") or row.get("body")
            if isinstance(image, str):
                image = base64.b64decode(image)

            # If the image is sent as bytesarray
            if isinstance(image, (bytearray, bytes)):
                image = Image.open(io.BytesIO(image))
                image = self.image_processing(image)
            else:
                # if the image is a list
                image = torch.FloatTensor(image)

            images.append(image)

        return torch.stack(images).to(self.device)


    def inference(self, data):
        """ Perform inference on the input data """
        with torch.no_grad():
            output = self.model(data)
        return output
