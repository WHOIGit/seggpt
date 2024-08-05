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

        from seggpt_models import seggpt_vit_large_patch16_input896x448

        # Determine device
        if torch.cuda.is_available():
            logger.info("Using CUDA")
            self.map_location = 'cuda'
            self.device = torch.device('cuda:0')
        #elif torch.backends.mps.is_available():
        #    logger.info("Using MPS")
        #    self.map_location = 'mps'
        #    self.device = torch.device('mps')
        #    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
        else:
            logger.info("Using CPU")
            self.map_location = 'cpu'
            self.device = torch.device('cpu')

        # Load model
        self.model = seggpt_vit_large_patch16_input896x448()
        self.model.seg_type = 'instance'
        seggpt_chkpt = torch.load(
            os.path.join(model_dir, "seggpt_vit_large.pth"), map_location=self.map_location, weights_only=True
        )
        self.model.load_state_dict(seggpt_chkpt['model'], strict=False)

        self.model.to(self.device)
        self.model.eval()

        logger.debug("Model file %s loaded successfully", model_pt_path)

        self.initialized = True

    def preprocess(self, data):
        """ Preprocess the data into a form usable by SegGPT """
        images = []
        logger.info('processing img')

        for row in data:
            logger.info('processing row')
            image = row.get("data") or row.get("body")
            logger.info('got image')
            if isinstance(image, str):
                logger.info('image is a str')
                image = base64.b64decode(image)

            # If the image is sent as bytesarray
            if isinstance(image, (bytearray, bytes)):
                logger.info('image is a bytes array')
                image = Image.open(io.BytesIO(image))

            else:
                # if the image is a list
                logger.info('image is neither')
                image = torch.FloatTensor(image)

            images.append(image)

        return images[0]


    def inference(self, data):
        """ Perform inference on the input data """
        from seggpt_engine import inference_single_image                                                                    

        output = inference_single_image(self.model, self.device, data, 'prompts', 'targets', 'outputs', True, 8, save_image=False)
        return output

    def postprocess(self, data):
        """
        Convert the output PIL images into a binary form. 
        """
        binary = io.BytesIO()
        data.save(binary, format='PNG')
        binary_image = binary.getvalue()
        binary.close()
        return [binary_image]
