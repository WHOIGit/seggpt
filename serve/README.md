# Running with TorchServe

Create a virtual environment
```
python -m venv .venv
source .venv/bin/activate
```

Install dependencies
```
pip install -r seggpt_requirements.txt
```

Build and deploy the model
```
./create_mar.sh
./serve.sh
```

In another terminal session, send a request
```
python request.py --input_dir {directory of input images} --prompt_dir {directory of prompt images} --target_dir {directory of target images} --output_dir {desired output directory}
```

# Running with dockerized TorchServe

TODO
