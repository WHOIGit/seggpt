zip -r seggpt_extras.zip ../src/
mkdir -p model_store
torch-model-archiver --model-name seggpt --serialized-file seggpt_vit_large.pth --model-file ../src/seggpt.py --handler seggpt_handler.py --extra-files seggpt_extras.zip --requirements-file seggpt_requirements.txt -v 1.0 --config-file configs/mar_config.txt
mv seggpt.mar model_store
