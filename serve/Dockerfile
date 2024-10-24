FROM pytorch/torchserve:0.12.0-gpu

COPY model_store/seggpt.mar /home/model-server/model-store/

COPY configs/torchserve_config.txt /home/model-server/

COPY seggpt_requirements.txt /home/model-server

EXPOSE 8080 8081

# Switch to root to install Git
USER root

RUN apt-get update && apt-get install -y git

RUN pip install --no-cache-dir -r /home/model-server/seggpt_requirements.txt

# Switch back to model-server for serving
USER model-server

CMD ["torchserve", "--start", "--ncs", "--model-store", "/home/model-server/model-store", "--models", "seggpt.mar", "--disable-token-auth", "--ts-config", "/home/model-server/torchserve_config.txt"]
