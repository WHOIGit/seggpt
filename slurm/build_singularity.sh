cd ..
docker build -t seggpt:latest .
docker save -o seggpt.tar seggpt:latest
singularity build seggpt.sif docker-archive://seggpt.tar
