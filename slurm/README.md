# Running on a Slurm-enabled computer

To run SegGPT on a Slurm-enabled computer, you'll need to build a Singularity image. Run:

```
./build_singularity.sh
```

This will build the SegGPT Docker image, and convert it to a Singularity image. This is tested with Singularity v3.7.4.

Then, create a batch script for submitting the job to Slurm. An example batch script is:

```
#!/bin/bash
#SBATCH --partition=gpu             # Queue selection
#SBATCH --job-name=seggpt_job       # Job name
#SBATCH --mail-type=END             # Mail events (BEGIN, END, FAIL, ALL)
#SBATCH --mail-user={your email}    # Where to send mail
#SBATCH --ntasks=1                  # Run on a single CPU
#SBATCH --mem=5gb                   # Job memory request
#SBATCH --time=00:15:00             # Time limit hrs:min:sec
#SBATCH --output={output file}      # Standard error
#SBATCH --output={output file}      # Standard output
#SBATCH --gres=gpu:1                # Allocate GPUs

date

module load singularity/3.7

echo "Running SegGPT on GPU partition"

singularity exec --nv -B {data directory}:/main/data --pwd /main {absolute path to sif} python3 /main/infer.py --input_dir /main/data/{relative path to input dir} --prompt_dir /main/data/{relative path to prompt dir} --target_dir /main/data/{relative path to target dir} --output_dir /main/data/{relative path to output dir} --patch_images --num_prompts {number of prompts to use out of prompts submitted} --device cuda

date
``` 
