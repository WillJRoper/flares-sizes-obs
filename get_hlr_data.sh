#!/bin/bash -l
#SBATCH --ntasks 1 # The number of cores you need...
#SBATCH --array=1-480
#SBATCH -p cosma6 #or some other partition, e.g. cosma, cosma6, etc.
#SBATCH -A dp004
#SBATCH --cpus-per-task=1
#SBATCH -J MEGA-FLARES #Give it something meaningful.
#SBATCH -o logs/output_hlr_job.%A_%a.out
#SBATCH -e logs/error_hlr_job.%A_%a.err
#SBATCH -t 72:00:00

# Run the job from the following directory - change this to point to your own personal space on /lustre
cd /cosma7/data/dp004/dc-rope1/FLARES/flares-sizes-obs

module purge
#load the modules used to build your program.
module load pythonconda3/4.5.4

source activate flares-env

i=$(($SLURM_ARRAY_TASK_ID - 1))

# Run the program
#./UV_size_lumin_relation_distributed.py $i sim Total 0
#./UV_size_lumin_relation_distributed.py $i sim Total 1
#./UV_size_lumin_relation_distributed.py $i sim Total 2
#./UV_size_lumin_relation_distributed.py $i sim Intrinsic 0
#./UV_size_lumin_relation_distributed.py $i sim Intrinsic 1
#./UV_size_lumin_relation_distributed.py $i sim Intrinsic 2
#./UV_size_lumin_relation_distributed.py $i face-on Total
#./UV_size_lumin_relation_distributed.py $i face-on Intrinsic

source deactivate

echo "Job done, info follows..."
sacct -j $SLURM_JOBID --format=JobID,JobName,Partition,MaxRSS,Elapsed,ExitCode
exit

