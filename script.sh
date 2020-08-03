#!/usr/bin/env bash

#SBATCH --job-name=5_1percent       # The name of your job, you'll se it in squeue.
#SBATCH --mail-type=ALL              # Mail events (NONE, BEGIN, END, FAIL, ALL). Sends you an email when the job begins, ends, or fails; you can combine options.
#SBATCH --mail-user=samadio@sissa.it    # Where to send the mail
#
# ---- CPU resources configuration  ----  |  Clarifications at https://slurm.schedmd.com/mc_support.html
#
#SBATCH --ntasks=1                   # Number of MPI ranks (1 for MPI serial job)
#SBATCH --cpus-per-task=4           # Number of threads per MPI rank (MAX: 2x32 cores on _partition_2, 2x20 cores on _partition_1)
#SBATCH --nodes=1                    # Number of nodes
#SBATCH --ntasks-per-node=1          # How many tasks on each node
#SBATCH --ntasks-per-socket=1        # How many tasks on each socket
#SBATCH --ntasks-per-core=1          # How many tasks on each core (set to 1 to be sure that different tasks run on different cores on multi-threaded systems)
#[optional] #SBATCH --distribution=cyclic:cyclic # Distribute tasks cyclically on nodes and sockets. For other options, read the docs.
#
# ---- Other resources configuration (e.g. GPU) ----
#
#SBATCH --gpus=1                     # Total number of GPUs for the job (MAX: 2 x number of nodes, only available on gpu1 and gpu2)
#[optional] #SBATCH --gpus-per-node=1            # Number of GPUs per node (MAX: 2, only available on gpu1 and gpu2)
#[optional] #SBATCH --gpus-per-task=1            # Number of GPUs per MPI rank (MAX: 2, only available on gpu1 and gpu2); to be used with --ntasks
#
# ---- Memory configuration ----
#
#SBATCH --mem=5000mb                 # Memory per node (MAX: 63500 on the new ones, 40000 on the old ones); incompatible with --mem-per-cpu.
#[optional] #SBATCH --mem-per-cpu=4000mb         # Memory per thread; incompatible with --mem
#
# ---- Partition, Walltime and Output ----
#
#[unconfig] #SBATCH --array=01-10    # Create a job array. Useful for multiple, similar jobs. To use, read this: https://slurm.schedmd.com/job_array.html
#SBATCH --partition=gpu1,gpu2         # Partition (queue). Avail: regular1, regular2, long1, long2, wide1, wide2, gpu1, gpu2. Multiple partitions are possible.
#SBATCH --time=12:00:00              # Time limit hrs:min:sec
#SBATCH --output=%j.o%j              # Standard output log in TORQUE-style -- WARNING: %x requires a new enough SLURM. Use %j for regular jobs and %A-%a for array jobs
#SBATCH --error=%j.e%j               # Standard error  log in TORQUE-style -- WARNING: %x requires a new enough SLURM. Use %j for regular jobs and %A-%a for array jobs



NOW=`date +%H:%M-%a-%d/%b/%Y`
echo '------------------------------------------------------'
echo 'This job is allocated on '$SLURM_JOB_CPUS_PER_NODE' cpu(s)'
echo 'Job is running on node(s): '
echo  $SLURM_JOB_NODELIST
echo '------------------------------------------------------'
echo 'WORKINFO:'
echo 'SLURM: job starting at           '$NOW
echo 'SLURM: sbatch is running on      '$SLURM_SUBMIT_HOST
echo 'SLURM: executing on cluster      '$SLURM_CLUSTER_NAME
echo 'SLURM: executing on partition    '$SLURM_JOB_PARTITION
echo 'SLURM: working directory is      '$SLURM_SUBMIT_DIR
echo 'SLURM: current home directory is '$(getent passwd $SLURM_JOB_ACCOUNT | cut -d: -f6)
echo ""
echo 'JOBINFO:'
echo 'SLURM: job identifier is         '$SLURM_JOBID
echo 'SLURM: job name is               '$SLURM_JOB_NAME
echo ""
echo 'NODEINFO:'
echo 'SLURM: number of nodes is        '$SLURM_JOB_NUM_NODES
echo 'SLURM: number of cpus/node is    '$SLURM_JOB_CPUS_PER_NODE
echo 'SLURM: number of gpus/node is    '$SLURM_GPUS_PER_NODE
echo '------------------------------------------------------'


cd Entropy/src

module load gnu8/8.3.0 pgi/19.4
module load intelpython3/3.6.8 cuda/10.1

python3 main_cuda_dense_montecarlo.py

echo "Waiting for all the processes to finish..."
wait
