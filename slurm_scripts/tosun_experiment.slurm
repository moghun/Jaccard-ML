#!/bin/bash
#SBATCH --partition=cuda 
#SBATCH --account=cuda
#SBATCH --qos=cuda
#SBATCH --nodelist=cn04
#SBATCH -J jaccard
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH --mem=60G
#SBATCH --time=0-10:00:00
#SBATCH --output=out/jaccard-%j.out
##SBATCH --error=jaccard-%j.err

echo "SLURM NODELIST $SLURM_NODELIST"
echo "NUMBER OF SLURM CORES $SLURM_NTASKS"

# Setup
module purge 
module load cuda/10.2
module load cmake/3.16
module load gcc-5.3.0-gcc-9.2.0-7hkwitl

GRAPHS=(
	youtube_c.graph
        wiki-topcats_c.graph
        com-orkut_c.graph
        com-lj_c.graph
        soc-LiveJournal_c.graph
	indochina-2004_c.graph
	wb-edu_c.graph
	uk-2002_c.graph
        hyperlink2012_c.graph
        twitter_rv_c.graph
        soc-sinaweibo_c.graph
	com-friendster_c.graph
       )
# [SLURM] Job ID
JOB_ID=${SLURM_JOB_ID}
# Jaccard-ML code directory
JACCARD_PATH=/cta/users/aaljundi/Jaccard-ML/
# Graph files directory
DATA_PATH=/cta/users/aaljundi/graphs/uedgelist/
# Experiment parameter file
EXPERIMENT_PARAMS=${JACCARD_PATH}/parameters/experiment.json
# Number of runs to average over
AVG=10
# Experiment identifier
OUTPUT_FILE=model
CPU_THREADS=20
# [SLURM] Hostname of node running experiment 
NODE="$(echo -e "${SLURM_NODELIST}" | tr -d '[:space:]')"
# Folder to place result JSONs
RES_PATH=${NODE}/
# Folder to build executable
BUILD_FOLDER=${JACCARD_PATH}/build


export OMP_NUM_THREADS=${CPU_THREADS}
echo "Using ${CPU_THREADS} threads"
mkdir ${BUILD_FOLDER}
cd ${BUILD_FOLDER}
cmake ${JACCARD_PATH} -D_CPU=OFF -D CMAKE_C_COMPILER=$(which gcc) -D CMAKE_CXX_COMPILER=$(which g++)
make
mkdir ${RES_PATH}


for G in ${GRAPHS[*]}
  do
    srun ./jaccard -i ${DATA_PATH}${G} -e ${EXPERIMENT_PARAMS} -a ${AVG} -j ${RES_PATH}${G}-j${SLURM_JOB_ID}-n${NODE}-avg${AVG}-th${CPU_THREADS}-ex_${EXEC}-${OUTPUT_FILE}.json
    echo "srun ./jaccard -i ${DATA_PATH}${G} -e ${EXPERIMENT_PARAMS} -a ${AVG} -j ${RES_PATH}${G}-j${SLURM_JOB_ID}-n${NODE}-avg${AVG}-th${CPU_THREADS}-ex_${EXEC}-${OUTPUT_FILE}.json"
  done
