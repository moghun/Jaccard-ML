#!/bin/bash
#SBATCH -J jac_mod
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 20
#SBATCH --partition=akya-cuda 
##SBATCH --constraint=akya-cuda
#SBATCH --gres=gpu:1
#SBATCH --time=0-10:00:00
#SBATCH --output=res/jaccard-%j.out

echo "SLURM NODELIST $SLURM_NODELIST"
echo "NUMBER OF SLURM CORES $SLURM_NTASKS"

# Setup
module purge 
module load centos7.3/comp/cmake/3.18.0
module load centos7.3/comp/gcc/7
module load centos7.3/lib/cuda/10.1

source /truba/home/aalabsialjundi/anaconda3/bin/activate
conda init

# [SLURM] Job ID
JOB_ID=${SLURM_JOB_ID}
# Jaccard-ML code directory
JACCARD_PATH=/truba/home/aalabsialjundi/Jaccard-ML/
# Graph files directory
DATA_PATH=/truba/home/aalabsialjundi/graphs/
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

        #com-amazon_c.graph
        #com-dblp_c.graph
GRAPHS=(
        com-friendster_c.graph
        com-lj_c.graph
        com-orkut_c.graph
        flickr_c.graph
        hyperlink2012_c.graph
        indochina-2004_c.graph
        REDDIT-MULTI-12k_c.graph
        soc-LiveJournal_c.graph
        soc-sinaweibo_c.graph
        twitter_rv_c.graph
        uk-2002_c.graph
        wb-edu_c.graph
        wiki-topcats_c.graph
        youtube_c.graph
       )


export OMP_NUM_THREADS=${CPU_THREADS}
echo "Using ${CPU_THREADS} threads"
mkdir ${BUILD_FOLDER}
cd ${BUILD_FOLDER}
cmake ${JACCARD_PATH} -D_CPU=OFF
make
mkdir ${RES_PATH}


for G in ${GRAPHS[*]}
  do
    srun ./jaccard -i ${DATA_PATH}${G} -e ${EXPERIMENT_PARAMS} -a ${AVG} -j ${RES_PATH}${G}-j${SLURM_JOB_ID}-n${NODE}-avg${AVG}-th${CPU_THREADS}-ex_${EXEC}-${OUTPUT_FILE}.json
    echo "srun ./jaccard -i ${DATA_PATH}${G} -e ${EXPERIMENT_PARAMS} -a ${AVG} -j ${RES_PATH}${G}-j${SLURM_JOB_ID}-n${NODE}-avg${AVG}-th${CPU_THREADS}-ex_${EXEC}-${OUTPUT_FILE}.json"
  done
