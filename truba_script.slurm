#!/bin/bash
#SBATCH -J jac_mod
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 20
#SBATCH --partition=akya-cuda 
##SBATCH --constraint=akya-cuda
#SBATCH --gres=gpu:1
#SBATCH --time=5-00:00:00
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


GRAPHS=(
        com-amazon_c.graph
        com-dblp_c.graph
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

JOB_ID=${SLURM_JOB_ID}
JACCARD_PATH=/truba/home/aalabsialjundi/Jaccard-ML/
DATA_PATH=/truba/home/aalabsialjundi/graphs/
EXPERIMENT_PARAMS=/truba/home/aalabsialjundi/Jaccard-ML/parameters/experiment.json
AVG=10
OUTPUT_FILE=model
CPU_THREADS=20
NODE="$(echo -e "${SLURM_NODELIST}" | tr -d '[:space:]')"
RES_PATH=${NODE}/
cd ${JACCARD_PATH}
export OMP_NUM_THREADS=${CPU_THREADS}
echo "Using ${CPU_THREADS} threads"
mkdir build
cd build
cmake .. -D_CPU=OFF
make
mkdir ${RES_PATH}


for G in ${GRAPHS[*]}
  do
    srun ./jaccard -i ${DATA_PATH}${G} -e ${EXPERIMENT_PARAMS} -a ${AVG} -j ${RES_PATH}${G}-j${SLURM_JOB_ID}-n${NODE}-avg${AVG}-th${CPU_THREADS}-ex_${EXEC}-${OUTPUT_FILE}.json
    echo "srun ./jaccard -i ${DATA_PATH}${G} -e ${EXPERIMENT_PARAMS} -a ${AVG} -j ${RES_PATH}${G}-j${SLURM_JOB_ID}-n${NODE}-avg${AVG}-th${CPU_THREADS}-ex_${EXEC}-${OUTPUT_FILE}.json"
  done
