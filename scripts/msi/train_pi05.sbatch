#!/bin/bash
# π0.5 fine-tuning on MSI: one or more runs side by side, one GPU each, resumed across jobs.
#
# Submit from the humanoid-vla checkout on an MSI login node (docs/RUNBOOK.md, "π0.5 on MSI"):
#
#   sbatch scripts/msi/train_pi05.sbatch pi05_v2_expert:expert-float32 pi05_v2_full:full-bfloat16
#
# Each argument is RUN:SETUP (setups: humanoid_vla.policies.pi05.SETUPS). Run i trains on the
# job's GPU i with `scripts/train_pi05.py --resume`, so it continues from its latest checkpoint.
# Request one GPU per run (the default below is two A40s; override with --gres on the sbatch
# command line and set GPU_TYPE to match).
#
# GPU jobs stop after 24 hours. At its start the job queues a successor that starts once this job
# has ended, for any reason, and continues the runs that are not complete; when every run has
# completed or failed, the job cancels its successor. A run that exits with an error is marked
# outputs/train/<run>.failed and not retried. At most MAX_ATTEMPTS jobs are chained.
#
# Settings (environment variables, passed on to successors): DATASET (g1_sort_v2), STEPS (30000),
# BATCH_SIZE (32), SAVE_FREQ (5000), GPU_TYPE (a40), MAX_ATTEMPTS (4), and SAVE (1: write
# results/pi05/train_<run>.json when a run completes; 0 for test runs).
#SBATCH --job-name=hvla-pi05
#SBATCH --account=ztchen
#SBATCH --partition=interactive-gpu
#SBATCH --gres=gpu:a40:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --output=outputs/slurm/%x-%j.out

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:?run this script with sbatch}" || exit 2
if [[ ! -f scripts/train_pi05.py ]]; then
    echo "error: submit from the root of the humanoid-vla checkout" >&2
    exit 2
fi
export PATH="${HOME}/.local/bin:${PATH}"
export DATASET="${DATASET:-g1_sort_v2}" STEPS="${STEPS:-30000}" BATCH_SIZE="${BATCH_SIZE:-32}"
export SAVE_FREQ="${SAVE_FREQ:-5000}" GPU_TYPE="${GPU_TYPE:-a40}" MAX_ATTEMPTS="${MAX_ATTEMPTS:-4}"
export SAVE="${SAVE:-1}"
ATTEMPT="${ATTEMPT:-1}"

if [[ $# -eq 0 ]]; then
    echo "usage: sbatch scripts/msi/train_pi05.sbatch RUN:SETUP [RUN:SETUP ...]" >&2
    exit 2
fi

# A run is complete when its latest checkpoint is at STEPS.
is_complete() {
    local last="outputs/train/$1/checkpoints/last"
    [[ -L ${last} ]] && (( 10#$(basename "$(readlink -f "${last}")") >= STEPS ))
}

pending=()
for spec in "$@"; do
    run="${spec%%:*}"
    if [[ -e outputs/train/${run}.failed ]]; then
        echo "skip ${run}: it failed in an earlier job (outputs/train/${run}.failed)"
    elif is_complete "${run}"; then
        echo "skip ${run}: complete"
    else
        pending+=("${spec}")
    fi
done
echo "job ${SLURM_JOB_ID}, attempt ${ATTEMPT} of ${MAX_ATTEMPTS}, node ${SLURMD_NODENAME}:" \
    "${#pending[@]} run(s) to train: ${pending[*]:-none}"
if [[ ${#pending[@]} -eq 0 ]]; then
    exit 0
fi

IFS=, read -ra gpus <<< "${CUDA_VISIBLE_DEVICES:-}"
if (( ${#gpus[@]} < ${#pending[@]} )); then
    echo "error: ${#pending[@]} runs need ${#pending[@]} GPUs; this job has ${#gpus[@]}" >&2
    exit 2
fi
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
# GPU use once a minute, for sizing batches and spotting stalls.
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,memory.total --format=csv \
    -l 60 > "outputs/slurm/gpu-${SLURM_JOB_ID}.csv" 2>&1 &
monitor=$!

successor=""
if (( ATTEMPT < MAX_ATTEMPTS )); then
    if successor=$(sbatch --parsable --dependency="afterany:${SLURM_JOB_ID}" \
        --partition="${SLURM_JOB_PARTITION}" --gres="gpu:${GPU_TYPE}:${#pending[@]}" \
        --export="ALL,ATTEMPT=$((ATTEMPT + 1))" scripts/msi/train_pi05.sbatch "${pending[@]}"); then
        echo "queued successor job ${successor} (starts after this job ends)"
    else
        echo "warning: could not queue a successor; resubmit by hand if runs remain"
        successor=""
    fi
fi

save=()
[[ ${SAVE} == 1 ]] && save=(--save)
pids=()
for i in "${!pending[@]}"; do
    spec="${pending[$i]}"
    run="${spec%%:*}"
    setup="${spec#*:}"
    echo "start ${run} (${setup}) on GPU ${gpus[$i]}; log outputs/slurm/${run}-${SLURM_JOB_ID}.log"
    CUDA_VISIBLE_DEVICES="${gpus[$i]}" uv run --no-sync python scripts/train_pi05.py \
        --dataset "${DATASET}" --run "${run}" --setup "${setup}" --steps "${STEPS}" \
        --batch-size "${BATCH_SIZE}" --save-freq "${SAVE_FREQ}" --resume "${save[@]}" \
        > "outputs/slurm/${run}-${SLURM_JOB_ID}.log" 2>&1 &
    pids+=($!)
done

# Reached only if the runs end before the time limit: completed, or failed.
for i in "${!pending[@]}"; do
    run="${pending[$i]%%:*}"
    if wait "${pids[$i]}"; then
        echo "${run}: exited normally"
    else
        echo "${run}: failed (exit $?); marked outputs/train/${run}.failed, not retried"
        date > "outputs/train/${run}.failed"
    fi
done
kill "${monitor}" 2> /dev/null
left=0
for spec in "${pending[@]}"; do
    run="${spec%%:*}"
    if [[ ! -e outputs/train/${run}.failed ]] && ! is_complete "${run}"; then
        left=$((left + 1))
    fi
done
if [[ ${left} -eq 0 && -n ${successor} ]]; then
    scancel "${successor}" && echo "nothing left to train: cancelled successor ${successor}"
fi
