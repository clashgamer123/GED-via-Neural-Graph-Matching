#!/bin/bash
# Batch driver: runs every (dataset × encoder-variant) combo in demo mode.
# Each run trains for 1 epoch then tests, appending one line per phase to result/results.txt.
# Stdout/stderr go to logs/<tag>.log so we can recover details if needed.
set -u
cd "$(dirname "$0")"
PY=/mnt/bee/data/sabil/miniconda3/envs/768/bin/python
mkdir -p logs model_save_runs
RESULTS=result/all_runs.tsv
echo -e "tag\tdataset\tphase\tvalue_dump" > "$RESULTS"

run_one () {
    local tag="$1"; shift
    local dataset="$1"; shift
    local extra="$*"
    local mp="model_save_runs/${tag}_"
    echo "=== [$tag] dataset=$dataset extra=$extra ==="
    # mark in results.txt so we can split later
    echo -e "### TAG=${tag} DATASET=${dataset} ###" >> result/results.txt
    $PY src/main.py --model-name GedGNN --dataset "$dataset" \
        --demo --model-epoch-start 0 --model-epoch-end 1 --model-train 1 \
        --model-path "$mp" $extra > "logs/${tag}.log" 2>&1
    echo "    -> done (rc=$?)"
}

DATASETS=(AIDS Linux IMDB)

for ds in "${DATASETS[@]}"; do
    run_one "${ds}_gin"             "$ds" --gnn-operator gin
    run_one "${ds}_gcn"             "$ds" --gnn-operator gcn
    run_one "${ds}_tf_h4_l3"        "$ds" --gnn-operator transformer --heads 4 --num-layers 3
    run_one "${ds}_tf_h4_l3_res"    "$ds" --gnn-operator transformer --heads 4 --num-layers 3 --residual
    run_one "${ds}_tf_h2_l3"        "$ds" --gnn-operator transformer --heads 2 --num-layers 3
    run_one "${ds}_tf_h8_l3"        "$ds" --gnn-operator transformer --heads 8 --num-layers 3
    run_one "${ds}_tf_h4_l2"        "$ds" --gnn-operator transformer --heads 4 --num-layers 2
    run_one "${ds}_tf_h4_l4"        "$ds" --gnn-operator transformer --heads 4 --num-layers 4
done

echo "ALL DONE"
