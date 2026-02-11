## RANDose
Reproducing the RANDose paper 

# Getting started
1. Clone repo at ``https://github.com/iannyFARUE/Randose.git``
2. Install dependencies ``uv sync``
3. Run train loop ``uv run python train.py     --model Model_MTASP     --loss Loss_DC_PTV     --batch_size 2     --list_GPU_ids 0     --max_iter 80000     --project_name RANDose_reproduction``
