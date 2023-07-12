sbatch \
    --gpus=1 \
    --gres=gpumem:32g \
    --mem-per-cpu=32g \
    --error=jobs/llm-b.err \
    --output=jobs/llm-lemma.out \
    --wrap="CUDA_VISIBLE_DEVICES=0 python src/cil/llm/predict.py"