from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch


def load_model(model_id: str, cache_dir: str) -> AutoModelForCausalLM:
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        cache_dir=cache_dir,
        device_map="auto",
        torch_dtype="auto",
    )
    return model


def get_pipeline(
    tokenizer: AutoTokenizer, model: AutoModelForCausalLM
) -> transformers.pipeline:
    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    )
    return pipeline


def generate_text(pipeline: transformers.pipeline, input_text: str, tokenizer: transformers.pipeline) -> str:
    sequences = pipeline(
        input_text,
        max_length=200,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )
    return sequences[0]["generated_text"]
