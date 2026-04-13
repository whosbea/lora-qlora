import torch
from pathlib import Path

from datasets import load_dataset
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer


MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
TRAIN_FILE = "data/train.jsonl"
TEST_FILE = "data/test.jsonl"
OUTPUT_DIR = "outputs/qwen2_5_penguins_qlora"

MAX_LENGTH = 512


def prepare_example(example: dict) -> dict:
    prompt = str(example.get("prompt", "")).strip()
    response = str(example.get("response", "")).strip()

    return {
        "prompt": f"### Instrução:\n{prompt}\n\n### Resposta:\n",
        "completion": response,
    }


def main() -> None:
    if not Path(TRAIN_FILE).exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {TRAIN_FILE}")

    if not Path(TEST_FILE).exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {TEST_FILE}")

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=quantization_config,
        dtype=torch.float16,
        trust_remote_code=True,
    )

    model.config.use_cache = False
    model.config.pad_token_id = tokenizer.pad_token_id

    peft_config = LoraConfig(
        r=64,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )

    dataset = load_dataset(
        "json",
        data_files={
            "train": TRAIN_FILE,
            "test": TEST_FILE,
        },
    )

    dataset["train"] = dataset["train"].map(prepare_example)
    dataset["test"] = dataset["test"].map(prepare_example)

    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        learning_rate=2e-4,
        logging_steps=5,
        save_steps=50,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        report_to="none",
        max_length=MAX_LENGTH,
        packing=False,
        fp16=True,
        optim="paged_adamw_32bit",
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    trainer.train()

    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print(f"Adaptador salvo em: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()