import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER_PATH = "outputs/qwen2_5_penguins_qlora"


def main() -> None:
    print("Iniciando teste do modelo...")

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    print("Carregando tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Carregando modelo base...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=quantization_config,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    print("Carregando adaptador...")
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()

    prompt = (
        "### Instrução:\n"
        "Explique onde os pinguins vivem e como sobrevivem ao frio.\n\n"
        "### Resposta:\n"
    )

    print("Tokenizando prompt...")
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    print("Gerando resposta...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=80,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print("\n===== SAÍDA BRUTA =====")
    print(decoded)

    if "### Resposta:" in decoded:
        resposta = decoded.split("### Resposta:", 1)[1].strip()
    else:
        resposta = decoded.strip()

    print("\n===== RESPOSTA EXTRAÍDA =====")
    print(resposta if resposta else "[vazio]")


if __name__ == "__main__":
    main()