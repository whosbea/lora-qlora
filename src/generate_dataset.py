import json
import random
import re
from pathlib import Path
import os

import requests


OUTPUT_DIR = Path("data")
TRAIN_FILE = OUTPUT_DIR / "train.jsonl"
TEST_FILE = OUTPUT_DIR / "test.jsonl"

TOTAL_EXAMPLES = 60
TRAIN_RATIO = 0.9

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Troca aqui pelo modelo que tu quiser usar no OpenRouter
# Exemplos comuns:
# - "openai/gpt-3.5-turbo"
# - "openai/gpt-4o"
MODEL_NAME = "openai/gpt-3.5-turbo"

SYSTEM_PROMPT = """
Você é um gerador de dataset sintético para fine-tuning de um modelo de linguagem.

Sua tarefa é criar exemplos no domínio "pinguins".

Regras:
- Gere exatamente a quantidade de exemplos solicitada.
- Cada exemplo deve ser um objeto JSON com as chaves:
  - "prompt"
  - "response"
- Escreva tudo em português do Brasil.
- Os prompts devem parecer perguntas ou instruções reais de usuários.
- As respostas devem ser claras, corretas, curtas ou médias e educativas.
- Varie os temas: habitat, alimentação, espécies, reprodução, adaptação ao frio, comportamento, curiosidades e diferenças entre espécies.
- Evite repetição.
- Não use markdown.
- Retorne apenas JSON válido.
- O formato final deve ser uma lista JSON.
"""

USER_PROMPT = f"""
Gere exatamente {TOTAL_EXAMPLES} exemplos sintéticos sobre pinguins.

Formato esperado:
[
  {{
    "prompt": "...",
    "response": "..."
  }}
]

Retorne apenas a lista JSON.
"""


def extract_json(text: str) -> str:
    """
    Tenta extrair apenas o bloco JSON da resposta do modelo.
    """
    text = text.strip()

    # Remove blocos ```json ... ```
    text = re.sub(r"^```json\s*", "", text)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    return text.strip()


def save_jsonl(file_path: Path, records: list[dict]) -> None:
    with file_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def generate_with_openrouter(
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
) -> str:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",

        # Opcionais:
        # "HTTP-Referer": "https://seusite.com",
        # "X-OpenRouter-Title": "Gerador de Dataset",
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.8,
    }

    response = requests.post(
        OPENROUTER_URL,
        headers=headers,
        json=payload,
        timeout=120,
    )

    # Lança erro HTTP se vier 4xx/5xx
    response.raise_for_status()

    data = response.json()

    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as e:
        raise ValueError(
            f"Não foi possível extrair o conteúdo da resposta da API. Resposta recebida: {data}"
        ) from e


def main() -> None:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("A variável de ambiente OPENROUTER_API_KEY não foi encontrada.")

    raw_text = generate_with_openrouter(
        api_key=api_key,
        model=MODEL_NAME,
        system_prompt=SYSTEM_PROMPT,
        user_prompt=USER_PROMPT,
    )

    json_text = extract_json(raw_text)
    examples = json.loads(json_text)

    if not isinstance(examples, list):
        raise ValueError("A resposta do modelo não retornou uma lista JSON.")

    cleaned_examples = []
    for item in examples:
        if not isinstance(item, dict):
            continue

        prompt = str(item.get("prompt", "")).strip()
        response_text = str(item.get("response", "")).strip()

        if prompt and response_text:
            cleaned_examples.append(
                {
                    "prompt": prompt,
                    "response": response_text,
                    "text": f"### Instrução:\n{prompt}\n\n### Resposta:\n{response_text}",
                }
            )

    if len(cleaned_examples) < 50:
        raise ValueError(
            f"Foram gerados apenas {len(cleaned_examples)} exemplos válidos. "
            "O laboratório exige pelo menos 50."
        )

    random.shuffle(cleaned_examples)

    split_index = int(len(cleaned_examples) * TRAIN_RATIO)
    train_data = cleaned_examples[:split_index]
    test_data = cleaned_examples[split_index:]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    save_jsonl(TRAIN_FILE, train_data)
    save_jsonl(TEST_FILE, test_data)

    print(f"Modelo usado: {MODEL_NAME}")
    print(f"Total de exemplos válidos: {len(cleaned_examples)}")
    print(f"Treino: {len(train_data)} exemplos")
    print(f"Teste: {len(test_data)} exemplos")
    print(f"Arquivos salvos em: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()