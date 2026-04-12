import json
import random
from pathlib import Path

from openai import OpenAI


OUTPUT_DIR = Path("data")
TRAIN_FILE = OUTPUT_DIR / "train.jsonl"
TEST_FILE = OUTPUT_DIR / "test.jsonl"

TOTAL_EXAMPLES = 60
TRAIN_RATIO = 0.9

SYSTEM_PROMPT = """
Você é um gerador de dataset para fine-tuning.
Sua tarefa é criar exemplos no domínio "pinguins".

Cada exemplo deve conter:
- "prompt": uma pergunta ou instrução do usuário
- "response": uma resposta clara, correta, objetiva e educativa

Regras:
- Gere exemplos variados.
- Foque em temas como habitat, alimentação, espécies, reprodução, adaptação ao frio, comportamento, curiosidades e diferenças entre espécies.
- Evite perguntas repetidas.
- Escreva tudo em português do Brasil.
- Não use markdown.
- Retorne apenas JSON válido.
- O formato de saída deve ser uma lista JSON com objetos no formato:
  [{"prompt": "...", "response": "..."}, ...]
"""

USER_PROMPT = f"""
Gere {TOTAL_EXAMPLES} exemplos sintéticos sobre pinguins para um dataset de instruções.

Os exemplos devem:
- ser variados
- ter perguntas naturais
- ter respostas curtas ou médias
- ser informativos e coerentes
- evitar informações obviamente inventadas

Retorne apenas a lista JSON.
"""


def save_jsonl(file_path: Path, records: list[dict]) -> None:
    with file_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    client = OpenAI()

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.8,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT},
        ],
    )

    content = response.choices[0].message.content.strip()
    examples = json.loads(content)

    if not isinstance(examples, list):
        raise ValueError("A resposta da API não retornou uma lista JSON.")

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
                    "text": f"### Instrução:\n{prompt}\n\n### Resposta:\n{response_text}"
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

    print(f"Total de exemplos válidos: {len(cleaned_examples)}")
    print(f"Treino: {len(train_data)} exemplos")
    print(f"Teste: {len(test_data)} exemplos")
    print(f"Arquivos salvos em: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()