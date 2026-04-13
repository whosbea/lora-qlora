import json
import random
import re
from pathlib import Path
import os

from google import genai


OUTPUT_DIR = Path("data")
TRAIN_FILE = OUTPUT_DIR / "train.jsonl"
TEST_FILE = OUTPUT_DIR / "test.jsonl"

TOTAL_EXAMPLES = 60
TRAIN_RATIO = 0.9


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


def main() -> None:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("A variável de ambiente GEMINI_API_KEY não foi encontrada.")

    client = genai.Client(api_key=api_key)

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=f"{SYSTEM_PROMPT}\n\n{USER_PROMPT}",
    )

    raw_text = response.text
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

    print(f"Total de exemplos válidos: {len(cleaned_examples)}")
    print(f"Treino: {len(train_data)} exemplos")
    print(f"Teste: {len(test_data)} exemplos")
    print(f"Arquivos salvos em: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()