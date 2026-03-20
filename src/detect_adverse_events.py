import argparse
import os
import time
from pathlib import Path

import pandas as pd
from openai import AzureOpenAI


def clean_text_column(df: pd.DataFrame, text_column: str) -> pd.DataFrame:
    df = df.copy()
    df[text_column] = df[text_column].astype(str).str.replace('"', "'", regex=False)
    return df


def clean_first_word(first_word: str) -> str:
    return first_word.replace(",", "").replace(".", "").strip()


def load_prompt(prompt_name: str) -> str:
    prompt_files = {
        "durotomy": Path("prompts/durotomy_classification_prompt.txt"),
        "adverse_event": Path("prompts/adverse_event_classification_prompt.txt"),
    }

    if prompt_name not in prompt_files:
        raise ValueError(
            f"Invalid prompt_name: {prompt_name}. Use one of {list(prompt_files.keys())}."
        )

    prompt_path = prompt_files[prompt_name]
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

    return prompt_path.read_text(encoding="utf-8").strip()


def build_client() -> AzureOpenAI:
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")

    if not endpoint or not api_key:
        raise EnvironmentError(
            "Missing AZURE_OPENAI_ENDPOINT and/or AZURE_OPENAI_API_KEY environment variables."
        )

    return AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version=api_version,
    )


def run_inference(
    input_csv: str,
    output_csv: str,
    prompt_name: str,
    text_column: str,
    model_name: str,
    temperature: float,
    sleep_seconds: float,
) -> None:
    client = build_client()
    prompt = load_prompt(prompt_name)

    notes = pd.read_csv(input_csv)
    if text_column not in notes.columns:
        raise ValueError(
            f"Column '{text_column}' not found in input file. Found columns: {list(notes.columns)}"
        )

    notes = clean_text_column(notes, text_column)

    results = []

    for i, note in enumerate(notes[text_column]):
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": note},
            ],
            temperature=temperature,
        )

        response_text = response.choices[0].message.content.strip()
        split_response = response_text.split(maxsplit=1)

        if len(split_response) == 2:
            first_word, details = split_response
        else:
            first_word = response_text
            details = ""

        results.append(
            {
                "Prediction": clean_first_word(first_word),
                "Details": details,
                "RawResponse": response_text,
            }
        )

        print(f"Processed note {i + 1}/{len(notes)}")
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    results_df = pd.DataFrame(results)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Azure OpenAI inference on operative notes."
    )
    parser.add_argument("--input_csv", required=True, help="Path to input CSV file")
    parser.add_argument("--output_csv", required=True, help="Path to output CSV file")
    parser.add_argument(
        "--prompt_name",
        required=True,
        choices=["durotomy", "adverse_event"],
        help="Which predefined prompt to use",
    )
    parser.add_argument(
        "--text_column",
        default="TEXT",
        help="Name of the column containing operative note text",
    )
    parser.add_argument(
        "--model_name",
        default="GPT4",
        help="Azure OpenAI deployment/model name",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--sleep_seconds",
        type=float,
        default=0.0,
        help="Seconds to sleep between API calls",
    )

    args = parser.parse_args()

    run_inference(
        input_csv=args.input_csv,
        output_csv=args.output_csv,
        prompt_name=args.prompt_name,
        text_column=args.text_column,
        model_name=args.model_name,
        temperature=args.temperature,
        sleep_seconds=args.sleep_seconds,
    )
