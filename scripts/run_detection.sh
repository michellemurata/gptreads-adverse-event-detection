#!/usr/bin/env bash
set -e

python src/detect_adverse_events.py \
  --input_csv example_data/synthetic_oper_notes.csv \
  --output_csv results/sample_predictions.csv \
  --prompt_name adverse_event \
  --text_column TEXT \
  --model_name GPT4 \
  --temperature 0.2 \
  --sleep_seconds 0
