# Fine-tuned multilingual cross-encoder submission v1.5

This submission uses the cross-encoder fine-tuned on the joint dataset with
human labels, confident LLM labels, hard negatives, and reversed pairs.

The model is loaded locally from `/app/model`; no Internet access is needed
at runtime. The runner keeps the standard submission interface:

```bash
python -u run_submission.py \
  --items_path /path/to/items.parquet \
  --matches_path /path/to/matches.parquet \
  --output_path /path/to/predictions.csv
```

The model uses the same product text format as training:

```text
Name: ... Category: ... Attributes: key: value ...
```

Training report: 6,500,208 train pairs, 16,000 optimizer steps, maximum
sequence length 256, and learning rate `3e-7`. The human validation split
reported overall average precision `0.7712242248` and macro average precision
`0.7131270654`.
