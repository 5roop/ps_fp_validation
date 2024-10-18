# ps_fp_validation
Validation of our model for filled pauses

## Running:
```python
mamba env create -f fpval.yml
mamba activate fpval

python transcribe.py
python add_gold.py
python calculate_metrics.py
```