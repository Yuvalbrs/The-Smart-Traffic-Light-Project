# T-04-02/T-04-03 - final evaluation analysis (preregistration.md s9):
# "a single committed analysis notebook (analysis/final_eval.ipynb), runnable via make eval-final."

PYTHON := .venv/Scripts/python.exe

.PHONY: eval-final

eval-final:
	$(PYTHON) -m scripts.build_analysis
	$(PYTHON) -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=300 analysis/final_eval.ipynb
