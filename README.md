# Heart Attack Classification - VS Code Project

## Files
- `89026_updated_practical_github.ipynb`: original Jupyter notebook
- `run_analysis.py`: terminal-run version of the original notebook
- `heart_2022_no_nans.csv`: dataset
- `requirements.txt`: required packages

## Run in VS Code on Windows
1. Extract this ZIP and open the folder in VS Code.
2. Open Terminal > New Terminal.
3. Run `py -m venv .venv`.
4. Run `.\.venv\Scripts\python.exe -m pip install -r requirements.txt`.
5. Run `.\.venv\Scripts\python.exe run_analysis.py`.
6. To use the notebook, install the Python and Jupyter extensions, select the `.venv` kernel, and use Run All.

## Notes
- `run_analysis.py` follows the original notebook model settings, including 100 trees for Random Forest. These settings **do not match** the dissertation's described class weights and tree settings. Reconcile model settings and rerun before using results as dissertation evidence.
- The source ZIP did not include a Streamlit dashboard or saved model. This ZIP is an analysis-only project.
- The CSV is ~82 MB; GitHub browser uploads allow at most 25 MiB per file. The `.gitignore` excludes the CSV from commits. Keep the CSV locally for execution, or configure Git LFS for large file versioning.
