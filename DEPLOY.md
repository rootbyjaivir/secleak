# Publish and deploy

## 1. GitHub
```bash
cd ~/secleak
git init -b main && git add -A && git commit -m "SecLeak: initial release"
# create an empty repo named secleak on github.com first, then:
git remote add origin git@github.com:YOUR_USER/secleak.git
git push -u origin main
```
Never commit `data/` or `results/` raw files (already in `.gitignore`). Commit the small
`results/report/*` and audit cards you cite in the paper by removing them from the ignore rule
or using `git add -f`.

## 2. Free web app (Streamlit Community Cloud)
1. share.streamlit.io -> New app -> pick the repo, branch `main`, main file `app/streamlit_app.py`.
2. Advanced settings: Python 3.12, requirements file `requirements-app.txt`.
3. Deploy. Users upload a CSV and get an audit card.

## 3. Alternative: Docker anywhere
```bash
docker build -t secleak . && docker run -p 8501:8501 secleak
```

## 4. Alternative: PyPI
```bash
pip install build twine && python -m build && twine upload dist/*
```
Check the name `secleak` is free on pypi.org first; otherwise rename in `pyproject.toml`.
