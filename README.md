# Credit default model: Hugging Face Space with GitHub Actions CI/CD

Deploys the tuned random forest from Module 8 as a free static Hugging Face Space.
Every push to `main` runs the tests; only if they pass is the `space/` folder synced to the Space.

Hugging Face now requires a PRO subscription to host Gradio or Docker Spaces on the free
CPU Basic hardware, while static Spaces stay free. The forest is therefore exported to JSON
and run in the visitor's browser by a small JavaScript file, and the tests check that the
browser gives exactly the same predictions as scikit-learn.

```
.github/workflows/ci-cd.yml   test job -> deploy job (huggingface/hub-sync, space_sdk: static)
data/credit_default.csv       UCI Default of Credit Card Clients (Yeh, 2016), 30,000 rows
training/features.py          feature pipeline (29 inputs)
training/train.py             trains the model and exports it for the browser
model/model.joblib            scikit-learn model (not deployed; used by the tests)
space/                        everything the Space serves (this is all that gets deployed)
  README.md                   Space config (sdk: static, app_file: index.html)
  index.html                  the web app (single client + batch CSV scoring)
  model.js                    feature pipeline + forest inference in JavaScript
  model.json                  the 200 trees as arrays (2.7 MB)
  metadata.json               params, threshold, test metrics, library versions
tests/                        13 tests: features, quality gate, browser model vs scikit-learn
requirements.txt              pinned training libraries; requirements-dev.txt adds pytest
```

## One-time setup

1. Push this folder to the `main` branch of a GitHub repository.
2. Create a Hugging Face access token with Write permission.
3. In the GitHub repo, Settings > Secrets and variables > Actions:
   - Secret `HF_TOKEN` = the token.
   - Variable `HF_SPACE_ID` = `your-hf-username/credit-default-risk`.
4. Push to `main` or run the workflow from the Actions tab. The first run creates the Space.

## Local use

```bash
pip install -r requirements-dev.txt   # Node.js is also needed for the browser-model tests
python training/train.py              # retrain and re-export (about 3 s)
pytest -v
cd space && python -m http.server     # open http://localhost:8000
```
