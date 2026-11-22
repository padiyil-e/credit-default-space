# Credit default model: Hugging Face Space with GitHub Actions CI/CD

Deploys the tuned random forest from Module 8 as a Gradio app on Hugging Face Spaces.
Every push to `main` runs the tests; only if they pass is the `space/` folder synced to the Space.

```
.github/workflows/ci-cd.yml   test job -> deploy job (huggingface/hub-sync)
data/credit_default.csv       UCI Default of Credit Card Clients (Yeh, 2016), 30,000 rows
training/train.py             rebuilds space/model/ from the data
space/                        everything the Space needs (this is all that gets deployed)
  README.md                   Space config (sdk, sdk_version, python_version)
  app.py, features.py         Gradio UI and the shared feature pipeline
  requirements.txt            pinned versions (must match the version that saved the model)
  model/model.joblib          trained model, 2.9 MB
  model/metadata.json         params, threshold, test metrics, library versions
tests/                        feature, model quality gate, and app tests
```

## One-time setup

1. Create a new GitHub repository and push this folder to its `main` branch.
2. On Hugging Face, create a write access token (Settings > Access Tokens).
3. In the GitHub repo, Settings > Secrets and variables > Actions:
   - Secret `HF_TOKEN` = the token.
   - Variable `HF_SPACE_ID` = `your-hf-username/credit-default-risk`.
4. Settings > Environments: create an environment named `production`
   (optionally add yourself as a required reviewer for manual approval).
5. Push to `main` or run the workflow from the Actions tab. The first run creates the Space.

## Local use

```bash
pip install -r requirements-dev.txt
python training/train.py      # retrain (about 3 s)
pytest -v                     # 13 tests
cd space && python app.py     # http://127.0.0.1:7860
```

If you retrain with a different scikit-learn version, update the pin in
`space/requirements.txt`; a test fails if they differ.
