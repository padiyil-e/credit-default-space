---
title: Credit Default Risk
emoji: 💳
colorFrom: blue
colorTo: indigo
sdk: static
app_file: index.html
pinned: false
short_description: Tuned random forest predicting credit card default
---

# Credit card default risk

Static web app for the tuned random forest from Module 8, trained on the UCI Default of
Credit Card Clients data (Yeh, 2016). The forest is exported to `model.json` and run in the
browser by `model.js`, so the Space needs no server and runs on Hugging Face's free tier.
It returns a default probability and flags a client when the probability is at least 0.29.

Deployed automatically from GitHub by GitHub Actions; do not edit files here.
Course project only, not for real lending decisions.
