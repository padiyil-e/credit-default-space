---
title: Credit Default Risk
emoji: 💳
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.32.0
python_version: "3.12"
app_file: app.py
pinned: false
short_description: Tuned random forest predicting credit card default
---

# Credit card default risk

Gradio demo of the tuned random forest from Module 8, trained on the UCI Default of
Credit Card Clients data (Yeh, 2016). It returns a default probability and flags a
client when the probability is at least 0.29, the threshold chosen on training data.

This Space is deployed automatically from GitHub by GitHub Actions; do not edit files here.
Course project only, not for real lending decisions.
