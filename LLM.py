%pip install --upgrade --quiet google-genai gitingest
import sys

# Additional authentication is required for Google Colab
if "google.colab" in sys.modules:
    # Authenticate user to Google Cloud
    from google.colab import auth

    auth.authenticate_user()

import os

from google import genai

# fmt: off
PROJECT_ID = "dsci303-test"  # @param {type: "string", placeholder: "[your-project-id]", isTemplate: true}

# fmt: on
if not PROJECT_ID or PROJECT_ID == "[your-project-id]":
    PROJECT_ID = str(os.environ.get("GOOGLE_CLOUD_PROJECT"))

LOCATION = os.environ.get("GOOGLE_CLOUD_REGION", "global")

client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)

import nest_asyncio
from IPython.display import Audio, Image, Markdown, Video, display
from gitingest import ingest
from google.genai.types import CreateCachedContentConfig, GenerateContentConfig, Part

nest_asyncio.apply()

MODEL_ID = "gemini-2.5-flash"  # @param {type: "string"}


sample_index = 0
x = X_test.iloc[[sample_index]] # Use iloc to get a DataFrame row
true_label = y_test.iloc[sample_index] # Use iloc for Series

# Convert to a readable table
sample_df = x

print("\nTest sample:")
display(sample_df)
print("True label:", true_label)

prompt = f"""
You are an expert curling analyst. You are given a curling shot sample with the following features:

{sample_df.to_string(index=False)}

Possible points (Result of the shot):
0 = Shot failed its task
1 = Shot mostly failed task
2 = Shot mostly succeeded task
3 = Shot performed tax semi accurately
4 = Shot performed task accurately

Based on the feature values, predict the 'Points' for this shot (0, 1, 2, 3, or 4).
Only output the predicted points number.

Explain why you predict that way.
"""

print(prompt)

# -----------------------------
# 4. Send to Gemini
# -----------------------------

response = client.models.generate_content(
    config= GenerateContentConfig(temperature=1.0),
    model=MODEL_ID,
    contents=prompt
)

print("\nGemini Response:\n")
print(response.text)

print(true_label)

import pandas as pd

gemini_preds = []
gemini_reasons = []

N = 5  # number of samples to test

for i in range(N):
    x = X_test.iloc[[i]] # Use iloc to get a DataFrame row
    sample_df = x

    prompt = f"""
You are an expert curling analyst. You are given a curling shot sample with the following features:

{sample_df.to_string(index=False)}

Possible points (Result of the shot):
0 = Shot failed its task
1 = Shot mostly failed task
2 = Shot mostly succeeded task
3 = Shot performed tax semi accurately
4 = Shot performed task accurately

First, on the first line, output ONLY the predicted points number (0, 1, 2, 3, or 4).
Then, on the following lines, explain why you predict that way.
"""

    response = client.models.generate_content(
        model=MODEL_ID,
        contents=prompt
    )

    full_text = response.text.strip()
    lines = full_text.splitlines()

    # First line: class prediction
    raw_pred = lines[0].strip() if lines else ""
    try:
        pred = int(raw_pred)
    except:
        pred = None

    # Remaining lines: reasoning
    reasoning = "\n".join(lines[1:]).strip() if len(lines) > 1 else ""

    gemini_preds.append(pred)
    gemini_reasons.append(reasoning)

# Build results table
results = pd.DataFrame({
    "sample_index": range(N),
    "gemini_pred": gemini_preds,
    "true_label": y_test.iloc[:N],
    "reasoning": gemini_reasons
})

results
