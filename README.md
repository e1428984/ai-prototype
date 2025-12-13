# Spam Email Classification Program

### A Reproducible ML System with Embeddings, Logistic Regression, and LLM Explanations

---

## 1. What this project does (in simple words)

This project builds a **spam email detector**.

Given an email:

* The system decides **“forward” (normal email)** or **“discard” (spam)**
* The decision is made by a **trained machine-learning model**
* An **LLM is used only to explain the decision**, not to make it

Everything runs **locally**, can be **reproduced**, and can be run by anyone using one command.

---

## 2. Why this approach was chosen

Instead of letting an LLM “guess” spam or not spam:

* We use a **classical ML classifier** for correctness and reproducibility
* We use **LLMs only for explanations**, which is safer and more transparent
* This avoids randomness and hidden behavior

The system separates:

* **Decision making** → machine learning model
* **Explanation** → language model

---

## 3. Models used (and why)

### 3.1 Embedding model: `mxbai-embed-large`

**What it does:**
Transforms raw email text into a long numeric vector (embedding).

Example:

```
"Reminder: meeting tomorrow" → [0.12, -0.44, 0.88, ...]
```

**Why we need it:**

* Machine learning models cannot work directly with text
* Embeddings convert meaning into numbers
* Similar emails get similar vectors

**Where it is used:**

* Training
* Validation
* Testing
* Inference (real email classification)

**File:** `src/embeddings.ts`

---

### 3.2 Classification model: Logistic Regression (custom)

**What it does:**

* Takes an embedding vector
* Computes a probability between 0 and 1
* Interprets it as **ham (normal email)** probability

**Decision rule:**

* `ham_score >= 0.5` → forward
* `ham_score < 0.5` → discard

**Why logistic regression:**

* Simple
* Explainable
* Deterministic
* Easy to evaluate
* Perfect for a prototype

**Stored in:**

```
logs/model.json
```

This file contains:

* Learned weights
* Bias
* Training metrics

---

### 3.3 Explanation model: `qwen2.5:1.5b`

**What it does:**

* Generates a short explanation **after** classification
* Does NOT affect decisions

**Example explanation:**

> “The email contains urgent language and a suspicious link, which is typical of spam.”

**Why this model is separate:**

* LLMs are slow and probabilistic
* Using them only for explanation avoids errors
* Keeps classification stable

**Used only in:**

* `src/aggregate.ts`

---

## 4. Dataset structure

All datasets use **JSONL (JSON Lines)** format.

### 4.1 `train.jsonl`

Used to train the classifier.

```
{"text":"Reminder: team sync tomorrow.","label":1}
{"text":"Win a free iPhone now!","label":0}
```

* `label = 1` → ham (normal email)
* `label = 0` → spam

---

### 4.2 `val.jsonl`

Used during training to check performance after each epoch.

This helps detect:

* Overfitting
* Underfitting

---

### 4.3 `test.jsonl`

Used only after training is complete.

This ensures:

* Fair evaluation
* No data leakage

---

## 5. Full code execution flow (step by step)

### STEP 1: Embeddings (`embeddings.ts`)

Every email text is converted into a numeric vector using:

```ts
ollama.embeddings({
  model: "mxbai-embed-large",
  prompt: text
});
```

This is the **foundation** of the system.

---

### STEP 2: Training (`train.ts`)

1. Load `train.jsonl` and `val.jsonl`
2. Convert all emails into embeddings
3. Initialize logistic regression weights
4. Train using gradient descent
5. After each epoch:

   * Evaluate on validation set
   * Log accuracy
6. Save model to `logs/model.json`

Output:

```
Epoch 0 accuracy = 1
Epoch 1 accuracy = 1
...
```

---

### STEP 3: Test evaluation (`eval_train.ts`)

1. Load `model.json`
2. Load `test.jsonl`
3. Predict spam/ham for each test email
4. Compute accuracy

This ensures the model works on **unseen data**.

---

### STEP 4: Prediction (`predict.ts`)

This file is the **inference core**.

Given an email:

1. Load trained weights
2. Compute embedding
3. Apply logistic regression
4. Output `ham_score`

No LLMs are involved here.

---

### STEP 5: Email classification (`main.ts`)

1. Read all `.txt` files in `emails/`
2. Predict `ham_score` for each email
3. Apply threshold:

   * ≥ 0.5 → forward
   * < 0.5 → discard
4. Save results
5. Compute metrics:

   * Precision
   * Recall
   * F1-score
   * Accuracy
6. Append metrics to `logs/metrics.json`

---

### STEP 6: Aggregation & explanation (`aggregate.ts`)

This step improves robustness and explainability.

For each email:

1. Run multiple predictions
2. Perform majority voting
3. Decide final action
4. Ask LLM to explain the final decision
5. Save output to `logs/aggregation.json`

Important:

> The LLM **never decides spam vs ham**.

---

### STEP 7: Orchestration (`run_all.ts`)

This file runs everything in the correct order:

1. Train
2. Evaluate
3. Classify emails
4. Aggregate + explain

This guarantees **reproducibility**.

---

## 6. How to run the project

### Requirements

* Deno
* Ollama
* Models:

  ```
  ollama pull mxbai-embed-large
  ollama pull qwen2.5:1.5b
  ```

### Run everything

```bash
deno task all
```

---

## 7. Outputs

| File                    | Purpose                        |
| ----------------------- | ------------------------------ |
| `logs/model.json`       | Trained classifier             |
| `logs/results.json`     | Per-email predictions          |
| `logs/metrics.json`     | Performance history            |
| `logs/aggregation.json` | Final decisions + explanations |

---

## 8. Why this system is correct and reproducible

* No black-box LLM decisions
* Deterministic ML model
* Fixed datasets
* Clear separation of responsibilities
* Single command to reproduce everything

---
