import { predictWithTrainedModel } from "./predict.ts";

const emailsFolder = "./emails";
const logsFolder = "./logs";
const resultsFile = `${logsFolder}/results.json`;
const metricsFile = `${logsFolder}/metrics.json`;

// ensure logs folder exists
await Deno.mkdir(logsFolder, { recursive: true });

// Evaluate classification results using filename-based ground truth
// mail*.txt → ham (label = 1, should be forwarded)
// spam*.txt → spam (label = 0, should be discarded)

// Computes standard classification metrics:
// precision, recall, F1-score, accuracy
async function evaluate(results: Array<{ email: string; classification: any }>) {

  let tp = 0, tn = 0, fp = 0, fn = 0;

  for (const r of results) {
    const name = r.email.toLowerCase();

    // Infer gold label from filename
    const gold = name.startsWith("mail") ? 1 : name.startsWith("spam") ? 0 : null;
    if (gold === null) continue;

    // Model prediction: ham if score >= 0.5
    const pred = r.classification.ham_score >= 0.5 ? 1 : 0;

    // Confusion matrix updates
    if (pred === 1 && gold === 1) tp++;
    if (pred === 0 && gold === 0) tn++;
    if (pred === 1 && gold === 0) fp++;
    if (pred === 0 && gold === 1) fn++;
  }

  // Metric calculations
  const precision = tp + fp ? tp / (tp + fp) : 0;
  const recall = tp + fn ? tp / (tp + fn) : 0;
  const f1 = (precision + recall) ? (2 * precision * recall) / (precision + recall) : 0;
  const accuracy = (tp + tn + fp + fn) ? (tp + tn) / (tp + tn + fp + fn) : 0;

  const m = {
    timestamp: new Date().toISOString(),
    tp, tn, fp, fn,
    precision,
    recall,
    f1,
    accuracy,
  };

  // Append metric entry to file
  let logArray = [];
  try {
    logArray = JSON.parse(await Deno.readTextFile(metricsFile));
  } catch (_) {}
  logArray.push(m);

  await Deno.writeTextFile(metricsFile, JSON.stringify(logArray, null, 2));
  console.log("Evaluation metrics updated:", m);
}

// Holds classification results for all emails
const results: Array<{ email: string; classification: any }> = [];

// Classify every email file using the trained logistic regression model
for await (const entry of Deno.readDir(emailsFolder)) {
  if (entry.isFile && entry.name.endsWith(".txt")) {
    const path = `${emailsFolder}/${entry.name}`;
    const emailText = await Deno.readTextFile(path);

    // Predict ham probability using trained model
    const ham_score = await predictWithTrainedModel(emailText);

    // Final decision threshold
    const classification = {
      ham_score,
      recommendation: ham_score >= 0.5 ? "forward" : "discard",
    };

    console.log(`Email: ${entry.name}`);
    console.log(classification, "\n");

    results.push({
      email: entry.name,
      classification,
    });
  }
}

// Save raw classification results
await Deno.writeTextFile(resultsFile, JSON.stringify(results, null, 2));
console.log(`Results saved to ${resultsFile}`);

// Run evaluation after classification of emails
await evaluate(results);