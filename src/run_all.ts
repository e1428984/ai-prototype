// Script that runs the full ML pipeline

// 1) Train the model
// Reads train.jsonl + val.jsonl
// Generates embeddings
// Trains a logistic regression classifier
// Saves weights + bias to logs/model.json
console.log("\n=== TRAINING MODEL ===");
await Deno.run({
    cmd: [
        "deno",
        "run",
        "--allow-net", // needed for Ollama API calls
        "--allow-read",   // read training data
        "--allow-write",  // save trained model
        "src/train.ts",
    ],
}).status();

// 2) Evaluate the trained model
// Loads logs/model.json
// Runs predictions on test.jsonl
// Prints test accuracy
console.log("\n=== EVALUATING TRAINED MODEL ===");
await Deno.run({
    cmd: [
        "deno",
        "run",
        "--allow-net",
        "--allow-read",
        "--allow-write",
        "src/eval_train.ts",
    ],
}).status();

// 3) Classify real email files
// Uses the trained model only (NO LLM)
// Produces ham_score + recommendation
// Saves results to logs/results.json
// Computes precision / recall / F1
console.log("\n=== CLASSIFYING EMAILS ===");
await Deno.run({
    cmd: [
        "deno",
        "run",
        "--allow-net",
        "--allow-read",
        "--allow-write",
        "src/main.ts",
    ],
}).status();

// 4) Aggregation mode
// Runs N predictions per email
// Majority vote determines final decision
// LLM is used only to explain the final decision
// Saves output to logs/aggregation.json
console.log("\n=== AGGREGATION MODE (20 SAMPLES) ===");
await Deno.run({
    cmd: [
        "deno",
        "run",
        "--allow-net",
        "--allow-read",
        "--allow-write",
        "src/aggregate.ts",
        "5",                // number of samples per email
    ],
}).status();

console.log("\n=== ALL TASKS COMPLETED ===\n");
