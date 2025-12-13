import { getEmbedding } from "./embeddings.ts";
import { ensureDir } from "https://deno.land/std/fs/mod.ts";

const logsFolder = "./logs";
const modelFile = `${logsFolder}/model.json`;

// Load a JSONL dataset
// Each line is a JSON object: { text, label }
async function loadDataset(path: string) {
    const text = await Deno.readTextFile(path);
    return text
        .split("\n")
        .map((l) => l.trim())
        .filter(Boolean)
        .map((l) => JSON.parse(l)); // {text, label}
}

// Sigmoid activation function
// Converts linear score → probability (0..1)
function sigmoid(x: number) {
    return 1 / (1 + Math.exp(-x));
}

// Train a logistic regression spam classifier
export async function trainModel(
    trainPath = "./data/train.jsonl",
    valPath = "./data/val.jsonl",
    epochs = 20,
) {
    // Ensure output directory exists
    await ensureDir(logsFolder);

    // Load training and validation datasets
    const train = await loadDataset(trainPath);
    const val = await loadDataset(valPath);

    console.log(`Loading embeddings…`);

    // Embed all emails
    const trainEmb = [];
    for (const t of train) {
        trainEmb.push({
            label: t.label,         // ground truth (0 = spam, 1 = ham)
            emb: await getEmbedding(t.text) }); // vector representation
    }

    // Dimensionality of the embedding vectors
    const dim = trainEmb[0].emb.length;

    // Initialize model parameters
    // Small random weights + zero bias
    let weights = Array(dim).fill(0).map(() => (Math.random() - 0.5) * 0.01);
    let bias = 0;

    // Learning rate for gradient descent
    const lr = 0.05;

    // Store validation accuracy per epoch
    const metrics: any[] = [];

    // Training loop (gradient descent)
    for (let epoch = 0; epoch < epochs; epoch++) {

        // Train on all training samples
        for (const item of trainEmb) {

            // Linear model: z = w·x + b
            const z = weights.reduce((s, w, i) => s + w * item.emb[i], bias);

            // Convert to probability
            const pred = sigmoid(z);

            // Error = prediction - true label
            const error = pred - item.label;

            // Gradient descent update
            for (let i = 0; i < dim; i++) {
                weights[i] -= lr * error * item.emb[i];
            }
            bias -= lr * error;
        }

        // Validation evaluation
        let correct = 0;
        let total = 0;

        for (const v of val) {
            const emb = await getEmbedding(v.text);
            const z = weights.reduce((s, w, i) => s + w * emb[i], bias);
            const pred = sigmoid(z) >= 0.5 ? 1 : 0;
            if (pred === v.label) correct++;
            total++;
        }

        const acc = correct / total;
        console.log(`Epoch ${epoch} accuracy = ${acc}`);

        // Track validation accuracy over time
        metrics.push({ epoch, accuracy: acc });
    }

    // Save trained model to disk
    const model = { weights, bias, metrics };
    await Deno.writeTextFile(modelFile, JSON.stringify(model, null, 2));

    console.log(`Model saved to ${modelFile}`);
    return model;
}

// Allow standalone execution
if (import.meta.main) {
    await trainModel("./data/train.jsonl", "./data/val.jsonl", 20);
}
