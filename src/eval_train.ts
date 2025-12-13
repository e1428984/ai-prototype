import ollama from "ollama";

const modelFile = "./logs/model.json";
const embedModel = "mxbai-embed-large"; // / Embedding model for training and evaluation

/**
 * Generate an embedding vector for a given text.
 * This converts raw text into numerical features used by the model
 */
async function getEmbedding(text: string): Promise<number[]> {
    const result = await ollama.embeddings({
        model: embedModel,
        prompt: text,
    });
    return result.embedding;
}

// Load a JSONL (JSON Lines) file
async function loadJSONL(path: string) {
    return Deno.readTextFile(path)
        .then((t) =>
            t.split("\n").map((l) => l.trim()).filter(Boolean).map(JSON.parse)
        );
}

/**
 * Evaluate the trained model on a test dataset
 * 1. Generate an embedding
 * 2. Compute the logistic regression score
 * 3. Convert the score to a binary prediction
 * 4. Compare against the ground-truth label
 */
export async function evaluateModel(testPath = "./data/test.jsonl") {
    // Load trained model parameters
    const model = JSON.parse(await Deno.readTextFile(modelFile));
    const { weights, bias } = model;

    // Load test dataset
    const test = await loadJSONL(testPath);

    let correct = 0;

    for (const t of test) {
        // Embed test email text
        const emb = await getEmbedding(t.text);

        // Linear combination: z = w·x + b
        const z = emb.reduce((s, x, i) => s + x * weights[i], bias);

        // Logistic function → probability → binary class
        const pred = 1 / (1 + Math.exp(-z)) >= 0.5 ? 1 : 0;

        // Count correct predictions
        if (pred === t.label) correct++;
    }

    // Final accuracy
    const accuracy = correct / test.length;
    console.log(`Test accuracy = ${accuracy}`);

    return { accuracy };
}

// Allow this file to be run directly with deno run
if (import.meta.main) {
    await evaluateModel();
}
