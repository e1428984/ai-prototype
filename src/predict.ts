import ollama from "ollama";

const embedModel = "mxbai-embed-large"; // Embedding model used during both training and inference
const modelPath = "./logs/model.json";

// Convert raw email text into a numerical embedding vector
// This uses a pretrained embedding model (no training here)
export async function getEmbedding(text: string): Promise<number[]> {
    const result = await ollama.embeddings({
        model: embedModel,
        prompt: text,
    });

    // Basic validation to ensure embedding exists
    if (!result.embedding || !Array.isArray(result.embedding)) {
        throw new Error("Embedding model error");
    }
    return result.embedding;
}

/**
 * Predict ham probability using the trained logistic regression model
 *
 * Returns:
 *  number between 0 and 1
 *  values >= 0.5 "ham" (forward)
 *  values < 0.5 "spam" (discard)
 */
export async function predictWithTrainedModel(text: string): Promise<number> {
    // Load trained model parameters from disk
    const model = JSON.parse(await Deno.readTextFile(modelPath));
    const { weights, bias } = model;

    // Generate embedding for the input email
    const emb = await getEmbedding(text);

    // Linear combination: z = w·x + b
    const z = emb.reduce((sum, v, i) => sum + v * weights[i], bias);

    // Sigmoid converts linear score into probability
    const score = 1 / (1 + Math.exp(-z));
    return score;
}
