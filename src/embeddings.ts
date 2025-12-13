// Converting raw email text into numeric vectors using a pretrained embedding model
// These vectors are later used by the logistic regression model
import ollama from "ollama";

const embedModel = "mxbai-embed-large"; // Model for training and inference

export async function getEmbedding(text: string): Promise<number[]> {
    const result = await ollama.embeddings({
        model: embedModel,
        prompt: text,
    });

    if (!Array.isArray(result.embedding)) {
        throw new Error("Invalid embedding result");
    }

    return result.embedding;
}
