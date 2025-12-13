import { predictWithTrainedModel } from "./predict.ts";
import ollama from "ollama";
import { ensureDir } from "https://deno.land/std/fs/mod.ts";

const targetModel = "qwen2.5:1.5b"; // reasoning-only LLM, not classification
const emailsFolder = "./emails";
const logsFolder = "./logs";
const outFile = `${logsFolder}/aggregation.json`;

/**
 * Build a prompt that asks the LLM to explain a decision
 * because the decision itself is already made by the trained model
 *
 * The LLM does not decide spam/ham
 * It only explains the final outcome
 */
function reasoningPrompt(emailText: string, finalDecision: string) {
    return `
You are a spam classification assistant.

Explain in ONE short sentence why the following email was classified as "${finalDecision}".

Email:
"""${emailText}"""
`;
}

/**
 * Ask the LLM for a short explanation of the final decision
 * Temperature is set to 0 to keep explanations deterministic
 */
async function getReasoning(emailText: string, finalDecision: string): Promise<string> {
    const response = await ollama.chat({
        model: targetModel,
        messages: [{ role: "user", content: reasoningPrompt(emailText, finalDecision) }],
        stream: false,
        options: { temperature: 0 },
    });

    return response.message?.content?.trim() || "No reasoning provided.";
}

/**
 * Aggregate multiple predictions from the trained model.
 * Steps:
 * 1. Run the trained classifier N times
 * 2. Perform majority voting
 * 3. Ask the LLM to explain the FINAL decision
 */
export async function aggregateEmail(emailText: string, n: number) {
    const scores: number[] = [];

    // N independent predictions from trained model
    for (let i = 0; i < n; i++) {
        const score = await predictWithTrainedModel(emailText);
        scores.push(score);
    }

    // Count how many predictions see it as ham (ham >= 0.5)
    const hamVotes = scores.filter((s) => s >= 0.5).length;
    // Majority vote determines the final action
    const finalDecision = hamVotes > n / 2 ? "forward" : "discard";

    // Generate one explanation after the decision is finalized
    const reasoning = await getReasoning(emailText, finalDecision);

    return {
        final_recommendation: finalDecision,
        reasoning,
        scores, // raw model outputs
    };
}

/**
 * Run aggregation for all emails in the input folder
 * Results are written to logs/aggregation.json
 */
export async function runAggregation(n = 5) {
    // Ensure output directory exists
    await ensureDir(logsFolder);

    const results = [];

    // Process each email file
    for await (const entry of Deno.readDir(emailsFolder)) {
        if (!entry.isFile || !entry.name.endsWith(".txt")) continue;

        const emailText = await Deno.readTextFile(`${emailsFolder}/${entry.name}`);
        const agg = await aggregateEmail(emailText, n);

        results.push({
            email: entry.name,
            final: agg.final_recommendation,
            reasoning: agg.reasoning,
            scores: agg.scores,
        });

        console.log(`Aggregated ${entry.name} → ${agg.final_recommendation}`);
    }

    // Persist aggregated results
    await Deno.writeTextFile(outFile, JSON.stringify(results, null, 2));
    console.log(`Aggregation saved → ${outFile}`);
}

/**
 * Allow this file to be run directly:
 *   deno run src/aggregate.ts 5
 */
if (import.meta.main) {
    const n = Number(Deno.args[0] ?? 5);
    await runAggregation(n);
}
