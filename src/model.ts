// src/model.ts

/**
 * Calls Ollama CLI to classify an email as spam or not.
 * @param emailText The content of the email
 * @returns JSON string with spam_score, reasoning, and recommendation
 */
export async function classifyEmail(emailText: string): Promise<string> {
    const prompt = `
Classify this email as spam or not spam.
Return a JSON object with:
- spam_score (0-1),
- reasoning (why it's spam or not),
- recommendation ("forward" or "discard").
Email content: """${emailText}"""
`;

    const process = Deno.run({
        cmd: ["ollama", "run", "rosemarla/qwen3-classify", "--prompt", prompt],
        stdout: "piped",
        stderr: "piped",
    });

    const output = await process.output();
    const error = await process.stderrOutput();
    process.close();

    if (error.length) {
        console.error("Ollama error:", new TextDecoder().decode(error));
        return JSON.stringify({
            spam_score: 0,
            reasoning: "Error occurred",
            recommendation: "discard",
        });
    }

    return new TextDecoder().decode(output).trim();
}