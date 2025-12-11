// src/model.ts
import ollama from "ollama";

const targetModel = "rosemarla/qwen3-classify";

/**
 * Classify a single email using the Ollama model
 * @param emailText - The content of the email
 * @returns object with spam_score, reasoning, recommendation
 */
export async function classifyEmail(emailText: string): Promise<{
  spam_score: number;
  reasoning: string;
  recommendation: "forward" | "discard";
}> {
  const prompt = `
Classify this email as spam or not spam.
Return a JSON object with:
- spam_score (0-1),
- reasoning (why it's spam or not),
- recommendation ("forward" or "discard").
Email content: """${emailText}"""
`;
  const res = await ollama.pull({ model: targetModel });
  console.log(`Model pulled.`, res);

  const message: Message = { role: "user", content: prompt };
  const response = await ollama.chat({
    model: targetModel,
    messages: [message],
    stream: true,
  });

  const textEncoder = new TextEncoder();
  let full = "";
  for await (const chunk of response) {
    await Deno.stdout.write(textEncoder.encode(chunk.message?.content ?? ""));
    full += chunk.message?.content ?? "";
  }

  // Manual parsing of streamed output
  // Extract reasoning between <think>...</think>
  const thinkMatch = full.match(/<think>([\s\S]*?)<\/think>/);
  const reasoning = thinkMatch ? thinkMatch[1].trim() : "No reasoning provided";

  // Extract final numeric answer (0 or 1)
  const scoreMatch = full.match(/([01])\s*$/m);
  const spam_score = scoreMatch ? Number(scoreMatch[1]) : 0;

  // Recommendation based on score (since Ollama didn't output it)
  const recommendation = spam_score === 1 ? "forward" : "discard";

  return {
    spam_score,
    reasoning,
    recommendation,
  };
}
