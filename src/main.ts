// src/main.ts
import { classifyEmail } from "./model.ts";

const emailsFolder = "./emails";
const logFile = "./logs/results.json";

// Ensure logs folder exists
await Deno.mkdir("./logs", { recursive: true });

const results: Array<{ email: string; classification: any }> = [];

for await (const entry of Deno.readDir(emailsFolder)) {
    if (entry.isFile && entry.name.endsWith(".txt")) {
        const emailText = await Deno.readTextFile(`${emailsFolder}/${entry.name}`);
        const classification = await classifyEmail(emailText);

        console.log(`Email: ${entry.name}`);
        console.log(classification, "\n");

        results.push({
            email: entry.name,
            classification: JSON.parse(classification),
        });
    }
}

// Save results to JSON file
await Deno.writeTextFile(logFile, JSON.stringify(results, null, 2));
console.log(`Results saved to ${logFile}`);