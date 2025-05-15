import { readFileSync } from "fs";
import * as core from "@actions/core";
import OpenAI from "openai";
import { Octokit } from "@octokit/rest";
import parseDiff, { Chunk, File } from "parse-diff";
import minimatch from "minimatch";

const GITHUB_TOKEN: string = core.getInput("GITHUB_TOKEN");
const OPENAI_API_KEY: string = core.getInput("OPENAI_API_KEY");
const OPENAI_API_MODEL: string = core.getInput("OPENAI_API_MODEL");
const MAX_COMMENTS_PER_PR: number = parseInt(core.getInput("MAX_COMMENTS_PER_PR") || "10", 10);

const octokit = new Octokit({ auth: GITHUB_TOKEN });

const openai = new OpenAI({
  apiKey: OPENAI_API_KEY,
});

interface PRDetails {
  owner: string;
  repo: string;
  pull_number: number;
  title: string;
  description: string;
}

async function getPRDetails(): Promise<PRDetails> {
  const { repository, number } = JSON.parse(
    readFileSync(process.env.GITHUB_EVENT_PATH || "", "utf8")
  );
  const prResponse = await octokit.pulls.get({
    owner: repository.owner.login,
    repo: repository.name,
    pull_number: number,
  });
  return {
    owner: repository.owner.login,
    repo: repository.name,
    pull_number: number,
    title: prResponse.data.title ?? "",
    description: prResponse.data.body ?? "",
  };
}

async function getDiff(
  owner: string,
  repo: string,
  pull_number: number
): Promise<string | null> {
  const response = await octokit.pulls.get({
    owner,
    repo,
    pull_number,
    mediaType: { format: "diff" },
  });
  // @ts-expect-error - response.data is a string
  return response.data;
}

async function analyzeCode(
  parsedDiff: File[],
  prDetails: PRDetails
): Promise<Array<{ body: string; path: string; line: number }>> {
  const comments: Array<{ body: string; path: string; line: number }> = [];
  const maxChangesPerChunk = 100; // Limit to prevent too large prompts

  for (const file of parsedDiff) {
    if (file.to === "/dev/null") continue; // Ignore deleted files
    
    // Group chunks by file to provide better context
    if (file.chunks.length <= 3) {
      // For files with few chunks, analyze the entire file at once
      const combinedChunk = {
        content: file.chunks.map(c => c.content).join("\n"),
        changes: file.chunks.flatMap(c => c.changes)
      };
      
      const prompt = createPrompt(file, combinedChunk, prDetails);
      const aiResponse = await getAIResponse(prompt);
      if (aiResponse) {
        const newComments = createComment(file, combinedChunk, aiResponse);
        if (newComments) {
          comments.push(...newComments);
        }
      }
    } else {
      // For larger files, process chunks individually but with a limit on changes
      for (let i = 0; i < file.chunks.length; i++) {
        const chunk = file.chunks[i];
        
        // Skip chunks that are too small (likely minor changes)
        if (chunk.changes.length < 3) {
          continue;
        }
        
        // For chunks with too many changes, we might want to split them
        if (chunk.changes.length > maxChangesPerChunk) {
          // Process the chunk as is, but the prompt will instruct to focus only on significant issues
          const prompt = createPrompt(file, chunk, prDetails);
          const aiResponse = await getAIResponse(prompt);
          if (aiResponse) {
            const newComments = createComment(file, chunk, aiResponse);
            if (newComments) {
              comments.push(...newComments);
            }
          }
        } else {
          // Try to combine with next chunk if they're close together
          if (i < file.chunks.length - 1) {
            const nextChunk = file.chunks[i + 1];
            const combinedChanges = [...chunk.changes, ...nextChunk.changes];
            
            if (combinedChanges.length <= maxChangesPerChunk) {
              // Combine chunks for better context
              const combinedChunk = {
                content: chunk.content + "\n" + nextChunk.content,
                changes: combinedChanges
              };
              
              const prompt = createPrompt(file, combinedChunk, prDetails);
              const aiResponse = await getAIResponse(prompt);
              if (aiResponse) {
                const newComments = createComment(file, combinedChunk, aiResponse);
                if (newComments) {
                  comments.push(...newComments);
                }
              }
              
              // Skip the next chunk since we've already processed it
              i++;
              continue;
            }
          }
          
          // Process the chunk individually
          const prompt = createPrompt(file, chunk, prDetails);
          const aiResponse = await getAIResponse(prompt);
          if (aiResponse) {
            const newComments = createComment(file, chunk, aiResponse);
            if (newComments) {
              comments.push(...newComments);
            }
          }
        }
      }
    }
  }
  
  // Limit the total number of comments to prevent overwhelming the PR
  return comments.slice(0, MAX_COMMENTS_PER_PR);
}

function createPrompt(file: File, chunk: Chunk | { content: string; changes: any[] }, prDetails: PRDetails): string {
  return `Your task is to review pull requests. Instructions:
- Provide the response in following JSON format:  {"reviews": [{"lineNumber":  <line_number>, "reviewComment": "<review comment>"}]}
- Do not give positive comments or compliments.
- Focus ONLY on significant functional issues, potential bugs, security concerns, and important performance problems.
- IGNORE minor stylistic issues like:
  * Variable naming (unless extremely misleading)
  * Indentation or whitespace
  * Missing newlines
  * Comment formatting or wording
  * Other cosmetic issues
- Provide comments ONLY for issues that could impact functionality, security, or performance.
- If no significant issues are found, "reviews" should be an empty array.
- Write the comment in GitHub Markdown format.
- Use the given description only for the overall context and only comment the code.
- IMPORTANT: NEVER suggest adding comments to the code.
- Limit to at most 2-3 high-value comments per chunk.

Review the following code diff in the file "${
    file.to
  }" and take the pull request title and description into account when writing the response.
  
Pull request title: ${prDetails.title}
Pull request description:

---
${prDetails.description}
---

Git diff to review:

\`\`\`diff
${chunk.content}
${chunk.changes
  .map((c) => `${c.ln ? c.ln : c.ln2} ${c.content}`)
  .join("\n")}
\`\`\`
`;
}

async function getAIResponse(prompt: string): Promise<Array<{
  lineNumber: string;
  reviewComment: string;
}> | null> {
  const queryConfig = {
    model: OPENAI_API_MODEL,
    temperature: 0.4,  // Increased to make the model less pedantic
    max_tokens: 700,
    top_p: 1,
    frequency_penalty: 0.2,  // Slight increase to reduce repetitive comments
    presence_penalty: 0.2,   // Slight increase to encourage more diverse thinking
  };

  try {
    const response = await openai.chat.completions.create({
      ...queryConfig,
      // return JSON if the model supports it:
      ...(OPENAI_API_MODEL === "gpt-4-1106-preview"
        ? { response_format: { type: "json_object" } }
        : {}),
      messages: [
        {
          role: "system",
          content: prompt,
        },
      ],
    });

    const res = response.choices[0].message?.content?.trim() || "{}";
    return JSON.parse(res).reviews;
  } catch (error) {
    console.error("Error:", error);
    return null;
  }
}

function createComment(
  file: File,
  chunk: Chunk | { content: string; changes: any[] },
  aiResponses: Array<{
    lineNumber: string;
    reviewComment: string;
  }>
): Array<{ body: string; path: string; line: number }> {
  return aiResponses.flatMap((aiResponse) => {
    if (!file.to) {
      return [];
    }
    
    // Validate that the line number is within the range of the file
    const lineNumber = Number(aiResponse.lineNumber);
    if (isNaN(lineNumber) || lineNumber <= 0) {
      console.log(`Skipping comment with invalid line number: ${aiResponse.lineNumber}`);
      return [];
    }
    
    // Check if the comment is too short (likely not meaningful)
    if (aiResponse.reviewComment.length < 20) {
      console.log(`Skipping too short comment: ${aiResponse.reviewComment}`);
      return [];
    }
    
    return {
      body: aiResponse.reviewComment,
      path: file.to,
      line: lineNumber,
    };
  });
}

async function createReviewComment(
  owner: string,
  repo: string,
  pull_number: number,
  comments: Array<{ body: string; path: string; line: number }>
): Promise<void> {
  await octokit.pulls.createReview({
    owner,
    repo,
    pull_number,
    comments,
    event: "COMMENT",
  });
}

async function main() {
  const prDetails = await getPRDetails();
  let diff: string | null;
  const eventData = JSON.parse(
    readFileSync(process.env.GITHUB_EVENT_PATH ?? "", "utf8")
  );

  console.log("Pull request event type:", eventData.action);

  if (eventData.action === "synchronize") {
    const newBaseSha = eventData.before;
    const newHeadSha = eventData.after;

    const response = await octokit.repos.compareCommits({
      headers: {
        accept: "application/vnd.github.v3.diff",
      },
      owner: prDetails.owner,
      repo: prDetails.repo,
      base: newBaseSha,
      head: newHeadSha,
    });

    diff = String(response.data);
  } else {
    diff = await getDiff(
      prDetails.owner,
      prDetails.repo,
      prDetails.pull_number
    );
  }

  if (!diff) {
    console.log("No diff found");
    return;
  }

  const parsedDiff = parseDiff(diff);

  const excludePatterns = core
    .getInput("exclude")
    .split(",")
    .map((s) => s.trim());

  const filteredDiff = parsedDiff.filter((file) => {
    return !excludePatterns.some((pattern) =>
      minimatch(file.to ?? "", pattern)
    );
  });

  const comments = await analyzeCode(filteredDiff, prDetails);
  if (comments.length > 0) {
    await createReviewComment(
      prDetails.owner,
      prDetails.repo,
      prDetails.pull_number,
      comments
    );
  }
}

main().catch((error) => {
  console.error("Error:", error);
  process.exit(1);
});
