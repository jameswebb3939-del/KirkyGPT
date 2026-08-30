import type { ChatMessage, ChatResponse } from "../types/chat";

const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL ?? "/api";

export async function sendChat(
  messages: ChatMessage[],
): Promise<string> {
  const response = await fetch(`${API_BASE_URL}/chat`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      messages,
    }),
  });

  if (!response.ok) {
    let detail = `Request failed with status ${response.status}`;

    try {
      const body = await response.json();

      if (body.detail) {
        detail = body.detail;
      }
    } catch {
      // Ignore invalid JSON error response.
    }

    throw new Error(detail);
  }

  const data: ChatResponse = await response.json();

  return data.response_text;
}

export async function getHealth() {
  const response = await fetch(`${API_BASE_URL}/health`);

  if (!response.ok) {
    throw new Error(
      `Health check failed with status ${response.status}`,
    );
  }

  return response.json();
}