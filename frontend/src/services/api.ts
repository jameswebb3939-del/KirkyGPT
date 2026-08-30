import type {
  Conversation,
  ConversationSummary,
} from "../types/chat";

const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL ??
  "/api";

async function apiRequest<T>(
  path: string,
  options?: RequestInit,
): Promise<T> {
  const response = await fetch(
    `${API_BASE_URL}${path}`,
    options,
  );

  if (!response.ok) {
    let detail =
      `Request failed with status ${response.status}`;

    try {
      const body =
        await response.json();

      if (
        typeof body.detail ===
        "string"
      ) {
        detail = body.detail;
      }
    } catch {
      // Ignore malformed/non-JSON body.
    }

    throw new Error(detail);
  }

  if (response.status === 204) {
    return undefined as T;
  }

  return response.json();
}

export async function listConversations():
  Promise<ConversationSummary[]> {
  return apiRequest<
    ConversationSummary[]
  >("/conversations");
}

export async function createConversation():
  Promise<Conversation> {
  return apiRequest<Conversation>(
    "/conversations",
    {
      method: "POST",

      headers: {
        "Content-Type":
          "application/json",
      },

      body: JSON.stringify({
        title: "New chat",
      }),
    },
  );
}

export async function getConversation(
  conversationId: string,
): Promise<Conversation> {
  return apiRequest<Conversation>(
    `/conversations/${conversationId}`,
  );
}

export async function deleteConversation(
  conversationId: string,
): Promise<void> {
  await apiRequest<void>(
    `/conversations/${conversationId}`,
    {
      method: "DELETE",
    },
  );
}

export async function sendMessage(
  conversationId: string,
  content: string,
): Promise<Conversation> {
  return apiRequest<Conversation>(
    `/conversations/${conversationId}/chat`,
    {
      method: "POST",

      headers: {
        "Content-Type":
          "application/json",
      },

      body: JSON.stringify({
        content,
      }),
    },
  );
}

export async function getHealth() {
  return apiRequest("/health");
}