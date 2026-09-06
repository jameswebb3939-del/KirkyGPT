import type {
  Conversation,
  ConversationSummary,
} from "../types/chat";

const STORAGE_KEY =
  "kirk-gpt-conversations-v1";

function readConversations(): Conversation[] {
  const raw =
    window.localStorage.getItem(
      STORAGE_KEY,
    );

  if (!raw) {
    return [];
  }

  try {
    const parsed = JSON.parse(raw);

    if (!Array.isArray(parsed)) {
      return [];
    }

    return parsed as Conversation[];
  } catch {
    return [];
  }
}

function writeConversations(
  conversations: Conversation[],
): void {
  window.localStorage.setItem(
    STORAGE_KEY,
    JSON.stringify(conversations),
  );
}

export async function listStoredConversations():
  Promise<ConversationSummary[]> {
  return readConversations()
    .map(
      ({
        id,
        title,
        created_at,
        updated_at,
      }) => ({
        id,
        title,
        created_at,
        updated_at,
      }),
    )
    .sort(
      (a, b) =>
        new Date(
          b.updated_at,
        ).getTime() -
        new Date(
          a.updated_at,
        ).getTime(),
    );
}

export async function createStoredConversation():
  Promise<Conversation> {
  const now =
    new Date().toISOString();

  const conversation: Conversation = {
    id: crypto.randomUUID(),
    title: "New chat",
    created_at: now,
    updated_at: now,
    messages: [],
  };

  const conversations =
    readConversations();

  writeConversations([
    conversation,
    ...conversations,
  ]);

  return conversation;
}

export async function getStoredConversation(
  conversationId: string,
): Promise<Conversation> {
  const conversation =
    readConversations().find(
      (item) =>
        item.id === conversationId,
    );

  if (!conversation) {
    throw new Error(
      "Conversation not found.",
    );
  }

  return conversation;
}

export async function saveStoredConversation(
  conversation: Conversation,
): Promise<Conversation> {
  const conversations =
    readConversations();

  const exists =
    conversations.some(
      (item) =>
        item.id === conversation.id,
    );

  const updated =
    exists
      ? conversations.map(
          (item) =>
            item.id === conversation.id
              ? conversation
              : item,
        )
      : [
          conversation,
          ...conversations,
        ];

  writeConversations(updated);

  return conversation;
}

export async function deleteStoredConversation(
  conversationId: string,
): Promise<void> {
  const remaining =
    readConversations().filter(
      (conversation) =>
        conversation.id !==
        conversationId,
    );

  writeConversations(remaining);
}

export function clearStoredConversations():
  void {
  window.localStorage.removeItem(
    STORAGE_KEY,
  );
}
