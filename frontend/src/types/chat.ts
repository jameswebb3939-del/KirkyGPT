export type MessageRole =
  | "user"
  | "assistant";

export interface ChatMessage {
  role: MessageRole;
  content: string;
}

export interface StoredMessage
  extends ChatMessage {
  id: string;
  position: number;
  created_at: string;
}

export interface ConversationSummary {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
}

export interface Conversation
  extends ConversationSummary {
  messages: StoredMessage[];
}