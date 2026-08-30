import type { ChatMessage } from "../types/chat";

interface MessageBubbleProps {
  message: ChatMessage;
}

export default function MessageBubble({
  message,
}: MessageBubbleProps) {
  return (
    <div
      className={`message-row ${
        message.role === "user"
          ? "message-row-user"
          : "message-row-assistant"
      }`}
    >
      <div
        className={`message-bubble ${
          message.role === "user"
            ? "message-user"
            : "message-assistant"
        }`}
      >
        {message.content}
      </div>
    </div>
  );
}