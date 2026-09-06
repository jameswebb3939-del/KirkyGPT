import {
  useEffect,
  useRef,
} from "react";

import type { ChatMessage } from "../types/chat";

import ChatInput from "./ChatInput";
import MessageBubble from "./MessageBubble";
import KirkFace from "./KirkFace";

interface ChatWindowProps {
  messages: ChatMessage[];
  onSend: (message: string) => Promise<void>;
  isLoading: boolean;
  error: string | null;
}

export default function ChatWindow({
  messages,
  onSend,
  isLoading,
  error,
}: ChatWindowProps) {
  const bottomRef = useRef<HTMLDivElement | null>(
    null,
  );

  useEffect(() => {
    bottomRef.current?.scrollIntoView({
      behavior: "smooth",
    });
  }, [messages, isLoading]);

  return (
    <main className="chat-window">
      <div className="messages-container">
        {messages.length === 0 && (
          <div className="welcome-screen">
            <KirkFace size="large" />

            <h2>KirkGPT</h2>

            <p>
              Start a conversation below.
            </p>
          </div>
        )}

        {messages.map((message, index) => (
          <MessageBubble
            key={`${index}-${message.role}`}
            message={message}
          />
        ))}

        {isLoading && (
          <div className="message-row message-row-assistant">
            <div className="message-bubble message-assistant typing">
              Generating...
            </div>
          </div>
        )}

        {error && (
          <div className="chat-error">
            {error}
          </div>
        )}

        <div ref={bottomRef} />
      </div>

      <ChatInput
        onSend={onSend}
        disabled={isLoading}
      />
    </main>
  );
}
