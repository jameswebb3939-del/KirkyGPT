import {
  useEffect,
  useMemo,
  useState,
} from "react";

import Header from "./components/Header";
import Sidebar from "./components/Sidebar";
import ChatWindow from "./components/ChatWindow";

import { sendChat } from "./services/api";

import type {
  ChatMessage,
  Conversation,
} from "./types/chat";

const STORAGE_KEY = "ec-pro-conversations";

function createConversation(): Conversation {
  return {
    id: crypto.randomUUID(),
    title: "New chat",
    messages: [],
    updatedAt: Date.now(),
  };
}

function loadConversations(): Conversation[] {
  try {
    const stored = localStorage.getItem(
      STORAGE_KEY,
    );

    if (!stored) {
      return [];
    }

    const parsed = JSON.parse(stored);

    if (!Array.isArray(parsed)) {
      return [];
    }

    return parsed;
  } catch {
    return [];
  }
}

function createTitle(message: string): string {
  const cleaned = message.trim();

  if (cleaned.length <= 35) {
    return cleaned;
  }

  return `${cleaned.slice(0, 35)}...`;
}

export default function App() {
  const [sidebarOpen, setSidebarOpen] =
    useState(true);

  const [
    conversations,
    setConversations,
  ] = useState<Conversation[]>(
    loadConversations,
  );

  const [
    activeConversationId,
    setActiveConversationId,
  ] = useState<string | null>(() => {
    const existing = loadConversations();

    return existing.length > 0
      ? existing[0].id
      : null;
  });

  const [isLoading, setIsLoading] =
    useState(false);

  const [error, setError] =
    useState<string | null>(null);

  /*
   * Save conversations whenever they change.
   */
  useEffect(() => {
    localStorage.setItem(
      STORAGE_KEY,
      JSON.stringify(conversations),
    );
  }, [conversations]);

  /*
   * Find the currently selected conversation.
   */
  const activeConversation = useMemo(
    () =>
      conversations.find(
        (conversation) =>
          conversation.id ===
          activeConversationId,
      ) ?? null,
    [
      conversations,
      activeConversationId,
    ],
  );

  /*
   * Create a new conversation.
   */
  function handleNewChat() {
    const conversation =
      createConversation();

    setConversations((current) => [
      conversation,
      ...current,
    ]);

    setActiveConversationId(
      conversation.id,
    );

    setError(null);
  }

  /*
   * Select an existing conversation.
   */
  function selectConversation(
    id: string,
  ) {
    setActiveConversationId(id);

    setError(null);

    /*
     * On mobile, close the sidebar
     * after selecting a chat.
     */
    if (window.innerWidth <= 768) {
      setSidebarOpen(false);
    }
  }

  /*
   * Delete a conversation.
   */
  function handleDeleteConversation(
    id: string,
  ) {
    setConversations((current) => {
      const remaining =
        current.filter(
          (conversation) =>
            conversation.id !== id,
        );

      /*
       * If the deleted conversation was
       * the active one, select another.
       */
      if (
        activeConversationId === id
      ) {
        setActiveConversationId(
          remaining.length > 0
            ? remaining[0].id
            : null,
        );
      }

      return remaining;
    });

    setError(null);
  }

  /*
   * Send a user message to the backend.
   */
  async function handleSend(
    content: string,
  ) {
    setError(null);

    let conversation =
      activeConversation;

    /*
     * If there isn't currently a chat,
     * automatically create one.
     */
    if (!conversation) {
      conversation =
        createConversation();

      setConversations(
        (current) => [
          conversation!,
          ...current,
        ],
      );

      setActiveConversationId(
        conversation.id,
      );
    }

    const conversationId =
      conversation.id;

    const userMessage: ChatMessage = {
      role: "user",
      content,
    };

    /*
     * Send the complete conversation
     * history to the backend.
     */
    const requestMessages: ChatMessage[] =
      [
        ...conversation.messages,
        userMessage,
      ];

    /*
     * Immediately display the user's
     * message in the UI.
     */
    setConversations(
      (current) =>
        current.map((item) =>
          item.id ===
          conversationId
            ? {
                ...item,

                /*
                 * Use the first message
                 * as the chat title.
                 */
                title:
                  item.messages
                    .length === 0
                    ? createTitle(
                        content,
                      )
                    : item.title,

                messages:
                  requestMessages,

                updatedAt:
                  Date.now(),
              }
            : item,
        ),
    );

    setIsLoading(true);

    try {
      /*
       * Call FastAPI /chat.
       */
      const responseText =
        await sendChat(
          requestMessages,
        );

      const assistantMessage: ChatMessage =
        {
          role: "assistant",
          content:
            responseText,
        };

      /*
       * Add the assistant response
       * to the correct conversation.
       */
      setConversations(
        (current) =>
          current.map((item) =>
            item.id ===
            conversationId
              ? {
                  ...item,

                  messages: [
                    ...item.messages,
                    assistantMessage,
                  ],

                  updatedAt:
                    Date.now(),
                }
              : item,
          ),
      );
    } catch (err) {
      if (err instanceof Error) {
        setError(err.message);
      } else {
        setError(
          "Something went wrong while contacting the server.",
        );
      }
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <div className="app">
      <Header
        onToggleSidebar={() =>
          setSidebarOpen(
            (current) =>
              !current,
          )
        }
      />

      <div className="app-content">
        <Sidebar
          open={sidebarOpen}
          conversations={
            conversations
          }
          activeConversationId={
            activeConversationId
          }
          onSelectConversation={
            selectConversation
          }
          onNewChat={
            handleNewChat
          }
          onDeleteConversation={
            handleDeleteConversation
          }
        />

        <ChatWindow
          messages={
            activeConversation
              ?.messages ?? []
          }
          onSend={
            handleSend
          }
          isLoading={
            isLoading
          }
          error={
            error
          }
        />
      </div>
    </div>
  );
}