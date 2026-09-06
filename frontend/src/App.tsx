import {
  useEffect,
  useMemo,
  useState,
} from "react";

import Header from "./components/Header";
import Sidebar from "./components/Sidebar";
import ChatWindow from "./components/ChatWindow";

import {
  createStoredConversation,
  deleteStoredConversation,
  getStoredConversation,
  listStoredConversations,
  saveStoredConversation,
} from "./services/browserStorage";

import {
  browserRuleEngine,
} from "./rules/runtime";

import type {
  Conversation,
  StoredMessage,
} from "./types/chat";

export default function App() {
  const [sidebarOpen, setSidebarOpen] =
    useState(true);

  const [
    conversations,
    setConversations,
  ] = useState<Conversation[]>([]);

  const [
    activeConversationId,
    setActiveConversationId,
  ] = useState<string | null>(
    null,
  );

  const [isLoading, setIsLoading] =
    useState(false);

  const [error, setError] =
    useState<string | null>(null);

  const activeConversation =
    useMemo(
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

  useEffect(() => {
    async function loadHistory() {
      try {
        const summaries =
          await listStoredConversations();

        const loaded =
          await Promise.all(
            summaries.map(
              (summary) =>
                getStoredConversation(
                  summary.id,
                ),
            ),
          );

        setConversations(loaded);

        if (loaded.length > 0) {
          setActiveConversationId(
            loaded[0].id,
          );
        }
      } catch {
        setError(
          "Could not load chat history.",
        );
      }
    }

    void loadHistory();
  }, []);

  async function handleNewChat() {
    setError(null);

    try {
      const conversation =
        await createStoredConversation();

      setConversations(
        (current) => [
          conversation,
          ...current,
        ],
      );

      setActiveConversationId(
        conversation.id,
      );
    } catch {
      setError(
        "Could not create conversation.",
      );
    }
  }

  async function selectConversation(
    id: string,
  ) {
    setActiveConversationId(id);

    if (
      window.innerWidth <= 768
    ) {
      setSidebarOpen(false);
    }
  }

  async function handleDeleteConversation(
    id: string,
  ) {
    setError(null);

    try {
      await deleteStoredConversation(id);

      setConversations(
        (current) => {
          const remaining =
            current.filter(
              (conversation) =>
                conversation.id !== id,
            );

          if (
            activeConversationId === id
          ) {
            setActiveConversationId(
              remaining[0]?.id ?? null,
            );
          }

          return remaining;
        },
      );
    } catch {
      setError(
        "Could not delete conversation.",
      );
    }
  }

  async function handleSend(
    content: string,
  ) {
    setError(null);
    setIsLoading(true);

    try {
      let conversation =
        activeConversation;

      if (!conversation) {
        conversation =
          await createStoredConversation();

        setActiveConversationId(
          conversation.id,
        );
      }

      const now =
        new Date().toISOString();

      const userMessage:
        StoredMessage = {
          id: crypto.randomUUID(),
          role: "user",
          content,
          position:
            conversation.messages.length,
          created_at: now,
        };

      const messagesWithUser = [
        ...conversation.messages,
        userMessage,
      ];

      const assistantContent =
        browserRuleEngine.respond(
          messagesWithUser,
        );

      const assistantMessage:
        StoredMessage = {
          id: crypto.randomUUID(),
          role: "assistant",
          content:
            assistantContent,
          position:
            messagesWithUser.length,
          created_at:
            new Date().toISOString(),
        };

      const firstUserMessage =
        conversation.messages.length === 0;

      const updated:
        Conversation = {
          ...conversation,

          title:
            firstUserMessage
              ? content
                  .trim()
                  .slice(0, 48) ||
                "New chat"
              : conversation.title,

          updated_at:
            new Date().toISOString(),

          messages: [
            ...messagesWithUser,
            assistantMessage,
          ],
        };

      await saveStoredConversation(
        updated,
      );

      setConversations(
        (current) => {
          const withoutUpdated =
            current.filter(
              (item) =>
                item.id !== updated.id,
            );

          return [
            updated,
            ...withoutUpdated,
          ];
        },
      );
    } catch (err) {
      if (err instanceof Error) {
        setError(err.message);
      } else {
        setError(
          "Something went wrong.",
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
