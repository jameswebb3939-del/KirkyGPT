import {
  useEffect,
  useMemo,
  useState,
} from "react";

import Header from "./components/Header";
import Sidebar from "./components/Sidebar";
import ChatWindow from "./components/ChatWindow";

import {
  createConversation,
  deleteConversation,
  getConversation,
  listConversations,
  sendMessage,
} from "./services/api";

import type {
  ChatMessage,
  Conversation,
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

  /*
   * Load conversation history
   * from SQLite when EC Pro starts.
   */
  useEffect(() => {
    async function loadHistory() {
      setError(null);

      try {
        const summaries =
          await listConversations();

        const loaded =
          await Promise.all(
            summaries.map(
              (summary) =>
                getConversation(
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
      } catch (err) {
        if (err instanceof Error) {
          setError(err.message);
        } else {
          setError(
            "Could not load chat history.",
          );
        }
      }
    }

    void loadHistory();
  }, []);

  async function handleNewChat() {
    setError(null);

    try {
      const conversation =
        await createConversation();

      setConversations(
        (current) => [
          conversation,
          ...current,
        ],
      );

      setActiveConversationId(
        conversation.id,
      );
    } catch (err) {
      if (err instanceof Error) {
        setError(err.message);
      } else {
        setError(
          "Could not create conversation.",
        );
      }
    }
  }

  async function selectConversation(
    id: string,
  ) {
    setError(null);

    try {
      const conversation =
        await getConversation(id);

      setConversations(
        (current) => {
          const exists =
            current.some(
              (item) =>
                item.id === id,
            );

          if (!exists) {
            return [
              conversation,
              ...current,
            ];
          }

          return current.map(
            (item) =>
              item.id === id
                ? conversation
                : item,
          );
        },
      );

      setActiveConversationId(id);

      if (
        window.innerWidth <= 768
      ) {
        setSidebarOpen(false);
      }
    } catch (err) {
      if (err instanceof Error) {
        setError(err.message);
      } else {
        setError(
          "Could not load conversation.",
        );
      }
    }
  }

  async function handleDeleteConversation(
    id: string,
  ) {
    setError(null);

    try {
      await deleteConversation(id);

      setConversations(
        (current) => {
          const remaining =
            current.filter(
              (conversation) =>
                conversation.id !== id,
            );

          if (
            activeConversationId ===
            id
          ) {
            setActiveConversationId(
              remaining.length > 0
                ? remaining[0].id
                : null,
            );
          }

          return remaining;
        },
      );
    } catch (err) {
      if (err instanceof Error) {
        setError(err.message);
      } else {
        setError(
          "Could not delete conversation.",
        );
      }
    }
  }

  async function handleSend(
    content: string,
  ) {
    setError(null);

    let conversation =
      activeConversation;

    /*
     * Sending from the blank landing
     * page automatically creates a chat.
     */
    if (!conversation) {
      try {
        conversation =
          await createConversation();

        setConversations(
          (current) => [
            conversation!,
            ...current,
          ],
        );

        setActiveConversationId(
          conversation.id,
        );
      } catch (err) {
        if (err instanceof Error) {
          setError(err.message);
        }

        return;
      }
    }

    const conversationId =
      conversation.id;

    /*
     * Optimistically display the user's
     * message while Llama generates.
     */
    const optimisticUser:
      ChatMessage = {
        role: "user",
        content,
      };

    setConversations(
      (current) =>
        current.map(
          (item) =>
            item.id ===
            conversationId
              ? {
                  ...item,

                  messages: [
                    ...item.messages,

                    {
                      id:
                        `pending-${Date.now()}`,

                      role:
                        optimisticUser.role,

                      content:
                        optimisticUser.content,

                      position:
                        item.messages
                          .length,

                      created_at:
                        new Date()
                          .toISOString(),
                    },
                  ],
                }
              : item,
        ),
    );

    setIsLoading(true);

    try {
      /*
       * Backend:
       *
       * history → model
       *         → transaction
       *         → SQLite
       */
      const updated =
        await sendMessage(
          conversationId,
          content,
        );

      /*
       * Replace optimistic state with
       * the authoritative SQLite data.
       */
      setConversations(
        (current) =>
          current.map(
            (item) =>
              item.id ===
              conversationId
                ? updated
                : item,
          ),
      );
    } catch (err) {
      /*
       * Reload from SQLite so failed
       * optimistic messages disappear.
       */
      try {
        const authoritative =
          await getConversation(
            conversationId,
          );

        setConversations(
          (current) =>
            current.map(
              (item) =>
                item.id ===
                conversationId
                  ? authoritative
                  : item,
            ),
        );
      } catch {
        // Keep original error.
      }

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