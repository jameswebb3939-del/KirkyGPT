import type { Conversation } from "../types/chat";

interface SidebarProps {
  open: boolean;
  conversations: Conversation[];
  activeConversationId: string | null;
  onSelectConversation: (id: string) => void;
  onNewChat: () => void;
  onDeleteConversation: (id: string) => void;
}

export default function Sidebar({
  open,
  conversations,
  activeConversationId,
  onSelectConversation,
  onNewChat,
  onDeleteConversation,
}: SidebarProps) {
  return (
    <aside
      className={`sidebar ${
        open ? "sidebar-open" : "sidebar-closed"
      }`}
    >
      <div className="sidebar-header">
        <h2>History</h2>

        <button
          className="new-chat-button"
          onClick={onNewChat}
        >
          + New chat
        </button>
      </div>

      <div className="conversation-list">
        {conversations.length === 0 && (
          <p className="empty-history">
            No chats yet.
          </p>
        )}

        {[...conversations]
          .sort(
            (a, b) =>
              new Date(b.updated_at).getTime() - new Date(a.updated_at).getTime(),
          )
          .map((conversation) => (
            <div
              key={conversation.id}
              className={
                conversation.id ===
                activeConversationId
                  ? "conversation-row conversation-active"
                  : "conversation-row"
              }
            >
              <button
                className="conversation-item"
                onClick={() =>
                  onSelectConversation(
                    conversation.id,
                  )
                }
              >
                {conversation.title}
              </button>

              <button
                className="delete-chat-button"
                aria-label={`Delete ${conversation.title}`}
                title="Delete chat"
                onClick={(event) => {
                  event.stopPropagation();

                  onDeleteConversation(
                    conversation.id,
                  );
                }}
              >
                ×
              </button>
            </div>
          ))}
      </div>
    </aside>
  );
}