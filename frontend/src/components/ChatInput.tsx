import {
  useState,
  type FormEvent,
  type KeyboardEvent,
} from "react";

interface ChatInputProps {
  onSend: (message: string) => Promise<void>;
  disabled?: boolean;
}

export default function ChatInput({
  onSend,
  disabled = false,
}: ChatInputProps) {
  const [value, setValue] = useState("");

  async function submit() {
    const message = value.trim();

    if (!message || disabled) {
      return;
    }

    setValue("");

    await onSend(message);
  }

  async function handleSubmit(
    event: FormEvent<HTMLFormElement>,
  ) {
    event.preventDefault();
    await submit();
  }

  async function handleKeyDown(
    event: KeyboardEvent<HTMLTextAreaElement>,
  ) {
    if (
      event.key === "Enter" &&
      !event.shiftKey
    ) {
      event.preventDefault();
      await submit();
    }
  }

  return (
    <form
      className="chat-input-container"
      onSubmit={handleSubmit}
    >
      <textarea
        className="chat-input"
        value={value}
        onChange={(event) =>
          setValue(event.target.value)
        }
        onKeyDown={handleKeyDown}
        placeholder="Message EC Pro..."
        rows={1}
        disabled={disabled}
      />

      <button
        className="send-button"
        type="submit"
        disabled={
          disabled ||
          value.trim().length === 0
        }
      >
        {disabled ? "..." : "Send"}
      </button>
    </form>
  );
}