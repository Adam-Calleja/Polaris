import { useState } from "react";
import type { FormEvent } from "react";

export function PromptComposer({
  placeholder,
  submitLabel = "➜",
  disabled = false,
  onSubmit,
}: {
  placeholder: string;
  submitLabel?: string;
  disabled?: boolean;
  onSubmit: (prompt: string) => boolean | Promise<boolean>;
}) {
  const [value, setValue] = useState("");

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const prompt = value.trim();
    if (!prompt || disabled) {
      return;
    }
    const handled = await onSubmit(prompt);
    if (handled) {
      setValue("");
    }
  }

  return (
    <form className="prompt-composer" onSubmit={handleSubmit}>
      <input
        aria-label="Prompt"
        className="prompt-composer__input"
        disabled={disabled}
        onChange={(event) => setValue(event.target.value)}
        placeholder={placeholder}
        value={value}
      />
      <button className="prompt-composer__button" disabled={disabled} type="submit">
        {submitLabel}
      </button>
    </form>
  );
}
