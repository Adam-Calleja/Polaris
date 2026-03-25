import type { ReactNode } from "react";

export function ExpandablePanel({
  title,
  children,
  defaultOpen = false,
}: {
  title: string;
  children: ReactNode;
  defaultOpen?: boolean;
}) {
  return (
    <details className="expandable-panel" open={defaultOpen}>
      <summary className="expandable-panel__summary">{title}</summary>
      <div className="expandable-panel__body">{children}</div>
    </details>
  );
}
