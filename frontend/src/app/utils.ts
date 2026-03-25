import type {
  AnswerSection,
  AssistantMessage,
  AssistantReplyMessage,
  AssistantUserMessage,
  ManualConstraintFields,
  QueryConstraintsPayload,
} from "./types";

export const EXAMPLE_QUERIES = [
  "We need to renew RDS and transfer ownership. How should this be handled?",
  "Can you confirm the exact new path for my project data?",
  "What is the latest GROMACS version available on CCLake?",
  "How do I check why my Slurm job is stuck in the queue?",
];

export const SECTION_LABELS: Record<string, string> = {
  CLASSIFICATION: "Classification",
  "QUICK ASSESSMENT": "Quick Assessment",
  ACTION: "Action",
  "ACTION STEPS (HELPDESK)": "Action Steps (Helpdesk)",
  "QUESTIONS TO ASK (ONLY IF NEEDED)": "Questions to Ask",
  "EXAMPLE CUSTOMER REPLY": "Example Customer Reply",
  "SAFETY / POLICY NOTES": "Safety / Policy Notes",
  "REFERENCE KEY": "Reference Key",
};

export function quickPromptCards(): string[] {
  return EXAMPLE_QUERIES.slice(0, 2);
}

export function parseAnswerSections(answer: string): AnswerSection[] {
  const text = String(answer ?? "").trim();
  if (!text) {
    return [];
  }

  const sections: AnswerSection[] = [];
  let currentKey: string | null = null;
  let currentLines: string[] = [];

  function flushSection() {
    if (currentKey === null) {
      return;
    }
    sections.push({
      key: currentKey,
      heading: SECTION_LABELS[currentKey] ?? toTitleCase(currentKey),
      body: currentLines.join("\n").trim(),
    });
    currentKey = null;
    currentLines = [];
  }

  for (const rawLine of text.split("\n")) {
    const candidate = rawLine.trim().replace(/:$/, "");
    if (candidate in SECTION_LABELS) {
      flushSection();
      currentKey = candidate;
      continue;
    }
    if (currentKey === null) {
      currentKey = "RESPONSE";
    }
    currentLines.push(rawLine);
  }

  flushSection();
  if (sections.length === 0) {
    return [{ key: "RESPONSE", heading: "Response", body: text }];
  }
  return sections;
}

function toTitleCase(value: string): string {
  return value
    .toLowerCase()
    .split(" ")
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

export function isAssistantReplyMessage(message: AssistantMessage): message is AssistantReplyMessage {
  return message.role === "assistant";
}

export function isAssistantUserMessage(message: AssistantMessage): message is AssistantUserMessage {
  return message.role === "user";
}

export function assistantViewState(messages: AssistantMessage[]): "landing" | "active" {
  return messages.some((message) => message.role === "assistant") ? "active" : "landing";
}

export function latestExchange(messages: AssistantMessage[]): {
  latestUser: AssistantUserMessage | null;
  latestAssistant: AssistantReplyMessage | null;
  olderMessages: AssistantMessage[];
} {
  if (messages.length === 0) {
    return { latestUser: null, latestAssistant: null, olderMessages: [] };
  }

  let latestAssistantIndex = -1;
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    if (messages[index]?.role === "assistant") {
      latestAssistantIndex = index;
      break;
    }
  }

  if (latestAssistantIndex < 0) {
    let latestUserIndex = -1;
    for (let index = messages.length - 1; index >= 0; index -= 1) {
      if (messages[index]?.role === "user") {
        latestUserIndex = index;
        break;
      }
    }
    return {
      latestUser: latestUserIndex >= 0 ? (messages[latestUserIndex] as AssistantUserMessage) : null,
      latestAssistant: null,
      olderMessages: latestUserIndex >= 0 ? messages.slice(0, latestUserIndex) : [...messages],
    };
  }

  let latestUserIndex = -1;
  for (let index = latestAssistantIndex - 1; index >= 0; index -= 1) {
    if (messages[index]?.role === "user") {
      latestUserIndex = index;
      break;
    }
  }

  return {
    latestUser: latestUserIndex >= 0 ? (messages[latestUserIndex] as AssistantUserMessage) : null,
    latestAssistant: messages[latestAssistantIndex] as AssistantReplyMessage,
    olderMessages:
      latestUserIndex >= 0 ? messages.slice(0, latestUserIndex) : messages.slice(0, latestAssistantIndex),
  };
}

export function formatTimingValue(value: number | null | undefined): string {
  if (typeof value === "number" && Number.isFinite(value)) {
    return String(value);
  }
  return "n/a";
}

export function createMessageId(): string {
  return `${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
}

export function nowIso(): string {
  return new Date().toISOString();
}

export function toCsvList(rawValue: string): string[] {
  return rawValue
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean);
}

export function buildManualQueryConstraints(
  fields: ManualConstraintFields,
): QueryConstraintsPayload | undefined {
  const payload: QueryConstraintsPayload = {};
  if (fields.queryType && fields.queryType !== "auto") {
    payload.query_type = fields.queryType;
  }

  const scopeFamilyNames = toCsvList(fields.scopeFamilyNames);
  const serviceNames = toCsvList(fields.serviceNames);
  const softwareNames = toCsvList(fields.softwareNames);
  const softwareVersions = toCsvList(fields.softwareVersions);

  if (scopeFamilyNames.length > 0) {
    payload.scope_family_names = scopeFamilyNames;
  }
  if (serviceNames.length > 0) {
    payload.service_names = serviceNames;
  }
  if (softwareNames.length > 0) {
    payload.software_names = softwareNames;
  }
  if (softwareVersions.length > 0) {
    payload.software_versions = softwareVersions;
  }

  return Object.keys(payload).length > 0 ? payload : undefined;
}

export function mergeQueryConstraints(
  base: QueryConstraintsPayload | undefined,
  overlay: QueryConstraintsPayload | null | undefined,
): QueryConstraintsPayload | undefined {
  const merged: QueryConstraintsPayload = {};
  const listKeys: Array<keyof QueryConstraintsPayload> = [
    "system_names",
    "partition_names",
    "service_names",
    "scope_family_names",
    "software_names",
    "software_versions",
    "module_names",
    "toolchain_names",
    "toolchain_versions",
  ];
  const scalarKeys: Array<keyof QueryConstraintsPayload> = [
    "query_type",
    "scope_required",
    "version_sensitive_guess",
  ];

  for (const key of listKeys) {
    const baseList = Array.isArray(base?.[key]) ? (base?.[key] as string[]) : [];
    const overlayList = Array.isArray(overlay?.[key]) ? (overlay?.[key] as string[]) : [];
    const mergedList = [...new Set([...baseList, ...overlayList])];
    if (mergedList.length > 0) {
      merged[key] = mergedList as never;
    }
  }

  for (const key of scalarKeys) {
    const overlayValue = overlay?.[key];
    const baseValue = base?.[key];
    if (overlayValue !== undefined && overlayValue !== null && overlayValue !== "") {
      merged[key] = overlayValue as never;
    } else if (baseValue !== undefined && baseValue !== null && baseValue !== "") {
      merged[key] = baseValue as never;
    }
  }

  return Object.keys(merged).length > 0 ? merged : undefined;
}

export function joinUrl(baseUrl: string, path: string): string {
  const base = String(baseUrl ?? "").replace(/\/+$/, "");
  const suffix = path.startsWith("/") ? path : `/${path}`;
  return `${base}${suffix}`;
}

export async function computeResponseFingerprint(
  query: string,
  answer: string,
  options: { contextDocIds?: string[]; scenarioId?: string | null } = {},
): Promise<string> {
  const payload = [query.trim(), answer.trim(), (options.contextDocIds ?? []).join(","), options.scenarioId ?? ""].join(
    "\n",
  );
  const data = new TextEncoder().encode(payload);
  const digest = await crypto.subtle.digest("SHA-256", data);
  return Array.from(new Uint8Array(digest))
    .map((value) => value.toString(16).padStart(2, "0"))
    .join("");
}
