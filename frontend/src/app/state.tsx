import { createContext, useContext, useEffect, useReducer } from "react";
import type { Dispatch, ReactNode } from "react";
import type {
  AssistantMessage,
  AssistantReplyMessage,
  EvaluationScenarioResult,
  FrontendRuntimeConfig,
  ManualConstraintFields,
  Workspace,
} from "./types";

const SESSION_STORAGE_KEY = "polaris-ui-state-v1";

const DEFAULT_MANUAL_CONSTRAINTS: ManualConstraintFields = {
  queryType: "auto",
  scopeFamilyNames: "",
  serviceNames: "",
  softwareNames: "",
  softwareVersions: "",
};

export interface AppState {
  currentWorkspace: Workspace;
  drawerOpen: boolean;
  apiBaseUrl: string;
  apiEndpointPath: string;
  timeoutS: number;
  debugMode: boolean;
  manualConstraintsOpen: boolean;
  manualConstraints: ManualConstraintFields;
  assistantMessages: AssistantMessage[];
  assistantHistoryOpen: boolean;
  evaluationResults: Record<string, EvaluationScenarioResult>;
  feedbackSubmissionIds: string[];
  displayName: string;
}

type Action =
  | { type: "set-workspace"; workspace: Workspace }
  | { type: "toggle-drawer" }
  | { type: "set-drawer-open"; open: boolean }
  | { type: "set-api-base-url"; value: string }
  | { type: "set-api-endpoint-path"; value: string }
  | { type: "increment-timeout"; delta: number }
  | { type: "set-debug-mode"; value: boolean }
  | { type: "toggle-manual-constraints" }
  | { type: "set-manual-constraint"; field: keyof ManualConstraintFields; value: string }
  | { type: "append-assistant-message"; message: AssistantMessage }
  | { type: "replace-assistant-message"; messageId: string; message: AssistantReplyMessage }
  | { type: "clear-assistant-session" }
  | { type: "toggle-assistant-history" }
  | { type: "set-evaluation-result"; result: EvaluationScenarioResult }
  | { type: "add-feedback-submission-id"; fingerprint: string };

function clampTimeout(value: number): number {
  return Math.min(600, Math.max(1, Math.round(value)));
}

function isWorkspace(value: unknown): value is Workspace {
  return value === "Assistant" || value === "Evaluation" || value === "System";
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return !!value && typeof value === "object" && !Array.isArray(value);
}

function hydrateState(runtimeDefaults: FrontendRuntimeConfig): AppState {
  const baseState: AppState = {
    currentWorkspace: "Assistant",
    drawerOpen: false,
    apiBaseUrl: runtimeDefaults.apiBaseUrl,
    apiEndpointPath: runtimeDefaults.apiEndpointPath,
    timeoutS: clampTimeout(runtimeDefaults.apiTimeoutS),
    debugMode: false,
    manualConstraintsOpen: false,
    manualConstraints: { ...DEFAULT_MANUAL_CONSTRAINTS },
    assistantMessages: [],
    assistantHistoryOpen: false,
    evaluationResults: {},
    feedbackSubmissionIds: [],
    displayName: runtimeDefaults.displayName,
  };

  if (typeof window === "undefined") {
    return baseState;
  }

  const rawState = window.sessionStorage.getItem(SESSION_STORAGE_KEY);
  if (!rawState) {
    return baseState;
  }

  try {
    const parsed = JSON.parse(rawState) as Record<string, unknown>;
    return {
      ...baseState,
      currentWorkspace: isWorkspace(parsed.currentWorkspace) ? parsed.currentWorkspace : baseState.currentWorkspace,
      drawerOpen: Boolean(parsed.drawerOpen),
      apiBaseUrl:
        typeof parsed.apiBaseUrl === "string" && parsed.apiBaseUrl.trim() ? parsed.apiBaseUrl : baseState.apiBaseUrl,
      apiEndpointPath:
        typeof parsed.apiEndpointPath === "string" && parsed.apiEndpointPath.trim()
          ? parsed.apiEndpointPath
          : baseState.apiEndpointPath,
      timeoutS: clampTimeout(Number(parsed.timeoutS ?? baseState.timeoutS)),
      debugMode: Boolean(parsed.debugMode),
      manualConstraintsOpen: Boolean(parsed.manualConstraintsOpen),
      manualConstraints: {
        ...DEFAULT_MANUAL_CONSTRAINTS,
        ...(isRecord(parsed.manualConstraints) ? parsed.manualConstraints : {}),
      } as ManualConstraintFields,
      assistantMessages: Array.isArray(parsed.assistantMessages)
        ? (parsed.assistantMessages as AssistantMessage[])
        : baseState.assistantMessages,
      assistantHistoryOpen: Boolean(parsed.assistantHistoryOpen),
      evaluationResults: isRecord(parsed.evaluationResults)
        ? (parsed.evaluationResults as Record<string, EvaluationScenarioResult>)
        : baseState.evaluationResults,
      feedbackSubmissionIds: Array.isArray(parsed.feedbackSubmissionIds)
        ? [...new Set(parsed.feedbackSubmissionIds.map((item) => String(item)))]
        : baseState.feedbackSubmissionIds,
      displayName: baseState.displayName,
    };
  } catch {
    return baseState;
  }
}

function reducer(state: AppState, action: Action): AppState {
  switch (action.type) {
    case "set-workspace":
      return { ...state, currentWorkspace: action.workspace };
    case "toggle-drawer":
      return { ...state, drawerOpen: !state.drawerOpen };
    case "set-drawer-open":
      return { ...state, drawerOpen: action.open };
    case "set-api-base-url":
      return { ...state, apiBaseUrl: action.value };
    case "set-api-endpoint-path":
      return { ...state, apiEndpointPath: action.value };
    case "increment-timeout":
      return { ...state, timeoutS: clampTimeout(state.timeoutS + action.delta) };
    case "set-debug-mode":
      return { ...state, debugMode: action.value };
    case "toggle-manual-constraints":
      return { ...state, manualConstraintsOpen: !state.manualConstraintsOpen };
    case "set-manual-constraint":
      return {
        ...state,
        manualConstraints: {
          ...state.manualConstraints,
          [action.field]: action.value,
        },
      };
    case "append-assistant-message":
      return {
        ...state,
        assistantMessages: [...state.assistantMessages, action.message],
      };
    case "replace-assistant-message":
      return {
        ...state,
        assistantMessages: state.assistantMessages.map((message) =>
          message.id === action.messageId ? action.message : message,
        ),
      };
    case "clear-assistant-session":
      return {
        ...state,
        assistantMessages: [],
        assistantHistoryOpen: false,
      };
    case "toggle-assistant-history":
      return { ...state, assistantHistoryOpen: !state.assistantHistoryOpen };
    case "set-evaluation-result":
      return {
        ...state,
        evaluationResults: {
          ...state.evaluationResults,
          [action.result.scenarioId]: action.result,
        },
      };
    case "add-feedback-submission-id":
      return state.feedbackSubmissionIds.includes(action.fingerprint)
        ? state
        : {
            ...state,
            feedbackSubmissionIds: [...state.feedbackSubmissionIds, action.fingerprint],
          };
    default:
      return state;
  }
}

const AppStateContext = createContext<{ state: AppState; dispatch: Dispatch<Action> } | null>(null);

export function AppProvider({
  children,
  runtimeDefaults,
}: {
  children: ReactNode;
  runtimeDefaults: FrontendRuntimeConfig;
}) {
  const [state, dispatch] = useReducer(reducer, runtimeDefaults, hydrateState);

  useEffect(() => {
    if (typeof window === "undefined") {
      return;
    }
    window.sessionStorage.setItem(SESSION_STORAGE_KEY, JSON.stringify(state));
  }, [state]);

  return <AppStateContext.Provider value={{ state, dispatch }}>{children}</AppStateContext.Provider>;
}

export function useAppState() {
  const context = useContext(AppStateContext);
  if (!context) {
    throw new Error("useAppState must be used within AppProvider");
  }
  return context;
}
