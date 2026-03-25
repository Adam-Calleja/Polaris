import "@testing-library/jest-dom/vitest";
import { afterEach, vi } from "vitest";
import { webcrypto } from "node:crypto";

if (!globalThis.crypto) {
  Object.defineProperty(globalThis, "crypto", {
    value: webcrypto,
    configurable: true,
  });
}

afterEach(() => {
  vi.restoreAllMocks();
});
