/**
 * trim-images — keep images only from the most recent image-bearing message.
 *
 * Why: pi re-sends the full conversation on every request, so images seen in
 * earlier turns stay in history and accumulate; once the request carries more
 * than the server's per-request image limit (MM_IMAGES), vLLM returns HTTP 400
 * and the whole session appears dead (every later request re-sends the same
 * oversize payload).
 *
 * What it does: before each LLM call, drop `image` parts from all messages
 * except the last message that contains one. Text, tool results and reasoning
 * parts are untouched; the model still sees the most recent image.
 */
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import type { AgentMessage } from "@earendil-works/pi-agent-core";

export default function (pi: ExtensionAPI) {
	pi.on("context", (event: { messages: AgentMessage[] }) => {
		const msgs = event.messages;
		if (!Array.isArray(msgs) || msgs.length === 0) return;

		// find the LAST message that carries at least one image part
		let lastImageMsg = -1;
		for (let i = 0; i < msgs.length; i++) {
			const c = (msgs[i] as unknown as { content?: unknown })?.content;
			if (Array.isArray(c) && c.some((p) => p && (p as { type?: string }).type === "image")) {
				lastImageMsg = i;
			}
		}

		// strip image parts from every older message
		for (let i = 0; i < msgs.length; i++) {
			if (i === lastImageMsg) continue;
			const m = msgs[i] as unknown as { content?: unknown[] };
			const c = m?.content;
			if (!Array.isArray(c)) continue;
			const kept = c.filter((p) => !(p && (p as { type?: string }).type === "image"));
			if (kept.length !== c.length) m.content = kept;
		}
	});
}