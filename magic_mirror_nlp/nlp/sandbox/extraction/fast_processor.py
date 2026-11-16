import re
import time
import threading
import queue


_FAST_COMMAND = re.compile(
    r"^(?:(show me|create|draw|make)|(?P<action>update|change|move|put|animate|remove|delete|hide))\s+"
    r"(?:(?:a|an|the)\s+)?(?P<body>.+?)[.!]?$", re.IGNORECASE
)
_COLORS = {"red", "blue", "green", "yellow", "black", "white", "orange", "purple", "pink", "brown", "gray", "grey"}
_ANCHORS = {"on my hand": "hand", "on my palm": "hand", "in front of me": "person"}


def parse_simple_command(text):
    """Parse only single-object, literal commands; return None when anything is uncertain."""
    match = _FAST_COMMAND.fullmatch(text.strip())
    if not match or re.search(r"\b(it|this|that|these|those|not|don't|do not|but|and|or|then|left|right|above|below|beside|behind|between)\b", text, re.I):
        return None
    lead = match.group(1)
    action = match.group("action") or "create"
    body = match.group("body").strip().lower()
    anchor = None
    action_detail = None
    verb = (lead or action).lower()
    if verb in {"update", "change"}:
        return None
    if verb in {"animate"}:
        for phrase, value in ((" to rotate", "rotating"), (" to run", "running"), (" to stand", "standing")):
            if body.endswith(phrase):
                body, action_detail = body[:-len(phrase)].strip(), value
                break
        if action_detail is None:
            return None
    for phrase, value in _ANCHORS.items():
        suffix = " " + phrase
        if body.endswith(suffix):
            body, anchor = body[:-len(suffix)].strip(), value
            break
    if verb in {"move", "put"} and anchor is None:
        return None
    words = body.split()
    modifier = words[0] if words and words[0] in _COLORS else None
    concept_words = words[1:] if modifier else words
    if not concept_words or len(concept_words) > 3 or any(w in _COLORS for w in concept_words):
        return None
    concept = " ".join(concept_words)
    if not re.fullmatch(r"[a-z]+(?: [a-z]+){0,2}", concept):
        return None
    if verb in {"show me", "create", "draw", "make"}:
        intent = "demonstrating"
    elif verb in {"remove", "delete", "hide"}:
        intent = "instructing"
    elif verb in {"move", "put"}:
        intent = "instructing"
    elif verb == "animate":
        intent = "instructing"
    else:
        return None
    relationships = []
    if modifier:
        relationships.append({"subject": concept, "predicate": "color", "object": modifier, "type": "property", "confidence": 1.0})
    if anchor:
        relationships.append({"subject": concept, "predicate": "anchored_on", "object": anchor, "type": "spatial", "confidence": 1.0})
    if action_detail:
        relationships.append({"subject": concept, "predicate": "animation", "object": action_detail, "type": "action", "confidence": 1.0})
    return {
        "timestamp": time.time(), "is_partial": False, "text_segment": text.strip(),
        "speaker_activity": [{"intent": intent, "confidence": 1.0}],
        "semantic_graph": {"entities": [{"id": concept.replace(" ", "_"), "name": concept, "class": "object", "status": "new_explicit"}], "relationships": relationships, "quantities": []},
        "explicit_information": ([f"{concept} color {modifier}"] if modifier else []) + ([f"{concept} anchored on {anchor}"] if anchor else []) + ([f"{concept} {action_detail}"] if action_detail else []),
        "inferred_information": [], "coreference_resolutions": [],
        "explanation_state": {"current_topic": concept, "active_concept": concept, "recent_entities": [concept]},
    }


class FastProcessor:
    """
    Low-latency coordinator between ASR and the local Qwen extractor.
    Partial transcripts go to the UI.
    Literal supported commands are emitted directly; all other finals go to Qwen.
    """

    def __init__(self):
        self.is_running = False
        self.thread = None

    def start_processing(self, text_queue: queue.Queue, nlp_queue: queue.Queue, ui_queue: queue.Queue):
        if self.is_running:
            return

        self.is_running = True
        self.thread = threading.Thread(
            target=self._process_loop,
            args=(text_queue, nlp_queue, ui_queue),
            daemon=True
        )
        self.thread.start()
        print("Fast Processor started.")

    def stop_processing(self):
        self.is_running = False
        if self.thread:
            self.thread.join()
        print("Fast Processor stopped.")

    def _process_loop(self, text_queue: queue.Queue, nlp_queue: queue.Queue, ui_queue: queue.Queue):
        while self.is_running:
            try:
                update = text_queue.get(timeout=0.1)
                text = update.get("text", "").strip()
                is_partial = update.get("is_partial", True)

                if not text:
                    continue

                if is_partial:
                    ui_queue.put({
                        "type": "transcript_update",
                        "text": text,
                        "is_partial": True
                    })
                else:
                    correlation_id = update.get("correlation_id")
                    finalized_ns = update.get("finalized_ns")
                    ui_queue.put({
                        "type": "transcript_update",
                        "text": text,
                        "is_partial": False
                    })
                    started_ns = time.perf_counter_ns()
                    fast_state = parse_simple_command(text)
                    parse_ms = (time.perf_counter_ns() - started_ns) / 1_000_000
                    if fast_state:
                        fast_state["timestamp"] = update.get("timestamp", fast_state["timestamp"])
                        ui_queue.put({"type": "nlp_update", "state": fast_state,
                                      "correlation_id": correlation_id, "parser": "deterministic",
                                      "parser_ms": parse_ms, "finalized_ns": finalized_ns})
                        print(f"[FAST] Deterministic command ({parse_ms:.3f} ms): {text!r}")
                    else:
                        nlp_queue.put({"text": text, "correlation_id": correlation_id,
                                       "finalized_ns": finalized_ns, "enqueued_ns": time.perf_counter_ns()})
                        print(f"[FAST→QWEN] Final utterance queued: {text!r}")

            except queue.Empty:
                continue
            except Exception as e:
                print(f"[FAST_PROCESSOR_ERROR] {e}")
