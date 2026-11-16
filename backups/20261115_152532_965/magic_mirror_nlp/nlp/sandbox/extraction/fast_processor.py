import threading
import queue


class FastProcessor:
    """
    Low-latency coordinator between ASR and the local Qwen extractor.
    Partial transcripts go to the UI.
    Each finalized utterance is independently forwarded to Qwen.
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
                    ui_queue.put({
                        "type": "transcript_update",
                        "text": text,
                        "is_partial": False
                    })

                    nlp_queue.put({
                        "text": text
                    })

                    print(f"[FAST→QWEN] Final utterance queued: {text!r}")

            except queue.Empty:
                continue
            except Exception as e:
                print(f"[FAST_PROCESSOR_ERROR] {e}")