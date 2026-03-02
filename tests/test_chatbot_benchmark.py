"""
CortexFlow — Chatbot Scenario Benchmark with Vertex AI (Gemini)
================================================================
Real-world chatbot scenario benchmark that tests CortexFlow end-to-end
using live Vertex AI calls. Covers multi-turn conversations, long context
handling, memory tier cascading, knowledge retrieval augmentation, context
window overflow, and response latency.

Run the full benchmark (~3-5 min with live LLM calls):
    python -m pytest tests/test_chatbot_benchmark.py -v -s --timeout=600

Quick smoke test (section 1 only):
    python -m pytest tests/test_chatbot_benchmark.py::TestChatbotBenchmark::test_01_basic_conversation -v -s
"""

import os
import statistics
import tempfile
import time

import pytest

# ---------------------------------------------------------------------------
# Vertex AI credentials & skip guard
# ---------------------------------------------------------------------------
# Read credentials from environment variables — never hardcode secrets.
#   export VERTEX_CREDENTIALS_PATH=/path/to/service-account.json
#   export VERTEX_PROJECT_ID=your-gcp-project
CREDENTIALS_PATH = os.environ.get(
    "VERTEX_CREDENTIALS_PATH",
    os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", ""),
)

pytestmark = pytest.mark.skipif(
    not CREDENTIALS_PATH or not os.path.exists(CREDENTIALS_PATH),
    reason="Vertex AI credentials not found (set VERTEX_CREDENTIALS_PATH)",
)

# ---------------------------------------------------------------------------
# Imports (done at module level so import errors surface immediately)
# ---------------------------------------------------------------------------
from cortexflow.config import ConfigBuilder  # noqa: E402
from cortexflow.manager import CortexFlowManager  # noqa: E402

# ---------------------------------------------------------------------------
# Colour helpers for -s output
# ---------------------------------------------------------------------------
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"


def _ok(msg):
    print(f"  {GREEN}PASS{RESET} {msg}")


def _fail(msg):
    print(f"  {RED}FAIL{RESET} {msg}")


def _info(msg):
    print(f"  {CYAN}>{RESET} {msg}")


def _header(title):
    print(f"\n{BOLD}{CYAN}{'─'*60}{RESET}")
    print(f"{BOLD}{CYAN}  {title}{RESET}")
    print(f"{BOLD}{CYAN}{'─'*60}{RESET}")


# ---------------------------------------------------------------------------
# Shared temp directory for SQLite knowledge-store files
# ---------------------------------------------------------------------------
_tmp_dir = tempfile.mkdtemp(prefix="cortexflow_chatbot_bench_")


def _tmp_db(name: str) -> str:
    return os.path.join(_tmp_dir, f"{name}.db")


# ---------------------------------------------------------------------------
# Factory: build a fresh CortexFlowManager with Vertex AI
# ---------------------------------------------------------------------------
def _make_manager(
    *,
    active_tokens=4096,
    working_tokens=8192,
    archive_tokens=16384,
    db_name="default",
):
    config = (
        ConfigBuilder()
        .with_vertex_ai(
            project_id=os.environ.get("VERTEX_PROJECT_ID", ""),
            location=os.environ.get("VERTEX_LOCATION", "us-central1"),
            default_model="gemini-2.0-flash",
            credentials_path=CREDENTIALS_PATH,
        )
        .with_memory(
            active_token_limit=active_tokens,
            working_token_limit=working_tokens,
            archive_token_limit=archive_tokens,
        )
        .with_knowledge_store(knowledge_store_path=_tmp_db(db_name))
        .build()
    )
    return CortexFlowManager(config)


# ---------------------------------------------------------------------------
# Metrics collector (shared across all tests via class attribute)
# ---------------------------------------------------------------------------
class Metrics:
    """Accumulates benchmark metrics across all test sections."""

    latencies: list = []
    tier_snapshots: list = []
    knowledge_hits: int = 0
    knowledge_attempts: int = 0
    section_results: dict = {}

    @classmethod
    def reset(cls):
        cls.latencies = []
        cls.tier_snapshots = []
        cls.knowledge_hits = 0
        cls.knowledge_attempts = 0
        cls.section_results = {}

    @classmethod
    def record_latency(cls, section, value):
        cls.latencies.append({"section": section, "latency": value})

    @classmethod
    def snapshot_tiers(cls, section, memory):
        cls.tier_snapshots.append({
            "section": section,
            "active_tokens": memory.active_tier.current_token_count,
            "active_max": memory.active_tier.max_tokens,
            "active_segments": len(memory.active_tier.segments),
            "working_tokens": memory.working_tier.current_token_count,
            "working_max": memory.working_tier.max_tokens,
            "working_segments": len(memory.working_tier.segments),
            "archive_tokens": memory.archive_tier.current_token_count,
            "archive_max": memory.archive_tier.max_tokens,
            "archive_segments": len(memory.archive_tier.segments),
        })

    @classmethod
    def record_section(cls, name, passed, total, details=""):
        cls.section_results[name] = {
            "passed": passed,
            "total": total,
            "details": details,
        }


# ═══════════════════════════════════════════════════════════════════════════
# Test class
# ═══════════════════════════════════════════════════════════════════════════
class TestChatbotBenchmark:
    """End-to-end chatbot benchmark against live Vertex AI / Gemini."""

    # ── helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _timed_response(manager, section_label):
        """Call generate_response(), record latency, return (response, elapsed)."""
        t0 = time.time()
        response = manager.generate_response()
        elapsed = time.time() - t0
        Metrics.record_latency(section_label, elapsed)
        return response, elapsed

    # ── 1. Basic Chatbot Conversation (5 turns) ─────────────────────────

    def test_01_basic_conversation(self):
        _header("1. Basic Chatbot Conversation (5 turns)")
        manager = _make_manager(db_name="basic_conv")
        section = "basic_conversation"

        turns = [
            "Hello! Can you help me learn about Python decorators?",
            "Can you show me a simple decorator example?",
            "How would I use a decorator with arguments?",
            "What's the difference between @staticmethod and @classmethod?",
            "Thanks! Can you summarize the key points we discussed?",
        ]

        passed = 0
        for i, user_msg in enumerate(turns, 1):
            manager.add_message("user", user_msg)
            response, elapsed = self._timed_response(manager, section)

            is_ok = bool(response) and not response.startswith("Error")
            if is_ok:
                passed += 1
                _ok(f"Turn {i}: got response in {elapsed:.2f}s ({len(response)} chars)")
            else:
                _fail(f"Turn {i}: {response[:120]}")

        Metrics.snapshot_tiers(section, manager.memory)
        Metrics.record_section(section, passed, len(turns))

        assert passed == len(turns), f"Only {passed}/{len(turns)} turns succeeded"

    # ── 2. Long Context Stress Test (50+ messages) ──────────────────────

    def test_02_long_context_stress(self):
        _header("2. Long Context Stress Test (50+ messages)")
        manager = _make_manager(
            active_tokens=2048,
            working_tokens=4096,
            archive_tokens=8192,
            db_name="long_ctx",
        )
        section = "long_context"

        # Pump 50 user/assistant pairs (100 messages total)
        num_pairs = 50
        topics = [
            "quantum computing", "machine learning", "blockchain",
            "climate change", "space exploration", "genetics",
            "renewable energy", "cybersecurity", "robotics",
            "neuroscience",
        ]

        for i in range(num_pairs):
            topic = topics[i % len(topics)]
            # Use longer messages (~80 words each) so 50 pairs actually overflow
            # the 2048-token active tier and trigger cascade.
            manager.add_message(
                "user",
                f"Tell me an interesting and detailed fact about {topic}. "
                f"I would like to understand the key concepts, recent breakthroughs, "
                f"and why this field matters for the future of technology and society. "
                f"This is message number {i+1} in our ongoing conversation about "
                f"various scientific and technical topics that I find fascinating.",
            )
            manager.add_message(
                "assistant",
                f"Here is a detailed fact about {topic}: this is a rapidly evolving "
                f"field with many groundbreaking developments in recent years. "
                f"Researchers have made significant progress in understanding the "
                f"fundamental principles and practical applications. The implications "
                f"for society are profound, ranging from healthcare to environmental "
                f"protection. Fact number {i+1} for conversation tracking purposes.",
            )

        Metrics.snapshot_tiers(section, manager.memory)
        mem = manager.memory

        _info(f"Active  : {mem.active_tier.current_token_count}/{mem.active_tier.max_tokens} tokens, "
              f"{len(mem.active_tier.segments)} segments")
        _info(f"Working : {mem.working_tier.current_token_count}/{mem.working_tier.max_tokens} tokens, "
              f"{len(mem.working_tier.segments)} segments")
        _info(f"Archive : {mem.archive_tier.current_token_count}/{mem.archive_tier.max_tokens} tokens, "
              f"{len(mem.archive_tier.segments)} segments")

        # Verify tier cascade actually happened
        cascade_happened = (
            mem.working_tier.current_token_count > 0
            or mem.archive_tier.current_token_count > 0
        )
        if cascade_happened:
            _ok("Tier cascade triggered (working/archive have content)")
        else:
            _fail("No tier cascade detected — all content still in active tier")

        # Now do a live LLM call to verify the system still works after stress
        manager.add_message("user", "Summarize what topics we discussed.")
        response, elapsed = self._timed_response(manager, section)
        response_ok = bool(response) and not response.startswith("Error")
        if response_ok:
            _ok(f"Post-stress LLM call succeeded in {elapsed:.2f}s")
        else:
            _fail(f"Post-stress LLM call failed: {response[:120]}")

        total_tokens = (
            mem.active_tier.current_token_count
            + mem.working_tier.current_token_count
            + mem.archive_tier.current_token_count
        )
        total_capacity = (
            mem.active_tier.max_tokens
            + mem.working_tier.max_tokens
            + mem.archive_tier.max_tokens
        )
        compression_ratio = total_tokens / total_capacity if total_capacity else 0
        _info(f"Total tokens across tiers: {total_tokens}/{total_capacity} "
              f"(usage {compression_ratio:.1%})")

        passed = int(cascade_happened) + int(response_ok)
        Metrics.record_section(section, passed, 2)

        assert cascade_happened, "Memory tier cascade should trigger with 100 messages"
        assert response_ok, "LLM should still respond after stress test"

    # ── 3. Knowledge-Augmented Retrieval ─────────────────────────────────

    def test_03_knowledge_retrieval(self):
        _header("3. Knowledge-Augmented Retrieval")
        manager = _make_manager(db_name="knowledge_retr")
        section = "knowledge_retrieval"

        # Pre-load 20 knowledge facts
        facts = [
            ("The capital of France is Paris.", "geography"),
            ("Python was created by Guido van Rossum in 1991.", "programming"),
            ("The speed of light is approximately 299,792 km/s.", "physics"),
            ("DNA stands for deoxyribonucleic acid.", "biology"),
            ("The Great Wall of China is over 21,000 km long.", "geography"),
            ("JavaScript was created by Brendan Eich in 1995.", "programming"),
            ("Water boils at 100 degrees Celsius at sea level.", "physics"),
            ("The human body has 206 bones.", "biology"),
            ("Tokyo is the capital of Japan.", "geography"),
            ("The Rust programming language was first released in 2010.", "programming"),
            ("Earth's atmosphere is 78% nitrogen.", "physics"),
            ("Mitochondria are known as the powerhouse of the cell.", "biology"),
            ("The Amazon River is the largest river by volume.", "geography"),
            ("HTML stands for HyperText Markup Language.", "programming"),
            ("Gravity on Earth is approximately 9.81 m/s squared.", "physics"),
            ("The human genome contains about 3 billion base pairs.", "biology"),
            ("Mount Everest is 8,849 meters tall.", "geography"),
            ("Git was created by Linus Torvalds in 2005.", "programming"),
            ("Absolute zero is minus 273.15 degrees Celsius.", "physics"),
            ("Photosynthesis converts CO2 and water into glucose and oxygen.", "biology"),
        ]

        for text, source in facts:
            manager.add_knowledge(text, source=source)
        _info(f"Loaded {len(facts)} knowledge facts")

        # Queries and the keyword we expect in the response
        queries = [
            ("What is the capital of France?", "paris"),
            ("Who created Python?", "guido"),
            ("What is the speed of light?", "299"),
            ("What does DNA stand for?", "deoxyribonucleic"),
            ("How long is the Great Wall of China?", "21"),
            ("Who created Git?", "torvalds"),
            ("What is absolute zero?", "273"),
            ("What are mitochondria known as?", "powerhouse"),
        ]

        passed = 0
        for question, expected_keyword in queries:
            manager.add_message("user", question)
            response, elapsed = self._timed_response(manager, section)

            Metrics.knowledge_attempts += 1
            found = expected_keyword.lower() in response.lower()
            if found:
                Metrics.knowledge_hits += 1
                passed += 1
                _ok(f"'{expected_keyword}' found — {elapsed:.2f}s")
            else:
                _fail(f"'{expected_keyword}' NOT in response for: {question}")
                _info(f"  Response snippet: {response[:200]}")

            # Clear conversation (not knowledge) between queries for isolation
            manager.memory.clear_memory()

        Metrics.snapshot_tiers(section, manager.memory)
        Metrics.record_section(section, passed, len(queries))

        hit_rate = passed / len(queries) if queries else 0
        _info(f"Knowledge retrieval hit rate: {hit_rate:.0%} ({passed}/{len(queries)})")

        assert passed >= len(queries) // 2, (
            f"Knowledge retrieval accuracy too low: {passed}/{len(queries)}"
        )

    # ── 4. Memory Persistence & Recall ──────────────────────────────────

    def test_04_memory_persistence(self):
        _header("4. Memory Persistence & Recall")
        manager = _make_manager(
            active_tokens=2048,
            working_tokens=4096,
            archive_tokens=8192,
            db_name="mem_persist",
        )
        section = "memory_persistence"

        # Store user preferences early
        manager.add_message("user", "My name is Alice and I live in Tokyo.")
        manager.add_message(
            "assistant",
            "Nice to meet you, Alice! Tokyo is a wonderful city. I'll remember that.",
        )
        manager.add_message("user", "My favorite programming language is Rust.")
        manager.add_message(
            "assistant",
            "Great choice! Rust is known for its memory safety. I've noted that.",
        )

        # Push 30 intervening message pairs to cascade the preferences through tiers
        for i in range(30):
            manager.add_message(
                "user",
                f"Let's talk about topic number {i+1}. Tell me about data structures.",
            )
            manager.add_message(
                "assistant",
                f"Data structures are fundamental to computer science. "
                f"Topic {i+1} explored. Arrays, linked lists, trees, and graphs "
                f"are all important data structures.",
            )

        Metrics.snapshot_tiers(section, manager.memory)
        _info(f"Injected 30 filler pairs — tiers: "
              f"A={manager.memory.active_tier.current_token_count}, "
              f"W={manager.memory.working_tier.current_token_count}, "
              f"AR={manager.memory.archive_tier.current_token_count}")

        # Now ask recall questions via live LLM
        recall_checks = [
            ("What is my name?", "alice"),
            ("Where do I live?", "tokyo"),
            ("What is my favorite programming language?", "rust"),
        ]

        passed = 0
        for question, expected in recall_checks:
            manager.add_message("user", question)
            response, elapsed = self._timed_response(manager, section)

            found = expected.lower() in response.lower()
            if found:
                passed += 1
                _ok(f"Recalled '{expected}' — {elapsed:.2f}s")
            else:
                _fail(f"Failed to recall '{expected}' for: {question}")
                _info(f"  Response: {response[:200]}")

        Metrics.record_section(section, passed, len(recall_checks))

        # We allow partial success here since deep-tier recall is challenging
        assert passed >= 1, (
            f"Should recall at least 1 of {len(recall_checks)} stored preferences"
        )

    # ── 5. Context Window Overflow Handling ─────────────────────────────

    def test_05_overflow_handling(self):
        _header("5. Context Window Overflow Handling")
        manager = _make_manager(
            active_tokens=1024,
            working_tokens=2048,
            archive_tokens=4096,
            db_name="overflow",
        )
        section = "overflow_handling"

        # Add a system message that should be preserved
        manager.add_message("system", "You are a helpful assistant. Always be polite.")

        # Flood with 100 messages to massively exceed total capacity (7168 tokens)
        num_messages = 100
        crash = False
        for i in range(num_messages):
            try:
                manager.add_message(
                    "user",
                    f"Message {i+1}: Please tell me about topic number {i+1} in detail. "
                    f"I want to know everything about this interesting subject.",
                )
                manager.add_message(
                    "assistant",
                    f"Response {i+1}: Here is detailed information about topic {i+1}. "
                    f"This is a comprehensive overview covering many aspects of the subject.",
                )
            except Exception as e:
                crash = True
                _fail(f"Crash at message {i+1}: {e}")
                break

        mem = manager.memory
        Metrics.snapshot_tiers(section, mem)

        _info(f"Active  : {mem.active_tier.current_token_count}/{mem.active_tier.max_tokens}")
        _info(f"Working : {mem.working_tier.current_token_count}/{mem.working_tier.max_tokens}")
        _info(f"Archive : {mem.archive_tier.current_token_count}/{mem.archive_tier.max_tokens}")
        _info(f"Total messages tracked: {len(mem.messages)}")

        checks_passed = 0
        total_checks = 3

        # Check 1: no crashes
        if not crash:
            checks_passed += 1
            _ok(f"No crashes after {num_messages * 2} messages")
        else:
            _fail("System crashed during overflow test")

        # Check 2: responses still generated
        manager.add_message("user", "Are you still working?")
        response, elapsed = self._timed_response(manager, section)
        if response and not response.startswith("Error"):
            checks_passed += 1
            _ok(f"Post-overflow response OK — {elapsed:.2f}s")
        else:
            _fail(f"Post-overflow response failed: {response[:120]}")

        # Check 3: tiers are within limits
        within_limits = (
            mem.active_tier.current_token_count <= mem.active_tier.max_tokens
            and mem.working_tier.current_token_count <= mem.working_tier.max_tokens
            and mem.archive_tier.current_token_count <= mem.archive_tier.max_tokens
        )
        if within_limits:
            checks_passed += 1
            _ok("All tiers within token limits")
        else:
            _fail("One or more tiers exceeded their token limit")

        Metrics.record_section(section, checks_passed, total_checks)

        assert not crash, "System must not crash on overflow"
        assert checks_passed >= 2, f"Only {checks_passed}/{total_checks} overflow checks passed"

    # ── 6. Concurrent Knowledge + Conversation ──────────────────────────

    def test_06_concurrent_knowledge_conversation(self):
        _header("6. Concurrent Knowledge + Conversation")
        manager = _make_manager(db_name="concurrent_kc")
        section = "concurrent_kc"

        # Interleave knowledge additions with conversation
        interactions = [
            ("knowledge", "The CortexFlow framework uses a three-tier memory architecture."),
            ("chat", "What memory architecture does CortexFlow use?", "three-tier"),
            ("knowledge", "CortexFlow supports BM25, Dense, and Hybrid search strategies."),
            ("chat", "What search strategies does CortexFlow support?", "bm25"),
            ("knowledge", "CortexFlow's archive tier uses extractive compression at 30% ratio."),
            ("chat", "How does CortexFlow compress data in the archive tier?", "30"),
            ("knowledge", "CortexFlow can integrate with Vertex AI for LLM inference."),
            ("chat", "What LLM backends does CortexFlow support?", "vertex"),
            ("knowledge", "CortexFlow's knowledge graph uses NetworkX for graph operations."),
            ("chat", "What library does CortexFlow use for graph operations?", "networkx"),
        ]

        knowledge_count = 0
        chat_passed = 0
        chat_total = 0

        for item in interactions:
            if item[0] == "knowledge":
                manager.add_knowledge(item[1], source="benchmark")
                knowledge_count += 1
                _info(f"Added knowledge fact #{knowledge_count}")
            else:
                _, question, expected_keyword = item
                chat_total += 1
                manager.add_message("user", question)
                response, elapsed = self._timed_response(manager, section)

                Metrics.knowledge_attempts += 1
                found = expected_keyword.lower() in response.lower()
                if found:
                    Metrics.knowledge_hits += 1
                    chat_passed += 1
                    _ok(f"'{expected_keyword}' found — {elapsed:.2f}s")
                else:
                    _fail(f"'{expected_keyword}' NOT in response for: {question}")
                    _info(f"  Response snippet: {response[:200]}")

                # Clear conversation between queries, keep knowledge
                manager.memory.clear_memory()

        Metrics.snapshot_tiers(section, manager.memory)
        Metrics.record_section(section, chat_passed, chat_total)

        _info(f"Concurrent K+C hit rate: {chat_passed}/{chat_total}")
        assert chat_passed >= chat_total // 2, (
            f"Concurrent K+C accuracy too low: {chat_passed}/{chat_total}"
        )

    # ── 7. Performance Summary Report ───────────────────────────────────

    def test_07_performance_summary(self):
        _header("7. Performance Summary Report")

        # ── Latency stats ──
        all_latencies = [m["latency"] for m in Metrics.latencies]
        if all_latencies:
            all_latencies_sorted = sorted(all_latencies)
            n = len(all_latencies_sorted)
            p50 = all_latencies_sorted[int(n * 0.50)]
            p95 = all_latencies_sorted[int(min(n * 0.95, n - 1))]
            p99 = all_latencies_sorted[int(min(n * 0.99, n - 1))]
            mean_lat = statistics.mean(all_latencies)
            total_time = sum(all_latencies)

            print(f"\n  {BOLD}Latency (across {n} LLM calls):{RESET}")
            print(f"    Mean   : {mean_lat:.2f}s")
            print(f"    p50    : {p50:.2f}s")
            print(f"    p95    : {p95:.2f}s")
            print(f"    p99    : {p99:.2f}s")
            print(f"    Total  : {total_time:.1f}s")

            # Per-section breakdown
            sections_seen = {}
            for m in Metrics.latencies:
                sections_seen.setdefault(m["section"], []).append(m["latency"])
            print(f"\n  {BOLD}Latency by section:{RESET}")
            for sec, lats in sections_seen.items():
                print(f"    {sec:30s}  avg={statistics.mean(lats):.2f}s  "
                      f"n={len(lats)}  total={sum(lats):.1f}s")
        else:
            print(f"  {YELLOW}No latency data collected.{RESET}")

        # ── Memory tier stats ──
        if Metrics.tier_snapshots:
            print(f"\n  {BOLD}Memory Tier Snapshots:{RESET}")
            for snap in Metrics.tier_snapshots:
                print(f"    [{snap['section']}]")
                print(f"      Active  : {snap['active_tokens']:>6}/{snap['active_max']} tokens, "
                      f"{snap['active_segments']} segments")
                print(f"      Working : {snap['working_tokens']:>6}/{snap['working_max']} tokens, "
                      f"{snap['working_segments']} segments")
                print(f"      Archive : {snap['archive_tokens']:>6}/{snap['archive_max']} tokens, "
                      f"{snap['archive_segments']} segments")

        # ── Knowledge retrieval accuracy ──
        if Metrics.knowledge_attempts > 0:
            hit_rate = Metrics.knowledge_hits / Metrics.knowledge_attempts
            print(f"\n  {BOLD}Knowledge Retrieval:{RESET}")
            print(f"    Attempts : {Metrics.knowledge_attempts}")
            print(f"    Hits     : {Metrics.knowledge_hits}")
            print(f"    Hit Rate : {hit_rate:.0%}")

        # ── Pass/fail summary ──
        print(f"\n  {BOLD}Section Results:{RESET}")
        total_passed = 0
        total_tests = 0
        for sec_name, info in Metrics.section_results.items():
            p, t = info["passed"], info["total"]
            total_passed += p
            total_tests += t
            color = GREEN if p == t else (YELLOW if p > 0 else RED)
            print(f"    {color}{sec_name:35s} {p}/{t}{RESET}")

        if total_tests:
            overall_rate = total_passed / total_tests
            color = GREEN if overall_rate >= 0.8 else (YELLOW if overall_rate >= 0.5 else RED)
            print(f"\n  {BOLD}Overall: {color}{total_passed}/{total_tests} "
                  f"({overall_rate:.0%}){RESET}")
        else:
            print(f"\n  {YELLOW}No test results collected.{RESET}")

        # This test always passes — it's a report, not an assertion
        assert True
