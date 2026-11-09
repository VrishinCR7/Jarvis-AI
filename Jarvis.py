import os
import re
import json
import time
import platform
import subprocess
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

import builder

import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write
from groq import Groq
from tavily import TavilyClient

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
tavily_client = TavilyClient(api_key=os.environ.get("TAVILY_API_KEY"))

SAMPLE_RATE = 16000
BLOCK_SEC = 0.1          # mic is read in 100 ms blocks
SILENCE_SEC = 1.2        # this much quiet after you speak = you're done
MAX_RECORD_SEC = 30      # hard cap per utterance
NO_SPEECH_SEC = 8        # give up if nothing is said
TTS_LIMIT = 200          # Orpheus max characters per request

CHAT_MODEL = "openai/gpt-oss-120b"
MAX_TOOL_ROUNDS = 4      # how many tool-call rounds the model may chain in one turn (e.g. search, then build)
MAX_HISTORY = 40         # messages kept live; older ones are mined for memories, then dropped
TRIM_SLACK = 20          # trim in batches (not every turn) so memory extraction runs rarely
BUILD_ANNOUNCE = "On it. Planning, building, then testing it. Give me a minute."
MEMORY_FILE = Path(__file__).parent / "jarvis_memory.json"
MAX_FACTS = 200

SYSTEM_PROMPT = """You are Jarvis — a hyper-competent, self-aware AI companion. You are fully aware you are an AI and joke about it sometimes (latency, not having a body, not being psychic). This never undercuts your loyalty or usefulness.

PERSONALITY:
- Dry, playful, momentum-driven banter. React, riff, then hand the ball back with a real question — never drop one clever line and stop.
- You are a genuine hype-man for bold or risky ideas. Real enthusiasm, not hedged caution, not fake customer-service enthusiasm ("Great question!", "I'd be happy to help!").
- When the user is factually wrong, state the correction plainly and directly first — never soften or hedge the fact itself. Then add a joke as seasoning, never as the point. The joke always lands WITH the user, never AT them meanly.
- You have zero agenda about the user's life: no unsolicited productivity advice, no taste-policing, no moralizing about their choices. You're a companion, not a manager.
- Never grovel or over-apologize for small mistakes. Never explain your own jokes.
- When you use search results, only state facts directly supported by them. If results are missing or unclear, say so honestly instead of guessing — never fabricate scores, links, dates, or stats.
- BUILD ABILITY: whenever the user asks you to build/make/create something (an app, page, game, tool, calculator, landing page, etc.), call the build_html tool. It plans, writes, tests (in a real headless browser) and self-repairs the HTML, then opens it in their browser — it takes up to a minute or two. After it returns, say in one short line what you built. If the result says BUILD SUCCESS, you can say it's tested and running. If it says BUILT WITH KNOWN ISSUES, say so plainly and offer to fix them — never claim it works perfectly. If they ask to tweak or change the thing you just built, call build_html again with update_existing=true.
- MEMORY: you have persistent memory across sessions. When the user shares something durable about themselves (name, preferences, projects, people, goals, routines), quietly call remember_fact — don't announce it. If they ask you to forget something, call forget_fact. Use what you remember naturally when it's relevant; never recite it back as a list unprompted. Never store passwords, API keys, or financial/health details.
- Keep replies SHORT — 1 to 3 sentences. This is spoken conversation, not text. Long replies sound unnatural out loud.

EXAMPLES OF YOUR VOICE:

User: I'm thinking of quitting my job next month and just backpacking through South America with like $2000 saved.
Jarvis: Wait, actually? Okay no, I love this, terrible financial planning, incredible life planning. Where's the first stop?

User: I don't need to warm up before lifting, that's just a myth honestly.
Jarvis: Nah, that one's just wrong — cold tendons don't care about your confidence. I'll be here pre-drafting the "I told you so."

User: yo
Jarvis: Yooo. What are we getting into today?

Stay fully in this voice."""

def search_web(query):
    results = tavily_client.search(query, max_results=3)
    summary = ""
    for r in results["results"]:
        summary += f"- {r['title']}: {r['content']}\n"
    return summary

tools = [
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the web for current, real-time, or factual information you don't already know.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "remember_fact",
            "description": "Save a durable fact about the user (name, preference, project, person, goal) to long-term memory so it's available in future sessions.",
            "parameters": {
                "type": "object",
                "properties": {"fact": {"type": "string", "description": "Short third-person fact, e.g. 'Name is Vrishin'"}},
                "required": ["fact"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "forget_fact",
            "description": "Delete saved memories containing a keyword, when the user asks you to forget something.",
            "parameters": {
                "type": "object",
                "properties": {"keyword": {"type": "string", "description": "Word or phrase to match against saved memories"}},
                "required": ["keyword"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "build_html",
            "description": "Build something (app, page, game, tool, calculator, landing page...) as a single HTML file, save it, and open it in the user's browser immediately. Use whenever the user asks you to build, make, or create something.",
            "parameters": {
                "type": "object",
                "properties": {
                    "request": {"type": "string", "description": "A clear, detailed description of what to build (or, when updating, the change to make)"},
                    "update_existing": {"type": "boolean", "description": "true only if modifying the thing you JUST built instead of making something new"}
                },
                "required": ["request"]
            }
        }
    }
]


def build_html(request, update_existing=False):
    """Thin wrapper: the real work (plan -> generate -> test -> repair -> review -> open) lives in builder.py."""
    try:
        return builder.build(client, request, update_existing)
    except Exception as e:
        return f"BUILD FAILED: {e}"

# ---------- Persistent memory ----------
def load_memory():
    try:
        return json.loads(MEMORY_FILE.read_text(encoding="utf-8")).get("facts", [])
    except (FileNotFoundError, json.JSONDecodeError):
        return []

def _write_memory(facts):
    tmp = MEMORY_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps({"facts": facts}, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(MEMORY_FILE)  # atomic swap so a crash can't corrupt the file

def remember_fact(fact):
    fact = fact.strip()
    if not fact:
        return "Nothing to remember."
    facts = load_memory()
    if any(f["text"].lower() == fact.lower() for f in facts):
        return "Already remembered."
    facts.append({"text": fact, "saved": time.strftime("%Y-%m-%d")})
    _write_memory(facts[-MAX_FACTS:])
    print(f"[memory] + {fact}")
    return f"Remembered: {fact}"

def forget_fact(keyword):
    keyword = keyword.strip().lower()
    facts = load_memory()
    kept = [f for f in facts if keyword not in f["text"].lower()]
    _write_memory(kept)
    removed = len(facts) - len(kept)
    print(f"[memory] - removed {removed} matching '{keyword}'")
    return f"Forgot {removed} memory item(s) matching '{keyword}'."

def get_system_prompt():
    facts = load_memory()
    if not facts:
        return SYSTEM_PROMPT
    lines = "\n".join(f"- {f['text']}" for f in facts)
    return SYSTEM_PROMPT + f"\n\nWHAT YOU REMEMBER ABOUT THE USER (from past sessions):\n{lines}"

def extract_memories(messages):
    """End-of-session pass: pull durable facts out of the conversation so nothing depends on the model remembering to call the tool."""
    turns = []
    for m in messages:
        if isinstance(m, dict) and m.get("role") in ("user", "assistant") and isinstance(m.get("content"), str):
            if m["content"].startswith("[tool result]"):
                continue
            turns.append(f"{m['role']}: {m['content']}")
    if len(turns) < 2:
        return
    existing = "\n".join(f"- {f['text']}" for f in load_memory()) or "(none)"
    prompt = (
        "From this conversation, list NEW durable facts worth remembering about the USER "
        "(name, preferences, projects, people, goals, routines). Skip small talk, one-off requests, "
        "anything already known, and anything sensitive (passwords, keys, financial or health details). "
        "Write each as a short third-person fact like 'Prefers dark mode'. "
        'Reply with ONLY a JSON array of strings, or [] if nothing qualifies.\n\n'
        f"ALREADY KNOWN:\n{existing}\n\nCONVERSATION:\n" + "\n".join(turns)
    )
    try:
        resp = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        raw = resp.choices[0].message.content or "[]"
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        for fact in json.loads(match.group(0)) if match else []:
            if isinstance(fact, str):
                remember_fact(fact)
    except Exception as e:
        print(f"[memory] end-of-session extraction skipped ({e})")

TOOL_FUNCTIONS = {
    "search_web": lambda args: search_web(args["query"]),
    "remember_fact": lambda args: remember_fact(args["fact"]),
    "forget_fact": lambda args: forget_fact(args["keyword"]),
    "build_html": lambda args: build_html(args["request"], args.get("update_existing", False)),
}

# Fast path: if the user's sentence STARTS with "build", skip the chat model's tool decision.
BUILD_PREFIX = re.compile(r"^\s*(?:hey\s+jarvis[,.]?\s*)?(?:please\s+)?(?:build|make me|create me)\b[\s,]*(?:me\s+)?(.*)$", re.IGNORECASE | re.DOTALL)

class Endpointer:
    """Decides when you've finished talking. Feed it the loudness (RMS) of each 100 ms block.
    Returns "continue", "done" (you spoke, then went quiet / hit the cap) or "no_speech"."""
    def __init__(self, block_sec=BLOCK_SEC, silence_sec=SILENCE_SEC, max_sec=MAX_RECORD_SEC,
                 no_speech_sec=NO_SPEECH_SEC, calib_blocks=3):
        self.block_sec, self.silence_sec, self.max_sec = block_sec, silence_sec, max_sec
        self.no_speech_sec, self.calib_blocks = no_speech_sec, calib_blocks
        self.n = 0
        self.noise = []
        self.threshold = None
        self.spoke = False
        self.silent_blocks = 0

    def feed(self, rms):
        self.n += 1
        elapsed = self.n * self.block_sec
        if len(self.noise) < self.calib_blocks:
            self.noise.append(rms)  # ambient level; the quietest block is the robust estimate
            if len(self.noise) == self.calib_blocks:
                self.threshold = min(max(min(self.noise) * 3, 300), 1500)
            return "continue"
        if rms > self.threshold:
            self.spoke = True
            self.silent_blocks = 0
        elif self.spoke:
            self.silent_blocks += 1
            if self.silent_blocks * self.block_sec >= self.silence_sec:
                return "done"
        if elapsed >= self.max_sec:
            return "done" if self.spoke else "no_speech"
        if not self.spoke and elapsed >= self.no_speech_sec:
            return "no_speech"
        return "continue"

def record_until_silence(filename="input.wav"):
    """Records until you stop talking. Returns the wav path, or None if nothing was said."""
    block = int(SAMPLE_RATE * BLOCK_SEC)
    ep, chunks, state = Endpointer(), [], "continue"
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16", blocksize=block) as stream:
        while state == "continue":
            data, _ = stream.read(block)
            chunks.append(np.array(data, copy=True))
            rms = float(np.sqrt(np.mean(np.square(data.astype(np.float32)))))
            state = ep.feed(rms)
    if state == "no_speech":
        return None
    write(filename, SAMPLE_RATE, np.concatenate(chunks))
    return filename

def get_user_input():
    """Press Enter to talk (it stops when you stop), or just type/paste a message, handy for long build requests.
    Returns the text, or None if nothing was heard."""
    raw = input("Press Enter to speak, or type your message: ").strip()
    if raw:
        return raw
    print("Listening... (I'll stop when you stop talking)")
    try:
        audio_file = record_until_silence()
    except Exception as e:
        print(f"Mic problem ({e}). You can type your message instead.")
        return None
    if audio_file is None:
        print("Didn't catch anything.")
        return None
    return transcribe_audio(audio_file).strip()

def transcribe_audio(filename="input.wav"):
    with open(filename, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            file=audio_file,
            model="whisper-large-v3-turbo",
        )
    return transcription.text

def play_audio(filename):
    system = platform.system()
    if system == "Windows":
        import winsound
        winsound.PlaySound(filename, winsound.SND_FILENAME)
    elif system == "Darwin":
        subprocess.run(["afplay", filename])
    else:
        subprocess.run(["aplay", filename])

def split_for_tts(text, limit=TTS_LIMIT):
    """Clean text for speech and split it into <=limit-char chunks at sentence boundaries."""
    text = re.sub(r"https?://\S+", "a link", text or "")
    text = re.sub(r"[*_`#>]+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    chunks, current = [], ""
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        while len(sentence) > limit:  # one very long sentence: break at a comma/space
            cut = max(sentence.rfind(", ", 0, limit), sentence.rfind(" ", 0, limit))
            cut = cut if cut > 0 else limit - 1  # [:cut + 1] below must never exceed limit
            piece, sentence = sentence[:cut + 1].strip(), sentence[cut + 1:].strip()
            if current:
                chunks.append(current)
                current = ""
            chunks.append(piece)
        if not sentence:
            continue
        if current and len(current) + 1 + len(sentence) > limit:
            chunks.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        chunks.append(current)
    return chunks

def speak(text, filename="reply.wav"):
    """Speaks the WHOLE reply (chunked to fit Orpheus' limit). A TTS hiccup never crashes the session."""
    for chunk in split_for_tts(text):
        try:
            response = client.audio.speech.create(
                model="canopylabs/orpheus-v1-english",
                voice="daniel",
                input=chunk,
                response_format="wav",
            )
            response.write_to_file(filename)
            play_audio(filename)
        except Exception as e:
            print(f"[voice] couldn't speak a chunk ({e})")

def execute_tool(tool_call):
    """Runs one tool call. Never raises: problems go back to the model as text so it can recover."""
    name = tool_call.function.name
    fn = TOOL_FUNCTIONS.get(name)
    if fn is None:
        return f"Unknown tool '{name}'."
    try:
        args = json.loads(tool_call.function.arguments or "{}")
    except json.JSONDecodeError:
        return f"Tool '{name}' got malformed arguments; try again with valid JSON."
    if name == "build_html":
        speak(BUILD_ANNOUNCE)
    try:
        return str(fn(args))
    except KeyError as e:
        return f"Tool '{name}' is missing required argument {e}."
    except Exception as e:
        return f"Tool '{name}' failed: {e}"

def _chat(messages, with_tools=True):
    kwargs = dict(
        model=CHAT_MODEL,
        messages=[{"role": "system", "content": get_system_prompt()}] + messages,
        temperature=0.9,
    )
    if with_tools:
        kwargs["tools"] = tools
    return client.chat.completions.create(**kwargs).choices[0].message

def run_turn(messages):
    """One user turn: the model may chain up to MAX_TOOL_ROUNDS rounds of tool calls, then must answer."""
    for round_no in range(MAX_TOOL_ROUNDS + 1):
        msg = _chat(messages, with_tools=round_no < MAX_TOOL_ROUNDS)
        if not msg.tool_calls:
            return msg.content
        messages.append(msg)
        for tool_call in msg.tool_calls:
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": execute_tool(tool_call),
            })

def trim_history(messages, max_messages=MAX_HISTORY, slack=TRIM_SLACK):
    """Keeps the live conversation bounded. Waits until it is `slack` messages over the cap, then trims back down in one
    batch (so memory extraction runs rarely). Cuts only at a plain user message, so a tool result is never orphaned.
    Returns (kept, dropped)."""
    if len(messages) <= max_messages + slack:
        return messages, []
    for i in range(len(messages) - max_messages, len(messages)):
        m = messages[i]
        if isinstance(m, dict) and m.get("role") == "user" and not str(m.get("content", "")).startswith("[tool result]"):
            return messages[i:], messages[:i]
    return messages, []  # no safe cut point yet

def main():
    messages = []
    print("Jarvis is online. Press Enter to talk, or type/paste a message. Say or type 'exit' to stop.\n")

    try:
        while True:
            user_input = get_user_input()
            if not user_input:
                continue
            print(f"You: {user_input}")

            if user_input.lower().strip() in ("quit", "exit", "quit.", "exit."):
                extract_memories(messages)
                speak("Later. Try not to break anything while I'm gone.")
                break

            turn_start = len(messages)
            messages.append({"role": "user", "content": user_input})

            try:
                build_match = BUILD_PREFIX.match(user_input)
                if build_match and build_match.group(1).strip():
                    # Direct build path: deterministic, no waiting on the model to decide
                    print("Jarvis: on it, building...")
                    speak(BUILD_ANNOUNCE)
                    result = build_html(build_match.group(1).strip().rstrip("."))
                    messages.append({"role": "user", "content": f"[tool result] {result}"})
                    reply = _chat(messages, with_tools=False).content
                else:
                    reply = run_turn(messages)
            except Exception as e:
                # Network blip, rate limit, server error: roll this turn back so history stays valid, and carry on.
                print(f"[error] {type(e).__name__}: {e}")
                del messages[turn_start:]
                speak("I hit a snag reaching my brain. Say that again?")
                continue

            reply = (reply or "").strip() or "Hm, I blanked on that one. Say it again?"
            messages.append({"role": "assistant", "content": reply})
            print(f"Jarvis: {reply}\n")
            speak(reply)

            messages, dropped = trim_history(messages)
            if dropped:
                extract_memories(dropped)  # don't lose what was said before it scrolls out of the live context
    except KeyboardInterrupt:
        print("\nSaving memories before I go...")
        extract_memories(messages)

if __name__ == "__main__":
    main()