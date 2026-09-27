#!/usr/bin/env python3
"""CS-21: a LIVE conversation with the tool server using the real script tools, on the GPU.

What it exercises that nothing else does
----------------------------------------
One continuous session (the always-on use case), real model, real tools:

1. a weather question      -> get_weather runs directly (trusted output, not quarantined);
2. "fetch this page"       -> fetch_url is untrusted, so in 'auto' mode the main model must
                              DELEGATE; a worker fetches and answers; the answer goes to the
                              user and the quarantine store, the main model sees only a marker;
3. "what did it say again" -> recall (or a new delegate) - never the main model reading it;
4. a calculator question   -> local work still available after the web lookup (no lockout).

After the conversation it checks the quarantine held: no worker answer appears in the main
model's messages or the chat history. Every raw generation is recorded in --out.

With --turns weather it holds a four-turn weather conversation instead, to see whether the model
picks get_weather's hourly or daily detail for itself and answers amount questions from the
forecast amounts.

Usage (llama env):
    python live_script_tools_smoke.py --arch qwen35moe --out /tmp/live_tools.json [--turns weather]
"""
import argparse
import json
import os
import shutil
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
# testlib finds the library under test and this machine's paths (tests/README.md).
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
from testlib import settings  # noqa: E402
settings.use_src()
settings.use_area("llm", "llama", "streams")   # golden_response: the shared config builder
ROOT = settings.REPO_ROOT                                        # the tool scripts live under ROOT/scripts

import golden_response as GR                                   # noqa: E402
from amadeo_utils.ai.llm.llama.ToolStream import ToolStream    # noqa: E402
from live_tool_smoke import MODELS                             # noqa: E402

AGENT_TOOLS_PYTHON = settings.get("agent_tools_python")
CONFIGS = settings.get("ai_tools_config_dir")                  # the tools' own configs: outside the repo
TURN_SETS = {
    # The original conversation: every tool, the delegate/quarantine path, and recall.
    "default": [
        "Search the web for what the National Weather Service API is and summarise it in two sentences.",
        "What's the weather forecast for ZIP code 10001 today?",
        "Fetch https://example.com and tell me what that page is for.",
        "What did that page say again?",
        "What is 48 times 1375?",
    ],
    # get_weather's hourly mode and precipitation amounts (2026-09-25): does each model pick
    # detail=hourly when the question is about hours, daily when it is about days, and answer
    # amount questions from the amounts rather than from the chance of rain? No location given,
    # so every call should fall back to home.
    "weather": [
        "Will it rain here in the next few hours? Give me the hour-by-hour.",
        "How much rain are we expecting over the next day or so?",
        "What's the forecast for the rest of the week?",
        "What will the temperature be around 6 PM tonight?",
    ],
    # get_grades (2026-09-25): does the model call it unprompted, answer from it, and - after private data in
    # 'auto' mode - still reach the web only through a delegate worker?
    "grades": [
        "How are the grades looking this trimester? Is anything missing?",
        "Which assignments are missing, and for which classes?",
        "Search the web for a few practical tips on helping a student catch up on missing homework.",
    ],
    # get_datetime (2026-09-26): local default, UTC, the everyday "Eastern"/"EST" names (EST used to answer an hour
    # off in summer), and a city.
    "time": [
        "What time is it?",
        "What time is it in UTC?",
        "What time is it in Eastern time?",
        "And in EST?",
        "What time is it in London?",
    ],
    # The main model's reply after a delegate (2026-09-26): Gemma 4 told a voice user "I cannot see the trending topics
    # myself, but the answer has been provided to you above." Every turn here needs the web, so each should delegate;
    # what matters is the main model's own reply under the worker's answer - a pointer (dropped by the code), an
    # apology for not seeing it, or a second answer from memory.
    "delegate": [
        "What is the top headline on Twitter right now?",
        "Search the web for today's top news story.",
        "Look up who won the most recent Super Bowl.",
        "Search the web for the current price of Bitcoin.",
    ],
    # Worker context (2026-09-25): force the path that overflowed an 8K window - a worker reading several pages.
    "research": [
        "Search the web for practical tips on helping a student catch up on missing homework, and read at least "
        "two of the pages you find in full before answering.",
    ],
}
# --persona: a stand-in for a character prompt like the voice clients choose. Wordy, warm and addressing the user by
# name - the setting in which Gemma 4 apologised for not seeing a worker's answer (2026-09-26). The
# '##...@@NAME@@...##' line is filled from player_name.
PERSONA_ID = "wordy_host"
PERSONA_PLAYER = "Kevin"
PERSONA_PROMPT = (
    "You are a warm, erudite and rather wordy household assistant with the manner of a genteel radio host. You speak "
    "in full, gracious sentences, enjoy a well-turned phrase, and are always candid with the person you are helping. "
    "##The person you are speaking with is @@NAME@@; address them by name. ##"
    "Keep your replies conversational and suitable for being read aloud."
)
# Tools only some turn sets use: (tool name, script, python, config).
EXTRA_TOOLS = {
    "grades": [("get_grades", "scripts/ai/ai-tools/get_grades/get_grades.py",
                settings.get("bash_python"), "get_grades.json")],
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", required=True, choices=sorted(MODELS))
    parser.add_argument("--out", required=True)
    parser.add_argument("--turns", default="default", choices=sorted(TURN_SETS), help="which conversation to hold")
    parser.add_argument("--ctx", type=int, default=8192, help="the context window (max_context_tokens)")
    parser.add_argument("--spoken", action="store_true",
                        help="open a spoken session (spoken_response=True), as the voice server does")
    parser.add_argument("--worker-notes", choices=("off", "auto", "always"),
                        help="worker note-taking mode (default: the server's own default)")
    parser.add_argument("--persona", action="store_true",
                        help=f"use the stand-in character prompt '{PERSONA_ID}' and call the user {PERSONA_PLAYER}")
    parser.add_argument("--flash-q8", action="store_true",
                        help="flash attention + q8_0 KV cache, whatever the model's usual settings")
    args = parser.parse_args()
    turns = TURN_SETS[args.turns]

    model, extra = MODELS[args.arch]
    model = settings.model_path(model)
    workdir = tempfile.mkdtemp(prefix="cs21-live-tools-")
    try:
        config = GR.build_args(model, args.arch, args.ctx, workdir, workdir)
        config.update(extra)
        if args.flash_q8:
            config.update({"flash_attn": True, "kv_cache_type": "q8_0"})
        if args.worker_notes:
            config["worker_notes"] = args.worker_notes
        config.update({
            "generating_gpu_layers": -1, "embedding_gpu_layers": -1, "max_response_tokens": 768,
            "tools_allowed": ["get_datetime", "calculator", "get_weather", "fetch_url", "web_search"], "tool_mode": "auto",
            "script_tools": [
                {"script": os.path.join(ROOT, "scripts/ai/ai-tools/get_weather/get_weather.py"),
                 "python": AGENT_TOOLS_PYTHON, "config": os.path.join(CONFIGS, "get_weather.json")},
                {"script": os.path.join(ROOT, "scripts/ai/ai-tools/fetch_url/fetch_url.py"),
                 "python": AGENT_TOOLS_PYTHON, "config": os.path.join(CONFIGS, "fetch_url.json")},
                {"script": os.path.join(ROOT, "scripts/ai/ai-tools/web_search/web_search.py"),
                 "python": AGENT_TOOLS_PYTHON, "config": os.path.join(CONFIGS, "web_search.json")},
            ],
        })
        for name, script, python, tool_config in EXTRA_TOOLS.get(args.turns, []):
            config["tools_allowed"].append(name)
            config["script_tools"].append({"script": os.path.join(ROOT, script), "python": python,
                                           "config": os.path.join(CONFIGS, tool_config)})
        stream = ToolStream(config)

        generations = []
        real = stream.llm_generator.create_chat_completion

        def recording(**kwargs):
            response = real(**kwargs)
            is_worker = kwargs["messages"][0]["content"].startswith(ToolStream.WORKER_SYSTEM_MESSAGE)
            generations.append({"worker": is_worker,
                                "offered": sorted(t["function"]["name"] for t in kwargs.get("tools") or []),
                                "messages": kwargs["messages"],
                                "raw": response["choices"][0]["message"]["content"]})
            return response
        stream.llm_generator.create_chat_completion = recording

        if args.persona:
            with open(os.path.join(workdir, PERSONA_ID + ".txt"), "w", encoding="utf-8") as fh:
                fh.write(PERSONA_PROMPT)
            stream.create_session("live", "live", args.spoken, False, False, system_prompt_id=PERSONA_ID,
                                  player_name=PERSONA_PLAYER)
        else:
            stream.create_session("live", "live", args.spoken, False, False)
        session = stream.get_session("live")
        out = {"arch": args.arch, "persona": args.persona, "spoken": args.spoken, "turns": []}
        for prompt in turns:
            start = len(generations)
            turn_started = time.monotonic()
            response = stream.get_response({"sessionID": "live", "command": "request", "user_request": prompt})
            result = session.get("last_result")
            out["turns"].append({
                "prompt": prompt, "reply": response.get("response"), "message": response.get("message"),
                "executed": result.executed if result else None, "refused": result.refused if result else None,
                "shown_to_user": result.shown_to_user if result else None,
                "main_answer": result.answer if result else None,   # the main model's own reply (after the pointer filter)
                "generations": [{k: g[k] for k in ("worker", "offered", "raw")} for g in generations[start:]],
                "seconds": round(time.monotonic() - turn_started, 1),
                "notes_taken": result.notes_taken if result else None,
            })
            print(f"{args.arch} | {prompt[:45]:45} | ran {result.executed if result else '?'} | "
                  f"{len(generations) - start} gen(s) ({sum(g['worker'] for g in generations[start:])} worker) | "
                  f"notes {result.notes_taken if result else '?'} | {out['turns'][-1]['seconds']}s", flush=True)

        # The quarantine must have held: no worker answer in anything the MAIN model was sent, or in history.
        worker_answers = [a for t in out["turns"] for a in (t["shown_to_user"] or []) if a.strip()]
        main_text = json.dumps([g["messages"] for g in generations if not g["worker"]])
        history_text = json.dumps(session["chat_history"])
        leaked = [a[:60] for a in worker_answers if a[:60] in main_text or a[:60] in history_text]
        out["quarantine"] = {"worker_answers": len(worker_answers), "leaked_into_main_or_history": leaked,
                             "history": session["chat_history"],
                             "store": {k: v.task for k, v in session["quarantine"].records.items()}}
        print(f"quarantine: {len(worker_answers)} worker answer(s) shown to the user; leaked into main/history: {leaked or 'none'}")
        stream.remove_session("live")
        stream.cleanup()
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False, default=str)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
