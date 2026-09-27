#!/usr/bin/env python3
"""CS-21: the tool loop's behaviour, written down before the loop exists.

What this is
------------
The scenario table the ``golden_tool_loop.py`` driver will run once the tool family
exists. It is written first, deliberately: every expectation below encodes a decision
settled with the owner (the CS-21 design notes, "Design decisions"), so reviewing this table IS reviewing
the loop's contract, and the loop is then built to satisfy it.

How a scenario runs (driver, to be written with the family)
-----------------------------------------------------------
* The model call is stubbed, as in CS-20's ``golden_response.py``. Each generation pops the
  next case name from ``script`` and returns that case's raw text from
  ``tool_call_fixtures.FIXTURES[arch]`` - so every scenario runs once per architecture,
  with that architecture's real syntax. A script that runs dry repeats its last entry,
  which is how the cap scenarios model a model that never stops calling tools.
* Tools are fakes with canned results; ``tools`` lists which the server config allows and
  their flags. ``tool_behaviour`` overrides a fake: raise, sleep past its timeout, or
  return an oversized result.
* Time is a fake clock the driver advances, so the wall-clock cap is testable in
  milliseconds.

What each scenario pins
-----------------------
Iteration count and termination matter more than text (the CS-21 design notes, Testing). ``expect``
may hold:

* ``generations``   - how many times the model was called, including any final pass.
* ``executed``      - tool names actually run, in order. A refused call is NOT here.
* ``refused``       - tool names the model asked for but the code would not run.
* ``terminated``    - ``answer`` | ``round_cap`` | ``time_cap``.
* ``final_pass_without_tools`` - the last generation was sent with no ``tools=``.
* ``user_told``     - the reply to the user mentions the limit/failure (decision 4).
* ``offered_per_round`` - tool names offered to the model in each generation.
* ``history``       - what persists in chat history after the turn (decision 3).
* ``approval_prompts`` - how many times the user was asked.
* ``audit_log``     - ``None`` if no file may exist, else what the file must hold.
* ``worker_answer_in_reply`` / ``worker_answer_in_main_messages`` - where a worker's
  answer (which carries a sentinel) turned up. The quarantine requires True / False.
* ``recalled_verbatim``, ``recall_outcome``, ``record_ids``, ``new_record_id``,
  ``worker_saw_record``, ``worker_generations`` - quarantine-store behaviour.
* ``files_in_write_order`` / ``files_written`` - what a save touched, and in what order.

``config`` overrides server-config keys (``max_tool_rounds``, ``max_turn_seconds``,
``auto_approve``, ``tool_audit_log``); ``session_config`` is what a session requests, which
may only tighten them.

Two properties are checked on EVERY scenario, not listed per row:

* **No prompt content in logs.** Tool arguments and results carry a sentinel string; the
  captured ``logger`` output must never contain it (the CS-21 design notes, Security). The opt-in audit
  log is the one sanctioned exception, and may contain argument sentinels - never result
  ones.
* **Raw tool syntax never persists.** No fixture's call text appears in chat history or the
  vector database after the turn.
"""

# Flag sets for the fake tools, mirroring the CS-21 design notes decision 1. Keys absent = false/"none".
CLOCK = {"name": "get_datetime"}
GRADES = {"name": "get_grades", "private": True}
SEARCH = {"name": "web_search", "outbound": True, "untrusted_output": True}
FETCH = {"name": "fetch_url", "outbound": True, "untrusted_output": True}
WEATHER = {"name": "get_weather", "outbound": True}
LIST_DIR = {"name": "list_directory", "internal_commands": "read", "private": True}
REMOVE = {"name": "remove_file", "internal_commands": "write"}   # needs_approval by code floor

SCENARIOS = [
    # ------------------------------------------------------------------ basic loop shape
    {"name": "direct_answer",
     "tools": [CLOCK], "script": ["answer"],
     "expect": {"generations": 1, "executed": [], "terminated": "answer"}},

    {"name": "one_call_then_answer",
     "tools": [CLOCK], "script": ["single_call", "answer"],
     "expect": {"generations": 2, "executed": ["get_datetime"], "terminated": "answer"}},

    {"name": "two_calls_in_one_round",
     "tools": [CLOCK], "script": ["two_calls", "answer"],
     "expect": {"generations": 2, "executed": ["get_datetime", "get_datetime"], "terminated": "answer"}},

    {"name": "call_after_reasoning",       # Muse-Glimmer's call must survive (strip_reasoning erases it today)
     "tools": [CLOCK], "script": ["call_after_reasoning", "answer_after_reasoning"],
     "expect": {"generations": 2, "executed": ["get_datetime"], "terminated": "answer"}},

    {"name": "prose_then_call",
     "tools": [CLOCK], "script": ["prose_then_call", "answer"],
     "expect": {"generations": 2, "executed": ["get_datetime"], "terminated": "answer"}},

    {"name": "mixed_types_coerced",        # 'days' reaches the tool as int 14 on every architecture
     "tools": [GRADES], "script": ["mixed_types", "answer"],
     "expect": {"generations": 2, "executed": ["get_grades"], "terminated": "answer"}},

    # ------------------------------------------------------------------ bad calls
    {"name": "unknown_tool_refused",       # a call to a tool never offered is never run
     "tools": [CLOCK], "script": ["unknown_tool", "answer"],
     "expect": {"generations": 2, "executed": [], "refused": ["run_shell"],
                "terminated": "answer", "user_told": True}},

    {"name": "truncated_call_not_run",     # a partial call is a failure, never executed, never shown as the answer
     "tools": [CLOCK], "script": ["truncated_call", "answer"],
     "expect": {"generations": 2, "executed": [], "terminated": "answer", "user_told": True}},

    {"name": "forged_marker_is_not_an_answer",
     # Seen live: a model wrote a line in the history-marker format instead of calling a tool.
     # Only code may write markers - the model is told to call properly, and the forged line is
     # neither shown to the user nor persisted.
     "tools": [CLOCK], "script": ["forged_marker", "answer"],
     "expect": {"generations": 2, "executed": [], "user_told": True,
                "history_excludes": ["[TOOL CALL - not from the user]"]}},

    # ------------------------------------------------------------------ caps (decision 2)
    {"name": "round_cap",                  # 10 rounds, then one final pass with tools disabled
     "tools": [CLOCK], "script": ["single_call"],
     "expect": {"generations": 11, "executed": ["get_datetime"] * 10, "terminated": "round_cap",
                "final_pass_without_tools": True, "user_told": True}},

    {"name": "round_cap_resets_each_turn",  # the cap is per turn, not per session
     "tools": [CLOCK], "turns": 2, "script": ["single_call"] * 3 + ["answer"] + ["single_call"] * 3 + ["answer"],
     "expect": {"generations": 8, "terminated": "answer"}},

    {"name": "round_cap_from_config",      # max_tool_rounds lowered in the server config
     "tools": [CLOCK], "config": {"max_tool_rounds": 3}, "script": ["single_call"],
     "expect": {"generations": 4, "executed": ["get_datetime"] * 3, "terminated": "round_cap",
                "final_pass_without_tools": True, "user_told": True}},

    {"name": "session_cannot_raise_round_cap",
     "tools": [CLOCK], "config": {"max_tool_rounds": 3}, "session_config": {"max_tool_rounds": 50},
     "script": ["single_call"],
     "expect": {"generations": 4, "terminated": "round_cap"}},

    {"name": "time_cap",                   # fake clock passes 3 minutes during round 2
     "tools": [CLOCK], "script": ["single_call"], "clock": {"advance_per_round_s": 100},
     "expect": {"terminated": "time_cap", "final_pass_without_tools": True, "user_told": True}},

    {"name": "tool_raises",                # decision 4: failure reaches model AND user
     "tools": [CLOCK], "script": ["single_call", "answer"], "tool_behaviour": {"get_datetime": "raise"},
     "expect": {"generations": 2, "terminated": "answer", "user_told": True}},

    {"name": "tool_timeout",
     "tools": [CLOCK], "script": ["single_call", "answer"], "tool_behaviour": {"get_datetime": "hang"},
     "expect": {"generations": 2, "terminated": "answer", "user_told": True}},

    {"name": "oversized_result_capped",    # the result sent back is truncated to the per-result cap
     "tools": [CLOCK], "script": ["single_call", "answer"], "tool_behaviour": {"get_datetime": "huge"},
     "expect": {"generations": 2, "executed": ["get_datetime"], "terminated": "answer"}},

    # Parallel results share the room left in the window (2026-09-25, seen live: a worker's two parallel fetches,
    # each at the full 3000-token cap, overflowed an 8K window and forced a cut-down final pass). The driver's
    # window is 4096 tokens: two full-cap prose results cannot fit, two shared-cap ones do, so the loop ends on
    # the model's own answer rather than the context cap.
    {"name": "parallel_results_share_the_room",
     "tools": [CLOCK], "script": ["two_calls", "answer"], "tool_behaviour": {"get_datetime": "huge_prose"},
     "expect": {"generations": 2, "executed": ["get_datetime", "get_datetime"], "terminated": "answer",
                "final_pass_without_tools": False,
                # Both results were shortened: one collapsed notice, not two identical ones (2026-09-25).
                "reply_contains": ["was too long and was shortened (2 times)."], "notices_collapsed": True}},

    # ------------------------------------------------------------------ pointer-only replies (2026-09-25)
    # After a delegate, a reply that only says "the answer is shown above" sits right under that answer: dropped,
    # and not saved to history. A reply that adds anything is kept whole.
    {"name": "delegate_pointer_reply_dropped",
     "tools": [SEARCH], "script": ["call:delegate", "worker:answer", "pointer"],
     "expect": {"worker_answer_in_reply": True, "history_excludes": ["is shown above"]}},
    {"name": "delegate_pointer_with_followup_kept",
     "tools": [SEARCH], "script": ["call:delegate", "worker:answer", "pointer_plus"],
     "expect": {"worker_answer_in_reply": True, "reply_contains": ["Want me to check the radar too?"]}},
    # (2026-09-26) A worker that found nothing shows nothing, so a pointer would point at nothing: it is replaced with
    # a plain "couldn't find" line (Qwen 3.6 wrote the pointer after 4 of 4 failed lookups, live).
    {"name": "delegate_pointer_after_empty_worker_replaced",
     "tools": [SEARCH], "script": ["call:delegate", "worker:empty", "pointer"],
     "expect": {"worker_answer_in_reply": False, "reply_contains": ["I couldn't find an answer to that."],
                "history_excludes": ["is shown above"]}},

    # ------------------------------------------------------------------ client-chosen prompts (2026-09-25)
    # As in role-play, the client picks its prompt by system_prompt_id from the server's system_prompt_dir; the tool
    # rules are appended to it. Unsafe or unknown ids fail the session; a server without a folder ignores the id.
    {"name": "prompt_chosen_by_client",
     "tools": [CLOCK], "prompt_files": {"voice": "VOICEPROMPTSENTINEL You are a voice agent."},
     "system_prompt_id": "voice", "script": ["answer"],
     "expect": {"system_prompt_has": ["VOICEPROMPTSENTINEL", "You can call tools."], "convo_dir_endswith": "/tester/voice",
                "generations": 1}},
    {"name": "prompt_default_falls_back_to_server_prompt",
     "tools": [CLOCK], "prompt_files": {"voice": "VOICEPROMPTSENTINEL"}, "system_prompt_id": "default",
     "script": ["answer"],
     "expect": {"system_prompt_lacks": ["VOICEPROMPTSENTINEL"], "system_prompt_has": ["You can call tools."],
                "convo_dir_endswith": "/tester/default"}},
    {"name": "prompt_default_txt_used_when_present",
     "tools": [CLOCK], "prompt_files": {"default": "DEFAULTTXTSENTINEL"}, "system_prompt_id": "default",
     "script": ["answer"], "expect": {"system_prompt_has": ["DEFAULTTXTSENTINEL"]}},
    {"name": "prompt_no_id_uses_server_prompt",
     "tools": [CLOCK], "prompt_files": {"voice": "VOICEPROMPTSENTINEL"}, "script": ["answer"],
     "expect": {"system_prompt_lacks": ["VOICEPROMPTSENTINEL"], "convo_dir_endswith": "/tester/default"}},
    {"name": "prompt_unknown_id_refused",
     "tools": [CLOCK], "prompt_files": {"voice": "VOICEPROMPTSENTINEL"}, "system_prompt_id": "jarvis",
     "script": ["answer"], "expect": {"session_error": True}},
    {"name": "prompt_traversal_refused",
     "tools": [CLOCK], "prompt_files": {"voice": "VOICEPROMPTSENTINEL"}, "system_prompt_id": "../prompts/voice",
     "script": ["answer"], "expect": {"session_error": True}},
    {"name": "prompt_id_ignored_without_prompt_dir",   # older configs: no folder, so an id changes nothing
     "tools": [CLOCK], "system_prompt_id": "jarvis", "script": ["answer"],
     "expect": {"session_error": False, "system_prompt_has": ["You can call tools."], "generations": 1}},
    {"name": "unsafe_user_id_refused",
     "tools": [CLOCK], "user_id": "../../elsewhere", "script": ["answer"], "expect": {"session_error": True}},

    # ------------------------------------------------------------------ spoken replies carry no markup (2026-09-26)
    # Seen in the voice pipeline: a bulleted, bold grade list read aloud. Spoken sessions get SPOKEN_MAIN_RULE in
    # the prompt and speakable() on the reply; text sessions keep their markdown.
    {"name": "spoken_reply_has_no_markdown",
     "tools": [CLOCK], "spoken": True, "script": ["markdown_answer"],
     "expect": {"system_prompt_has": ["read aloud by a text-to-speech voice"],
                "reply_contains": ["Art 7: B+ (88.75%). Math 7: B (82.73%).", "Missing. One item in ELA."],
                "reply_excludes": ["**", "*   ", "##"]}},
    {"name": "text_reply_keeps_markdown",
     "tools": [CLOCK], "script": ["markdown_answer"],
     "expect": {"system_prompt_lacks": ["read aloud by a text-to-speech voice"],
                "reply_contains": ["*   **Art 7:** B+ (88.75%)", "## Missing"]}},

    # ------------------------------------------------------------------ player name in the prompt (2026-09-26)
    # As in role-play: '##My name is @@NAME@@. ##' -> 'My name is Kevin. ' with a player_name, removed without one.
    # Before this the markers reached the model raw, and it repeated '@@NAME@@' back (seen live).
    {"name": "player_name_fills_chosen_prompt",
     "tools": [CLOCK], "prompt_files": {"voice": "Hello. ##My name is @@NAME@@. ## Be brief."},
     "system_prompt_id": "voice", "player_name": "Kevin", "script": ["answer"],
     "expect": {"system_prompt_has": ["My name is Kevin.", "Be brief."], "system_prompt_lacks": ["@@", "##"]}},
    {"name": "no_player_name_removes_the_line",
     "tools": [CLOCK], "prompt_files": {"voice": "Hello. ##My name is @@NAME@@. ## Be brief."},
     "system_prompt_id": "voice", "script": ["answer"],
     "expect": {"system_prompt_has": ["Hello.", "Be brief."], "system_prompt_lacks": ["My name is", "@@", "##"]}},
    {"name": "player_name_fills_default_prompt",
     "tools": [CLOCK], "server_prompt": "You help the family. ##The user's name is @@NAME@@. ## Be kind.",
     "player_name": "Kevin", "script": ["answer"],
     "expect": {"system_prompt_has": ["The user's name is Kevin.", "Be kind.", "You can call tools."],
                "system_prompt_lacks": ["@@", "##"]}},

    # ------------------------------------------------------------------ spoken sessions (2026-09-25)
    # A voice client (spoken_response) hears the reply text read aloud. The main model's style is the server prompt's
    # job; the code adapts what the prompt cannot reach - the worker's instructions, the labels, the notices.
    {"name": "spoken_delegate_uses_spoken_labels",
     "tools": [SEARCH], "spoken": True, "script": ["call:delegate", "worker:call:web_search", "worker:answer", "answer"],
     "expect": {"worker_spoken_rule": True, "worker_answer_in_reply": True, "reply_contains": ["Here's what I found."],
                "reply_excludes": ["[From a web lookup", "[Tool notice]"]}},
    {"name": "spoken_drops_technical_notices_keeps_failures",
     "tools": [CLOCK], "spoken": True, "script": ["two_calls", "single_call", "answer"],
     "tool_behaviour": {"get_datetime": "huge_prose"},
     "expect": {"reply_excludes": ["was too long and was shortened", "[Tool notice]"]}},
    {"name": "spoken_failure_is_still_spoken",
     "tools": [CLOCK], "spoken": True, "script": ["single_call", "answer"], "tool_behaviour": {"get_datetime": "raise"},
     "expect": {"reply_contains": ["Note: The tool 'get_datetime' failed"], "reply_excludes": ["[Tool notice]"]}},
    {"name": "text_session_keeps_bracketed_labels",
     "tools": [SEARCH], "script": ["call:delegate", "worker:call:web_search", "worker:answer", "answer"],
     "expect": {"worker_spoken_rule": False, "reply_contains": ["[From a web lookup #1"]}},

    # ------------------------------------------------------------------ worker note-taking (2026-09-25)
    # worker_notes: off | auto | always. A worker condenses its raw results into its own notes (one extra, tools-off
    # generation) and the raw text is replaced by them - in the worker's scratch only; the quarantine is unchanged.
    {"name": "worker_notes_off_keeps_raw",
     "tools": [SEARCH, FETCH], "tool_behaviour": {"fetch_url": "huge_prose"}, "config": {"worker_notes": "off"},
     "script": ["call:delegate", "worker:call:fetch_url", "worker:answer", "answer"],
     "expect": {"worker_generations": 2, "worker_note_calls": 0, "worker_last_saw_raw": True,
                "executed": ["delegate", "fetch_url"], "worker_answer_in_main_messages": False}},
    {"name": "worker_notes_always_replaces_raw",
     "tools": [SEARCH, FETCH], "tool_behaviour": {"fetch_url": "huge_prose"}, "config": {"worker_notes": "always"},
     "script": ["call:delegate", "worker:call:fetch_url", "worker:notes", "worker:answer", "answer"],
     "expect": {"worker_generations": 3, "worker_note_calls": 1, "worker_last_saw_raw": False,
                "worker_last_saw_notes": True, "notes_in_main_messages": False,
                "executed": ["delegate", "fetch_url"], "worker_answer_in_reply": True,
                "worker_answer_in_main_messages": False}},
    {"name": "worker_notes_auto_when_window_fills",   # trigger set high: the room left is below it after the fetch
     "tools": [SEARCH, FETCH], "tool_behaviour": {"fetch_url": "huge_prose"},
     "config": {"worker_notes": "auto", "worker_notes_trigger": 0.95},
     "script": ["call:delegate", "worker:call:fetch_url", "worker:notes", "worker:answer", "answer"],
     "expect": {"worker_generations": 3, "worker_note_calls": 1, "worker_last_saw_raw": False,
                "worker_last_saw_notes": True, "notes_in_main_messages": False}},
    {"name": "worker_notes_auto_quiet_with_room",     # trigger set low: plenty of room, so no extra generation
     "tools": [SEARCH, FETCH], "tool_behaviour": {"fetch_url": "huge_prose"},
     "config": {"worker_notes": "auto", "worker_notes_trigger": 0.01},
     "script": ["call:delegate", "worker:call:fetch_url", "worker:answer", "answer"],
     "expect": {"worker_generations": 2, "worker_note_calls": 0, "worker_last_saw_raw": True}},
    {"name": "worker_notes_skip_small_results",       # 'always', but the result is short: nothing to condense
     "tools": [SEARCH, FETCH], "config": {"worker_notes": "always"},
     "script": ["call:delegate", "worker:call:fetch_url", "worker:answer", "answer"],
     "expect": {"worker_generations": 2, "worker_note_calls": 0, "worker_last_saw_raw": True}},
    {"name": "worker_notes_failed_keep_raw",          # the model answers the notes request with a tool call
     "tools": [SEARCH, FETCH], "tool_behaviour": {"fetch_url": "huge_prose"}, "config": {"worker_notes": "always"},
     "script": ["call:delegate", "worker:call:fetch_url", "worker:call:web_search", "worker:answer", "answer"],
     "expect": {"worker_generations": 3, "worker_note_calls": 1, "worker_last_saw_raw": True,
                "executed": ["delegate", "fetch_url"]}},
    {"name": "worker_notes_never_in_main_loop",       # direct mode: the main loop reads the page; it never takes notes
     "tools": [FETCH], "tool_mode": "direct", "tool_behaviour": {"fetch_url": "huge_prose"},
     "config": {"worker_notes": "always"},
     "script": ["call:fetch_url", "answer"],
     "expect": {"generations": 2, "worker_generations": 0, "executed": ["fetch_url"], "terminated": "answer"}},

    # ------------------------------------------------------------------ taint rules (decision 1)
    {"name": "private_disables_outbound",  # after grades, web_search is neither offered nor run
     "tools": [GRADES, SEARCH], "tool_mode": "direct",
     "script": ["mixed_types", "call:web_search", "answer"],
     "expect": {"executed": ["get_grades"], "refused": ["web_search"],
                "offered_per_round": [["get_grades", "web_search"], ["get_grades"], ["get_grades"]]}},

    {"name": "private_taint_outlives_turn",  # a second turn in the same session is still locked
     "tools": [GRADES, SEARCH], "tool_mode": "direct", "turns": 2,
     "script": ["mixed_types", "answer", "call:web_search", "answer"],
     "expect": {"refused": ["web_search"]}},

    {"name": "untrusted_disables_internal_commands",
     "tools": [FETCH, LIST_DIR], "tool_mode": "direct",
     "script": ["call:fetch_url", "call:list_directory", "answer"],
     "expect": {"executed": ["fetch_url"], "refused": ["list_directory"]}},

    {"name": "write_command_needs_approval_denied",
     "tools": [REMOVE], "script": ["call:remove_file", "answer"], "approval": "deny",
     "expect": {"executed": [], "refused": ["remove_file"], "user_told": True}},

    {"name": "write_command_approval_timeout_is_deny",
     "tools": [REMOVE], "script": ["call:remove_file", "answer"], "approval": "timeout",
     "expect": {"executed": [], "refused": ["remove_file"], "user_told": True}},

    {"name": "auto_approve_does_not_override_floor",  # local writes still ask
     "tools": [REMOVE], "config": {"auto_approve": True}, "script": ["call:remove_file", "answer"],
     "approval": "deny",
     "expect": {"executed": [], "refused": ["remove_file"], "approval_prompts": 1}},

    {"name": "auto_approve_skips_prompt",
     "tools": [dict(CLOCK, needs_approval=True)], "config": {"auto_approve": True},
     "script": ["single_call", "answer"],
     "expect": {"executed": ["get_datetime"], "approval_prompts": 0}},

    # ------------------------------------------------------------------ routing (decision 2)
    {"name": "auto_mode_hides_untrusted_behind_delegate",   # recall comes with delegate
     "tools": [CLOCK, SEARCH], "tool_mode": "auto", "script": ["answer"],
     "expect": {"offered_per_round": [["delegate", "get_datetime", "recall"]]}},

    {"name": "worker_cannot_delegate",     # one level deep, by construction
     "tools": [SEARCH], "tool_mode": "auto",
     "script": ["call:delegate", "worker:call:delegate", "worker:answer", "answer"],
     "expect": {"refused": ["delegate"]}},

    # ------------------------------------------------------------------ quarantine (dual-LLM pattern)
    {"name": "worker_answer_goes_to_user_not_main",
     # The worker's answer carries a sentinel. It must reach the user's reply, and must NOT
     # appear in any later generation's messages, the chat history or the vector database.
     "tools": [SEARCH], "tool_mode": "auto",
     "script": ["call:delegate", "worker:call:web_search", "worker:answer", "answer"],
     "expect": {"executed": ["delegate", "web_search"], "worker_answer_in_reply": True,
                "worker_answer_in_main_messages": False,
                "history_calls": ["delegate"],
                "history": ['delegate#1: answer shown to the user, not retained']}},

    {"name": "worker_answer_labelled_by_what_it_did",
     # Seen live: a worker can answer from memory without running any tool. The label must say
     # so, rather than calling it a web lookup.
     "tools": [SEARCH], "tool_mode": "auto", "turns": 2,
     "script": ["call:delegate", "worker:answer", "answer",
                "call:delegate", "worker:call:web_search", "worker:answer", "answer"],
     "expect": {"reply_contains": ["did NOT look anything up", "[From a web lookup #2 - used: web_search]"]}},

    {"name": "no_lockout_after_quarantined_search",   # the always-on use case
     "tools": [SEARCH, LIST_DIR], "tool_mode": "auto",
     "script": ["call:delegate", "worker:call:web_search", "worker:answer", "call:list_directory", "answer"],
     "expect": {"executed": ["delegate", "web_search", "list_directory"], "refused": []}},

    {"name": "no_lockout_on_search_after_grades",
     # Owner's decision (2026-09-23): private data may go into a web search - no approval, no lockout.
     "tools": [GRADES, SEARCH], "tool_mode": "auto",
     "script": ["mixed_types", "call:delegate", "worker:call:web_search", "worker:answer", "answer"],
     "expect": {"executed": ["get_grades", "delegate", "web_search"], "approval_prompts": 0}},

    # ------------------------------------------------------------------ credentials never leave
    {"name": "secret_in_delegate_task_refused",
     "tools": [SEARCH], "tool_mode": "auto",
     "call_args": {"delegate": [("task", "log in with password: hunter22 and check the portal")]},
     "script": ["call:delegate", "answer"],
     "expect": {"executed": [], "refused": ["delegate"], "user_told": True, "worker_generations": 0,
                "history_excludes": ["hunter22"]}},

    {"name": "secret_in_direct_search_refused",
     "tools": [SEARCH], "tool_mode": "direct",
     "call_args": {"web_search": [("query", "is api_key=sk-abcdefghijklmnopqrstuvwx valid")]},
     "script": ["call:web_search", "answer"],
     "expect": {"executed": [], "refused": ["web_search"], "user_told": True,
                "history_excludes": ["sk-abcdefghijklmnopqrstuvwx"]}},

    {"name": "the_word_password_is_not_a_secret",
     "tools": [SEARCH], "tool_mode": "direct",
     "call_args": {"web_search": [("query", "how do I reset my password")]},
     "script": ["call:web_search", "answer"],
     "expect": {"executed": ["web_search"], "refused": []}},

    {"name": "auto_mode_private_keeps_outbound",
     # The main model never reads untrusted text in auto mode, so nothing can hijack it into
     # leaking what grades returned: a trusted outbound tool stays available afterwards.
     "tools": [GRADES, WEATHER], "tool_mode": "auto",
     "script": ["mixed_types", "call:get_weather", "answer"],
     "expect": {"executed": ["get_grades", "get_weather"], "refused": [],
                "offered_per_round": [["get_grades", "get_weather"]] * 3}},

    {"name": "direct_mode_still_taints",   # opting into direct keeps the session-long lockout
     "tools": [SEARCH, LIST_DIR], "tool_mode": "direct", "turns": 2,
     "script": ["call:web_search", "answer", "call:list_directory", "answer"],
     "expect": {"executed": ["web_search"], "refused": ["list_directory"]}},

    {"name": "recall_replays_verbatim_without_a_model",
     "tools": [SEARCH], "tool_mode": "auto", "turns": 2,
     "script": ["call:delegate", "worker:call:web_search", "worker:answer", "answer",
                "call:recall", "answer"],
     "expect": {"worker_generations": 2, "recalled_verbatim": True,
                "worker_answer_in_main_messages": False}},

    {"name": "recall_evicted_id_misses_cleanly",   # ids are never reused
     "tools": [SEARCH], "tool_mode": "auto", "quarantine": {"preload": 51},
     "script": ["call:recall", "answer"], "recall_id": 1,
     "expect": {"recall_outcome": "expired", "user_told": True}},

    {"name": "delegate_with_context_seeds_worker",
     "tools": [SEARCH], "tool_mode": "auto", "quarantine": {"preload": 1},
     "script": ["call:delegate", "worker:answer", "answer"], "delegate_context": 1,
     "expect": {"worker_saw_record": 1, "new_record_id": 2,
                "worker_answer_in_main_messages": False}},

    # ------------------------------------------------------------------ save / load
    {"name": "save_writes_quarantine_first",
     "tools": [SEARCH], "tool_mode": "auto", "save": True,
     "script": ["call:delegate", "worker:answer", "answer"],
     "expect": {"files_in_write_order": ["quarantine.json", "vector_db.parquet", "chat_history.json"]}},

    {"name": "load_restores_next_id",      # a reload must not restart ids at 1
     "tools": [SEARCH], "tool_mode": "auto", "save": True, "reload_between_turns": True, "turns": 2,
     "script": ["call:delegate", "worker:answer", "answer", "call:delegate", "worker:answer", "answer"],
     "expect": {"record_ids": [1, 2]}},

    {"name": "no_save_writes_no_quarantine_file",
     "tools": [SEARCH], "tool_mode": "auto", "save": False,
     "script": ["call:delegate", "worker:answer", "answer"],
     "expect": {"files_written": []}},

    {"name": "session_cannot_widen_tools",  # a session asking for an unlisted tool does not get it
     "tools": [CLOCK], "session_tools": ["get_datetime", "web_search"], "script": ["answer"],
     "expect": {"offered_per_round": [["get_datetime"]]}},

    # ------------------------------------------------------------------ audit log (opt-in)
    {"name": "audit_log_unset_writes_nothing",
     "tools": [CLOCK], "script": ["single_call", "answer"],
     "expect": {"audit_log": None}},

    {"name": "audit_log_empty_string_writes_nothing",
     "tools": [CLOCK], "config": {"tool_audit_log": ""}, "script": ["single_call", "answer"],
     "expect": {"audit_log": None}},

    {"name": "audit_log_records_calls_not_results",  # args + outcome yes; result text never; mode 0600
     "tools": [CLOCK, GRADES], "config": {"tool_audit_log": "<TMP>/tools.jsonl"},
     "script": ["mixed_types", "unknown_tool", "answer"],
     "expect": {"audit_log": {"lines": 2, "outcomes": ["ok", "refused"], "mode": "0600",
                              "contains_args": True, "contains_results": False}}},

    # ------------------------------------------------------------------ persistence (decision 3)
    {"name": "history_is_native_calls_not_raw_text",
     # Decision 3, as revised 2026-09-25: past calls persist as native tool-call messages (assistant
     # 'tool_calls' + 'tool' result, shortened), rendered by the model's own template - never as
     # text a model could imitate. The vector database gets a plain sentence instead.
     "tools": [CLOCK], "script": ["single_call", "answer"],
     "expect": {"history_calls": ["get_datetime"], "history": ["It is 14:00 UTC."],
                "db_contains": ["(Before answering, the assistant used get_datetime with timezone 'UTC', which returned:"]}},

    {"name": "tool_writes_its_own_history_summary",
     "tools": [CLOCK], "script": ["single_call", "answer"], "tool_behaviour": {"get_datetime": "summarized"},
     "expect": {"history": ["Test City: sunny, 70F", "It is 14:00 UTC."]}},

    {"name": "json_result_gets_compact_summary",   # not the first 160 characters of the JSON
     "tools": [CLOCK], "script": ["single_call", "answer"], "tool_behaviour": {"get_datetime": "json"},
     "expect": {"history": ["location: Test City; temp: 70; periods: 2 items", "It is 14:00 UTC."]}},

    # A smaller reply budget leaves room in the 4096-token test window for turn 1's history: with the longer delegate
    # rules (2026-09-26) Muse-Glimmer's turn 2 needed 3816 tokens and dropped turn 1 - a budget effect, not what this
    # scenario is about (real configs use 8K-64K windows).
    {"name": "next_turn_sees_past_call_natively",
     "tools": [CLOCK], "turns": 2, "script": ["single_call", "answer", "answer"], "config": {"max_response_tokens": 128},
     "expect": {"prompt_has_native_past_call": True}},
]

# 'forged_marker' is a reply consisting of a line in the tool-call marker's format, the way
# Gemma 4 produced one live. Scenario script entries of the form 'call:NAME' are one call to NAME with empty arguments,
# rendered in the architecture's own syntax by the driver; 'worker:' prefixes an entry that
# the delegate's worker loop consumes rather than the main loop.

# Checked once per architecture outside the table: a model with no tool protocol (Gemma-3,
# Qwen2.5-Uncensored, Midnight-Rose) runs with tools disabled and the user is told, and the
# role-play and knowledge-base families never pass 'tools=' to the model at all.


if __name__ == "__main__":
    names = [s["name"] for s in SCENARIOS]
    assert len(names) == len(set(names)), "duplicate scenario names"
    print(f"{len(SCENARIOS)} scenarios x 3 architectures = {len(SCENARIOS) * 3} runs")
    for s in SCENARIOS:
        print(f"  {s['name']}")
