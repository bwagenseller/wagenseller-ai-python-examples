"""
Keeps a process that holds decrypted secrets (a passphrase, decrypted prompts or chat history) from being copied out
of memory by ordinary means (CS-25).

protect_process_memory() does two things, both Linux-only and both one-way for the life of the process:

* prctl(PR_SET_DUMPABLE, 0) - the kernel will not write a core dump of this process, whatever apport or
  systemd-coredump are configured to do, and other processes running as the SAME user can no longer attach to it
  (ptrace) or read /proc/<pid>/mem, environ, maps and the like. root still can. It is reset by execve(), so a
  subprocess this one starts (e.g. an agent tool script) is NOT protected by it - those are not handed the secrets.
* setrlimit(RLIMIT_CORE, 0) - the classic "write a 'core' file" path is closed too, as a second layer. This one IS
  inherited by subprocesses, which is harmless.

It is meant to be called only when encryption is in use, just before the passphrase is asked for, so plaintext-only
runs behave exactly as before (and stay debuggable).
"""
import ctypes
import resource
import sys
from typing import Callable

# From <linux/prctl.h>; stable kernel ABI.
PR_GET_DUMPABLE = 3
PR_SET_DUMPABLE = 4


def protect_process_memory(log_func: Callable[[str], None] = print) -> bool:
    """
    Makes the current process non-dumpable and sets its core file size limit to 0.

    Failures are reported through log_func rather than raised: the protection is defence in depth, and refusing to
    start because of it would be worse than running without it.

    Args:
        log_func: Where to report a step that could not be applied (usually print or a logger method).

    Returns:
        bool: True if both steps were applied, False if either one was not (or this is not Linux).
    """
    if not sys.platform.startswith("linux"):
        log_func("protect_process_memory: not Linux; core dumps and same-user memory reads are NOT blocked.")
        return False

    ok = True
    try:
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    except (ValueError, OSError) as e:
        log_func(f"protect_process_memory: could not set the core file limit to 0: {e}")
        ok = False

    try:
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(PR_SET_DUMPABLE, 0, 0, 0, 0) != 0:
            log_func(f"protect_process_memory: prctl(PR_SET_DUMPABLE, 0) failed: errno {ctypes.get_errno()}")
            ok = False
    except (OSError, AttributeError) as e:
        log_func(f"protect_process_memory: prctl is unavailable: {e}")
        ok = False
    return ok


def is_process_dumpable() -> bool:
    """
    True if the kernel would still dump this process (or allow same-user ptrace). Used by the tests; also handy
    for checking a running app by hand.
    """
    libc = ctypes.CDLL(None, use_errno=True)
    return libc.prctl(PR_GET_DUMPABLE, 0, 0, 0, 0) != 0
