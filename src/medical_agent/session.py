"""Session identity helpers for checkpoint isolation."""

import re

_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


def checkpoint_thread_id(actor_id: str, thread_id: str) -> str:
    """Create an unambiguous internal checkpoint key for one actor and thread."""

    if not _SAFE_ID.fullmatch(actor_id):
        raise ValueError("actor_id 只能包含字母、数字、点、下划线和连字符，最长 64 位")
    if not _SAFE_ID.fullmatch(thread_id):
        raise ValueError("thread_id 只能包含字母、数字、点、下划线和连字符，最长 64 位")

    return f"actor={actor_id}::thread={thread_id}"
