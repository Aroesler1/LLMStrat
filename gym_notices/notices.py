"""
Project-local shim for `gym_notices`.

Gym prints a large deprecation banner by reading `gym_notices.notices` and
writing directly to stderr at import time. We keep the interface but provide
no notices to avoid noisy, non-actionable output in reproducible runs.
"""

notices = {}

