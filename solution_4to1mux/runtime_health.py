"""Fault classification; do not interpret action-server discovery as activation."""


def is_controller_deactivation(message):
    text = str(message).casefold()
    return any(fragment in text for fragment in (
        'deactivate transition', 'deactivated', 'controller is not active',
        'controller is inactive', 'controller not active', 'controller inactive',
    ))


def sample_age(now, stamp):
    return None if stamp is None else now-stamp


def current_transform(now, stamp, maximum_age):
    """The odom/base transform must be dynamic and close to the robot clock."""
    return stamp > 0 and -0.10 <= now-stamp <= maximum_age
