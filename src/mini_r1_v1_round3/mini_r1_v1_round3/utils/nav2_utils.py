"""Nav2 cancel-race guard utility."""
import time
from nav2_simple_commander.robot_navigator import TaskResult


def safe_cancel_and_go(navigator, new_pose, logger, cancel_timeout_s: float = 2.0,
                       settle_s: float = 0.3) -> bool:
    """Cancel current Nav2 task, wait until it's actually CANCELED (with timeout),
    then issue new goal. Returns False if cancel timed out but issues goal anyway.
    """
    cancel_ok = True
    try:
        if not navigator.isTaskComplete():
            logger.info("[NAV_UTIL] Prior task active, cancelling...")
            navigator.cancelTask()
            deadline = time.monotonic() + cancel_timeout_s
            cancelled = False
            while time.monotonic() < deadline:
                if navigator.isTaskComplete() and navigator.getResult() == TaskResult.CANCELED:
                    cancelled = True
                    break
                time.sleep(0.05)
            if not cancelled:
                logger.warn(f"[NAV_UTIL] cancel timed out after {cancel_timeout_s}s")
                cancel_ok = False
            else:
                logger.info("[NAV_UTIL] prior task cancelled cleanly")
            time.sleep(settle_s)
    except Exception as e:
        logger.warn(f"[NAV_UTIL] cancel raised {e!r}; issuing goal anyway")
        cancel_ok = False

    try:
        p = new_pose.pose.position
        logger.info(
            f"[NAV_UTIL] Calling goToPose() → ({p.x:.2f}, {p.y:.2f}) in frame "
            f"{new_pose.header.frame_id}")
        navigator.goToPose(new_pose)
        logger.info("[NAV_UTIL] goToPose() returned (goal submitted)")
    except Exception as e:
        logger.error(f"[NAV_UTIL] goToPose failed: {e!r}")
        return False
    return cancel_ok
