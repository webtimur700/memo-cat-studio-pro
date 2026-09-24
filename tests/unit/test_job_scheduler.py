from pipeline.job_scheduler import JobScheduler


def _make(max_concurrent):
    started, idle = [], []
    scheduler = JobScheduler(max_concurrent, started.append, on_idle=lambda: idle.append(True))
    return scheduler, started, idle


def test_default_one_at_a_time_rest_wait():
    scheduler, started, _ = _make(1)
    for i in range(3):
        scheduler.submit(f"j{i}")
    assert started == ["j0"]
    assert (scheduler.running_count, scheduler.waiting_count) == (1, 2)


def test_next_starts_in_fifo_order_and_idle_fires_once_at_the_end():
    scheduler, started, idle = _make(1)
    for i in range(3):
        scheduler.submit(f"j{i}")
    scheduler.job_done("j0")
    assert started == ["j0", "j1"] and idle == []
    scheduler.job_done("j1")
    assert idle == []
    scheduler.job_done("j2")
    assert started == ["j0", "j1", "j2"] and idle == [True] and scheduler.is_idle


def test_concurrency_limit_respected_and_raised_at_runtime():
    scheduler, started, _ = _make(2)
    for i in range(5):
        scheduler.submit(f"j{i}")
    assert started == ["j0", "j1"]
    scheduler.set_max_concurrent(4)
    assert started == ["j0", "j1", "j2", "j3"]
    scheduler.set_max_concurrent(1)   # идущие не прерываются, новые не стартуют, пока слотов нет
    scheduler.job_done("j0")
    assert started == ["j0", "j1", "j2", "j3"]


def test_queue_of_10000_is_cheap():
    scheduler, started, idle = _make(1)
    for i in range(10_000):
        scheduler.submit(str(i))
    for i in range(10_000):
        scheduler.job_done(str(i))
    assert len(started) == 10_000 and idle == [True]
