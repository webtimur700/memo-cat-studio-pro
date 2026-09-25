import time
from pathlib import Path

import pytest
from PySide6.QtCore import QCoreApplication

from tests.unit.test_git_updater import GIT, git, publish, repos  # noqa: F401  (фикстура repos: сервер + две копии)
from ui.widgets.update_panel import UpdatePanel
from updater.git_updater import UpdateResult, UpdateStatus


def _wait(panel, predicate, timeout=20.0):
    end = time.time() + timeout
    while time.time() < end:
        QCoreApplication.processEvents()
        if predicate():
            return True
        time.sleep(0.02)
    return False


def _panel(checker, applier=None):
    return UpdatePanel(Path("."), checker=checker, applier=applier or (lambda repo: UpdateResult(True, "ok")))


def test_nothing_runs_until_the_button_is_pressed(qapp):
    calls = []
    panel = _panel(lambda repo: calls.append("check") or UpdateStatus(branch="main", upstream="origin/main"))
    QCoreApplication.processEvents()
    assert calls == [] and panel.buttons_enabled() == (True, False) and "только по кнопке" in panel.label_text()


def test_check_shows_new_commits_and_update_needs_confirmation(qapp):
    status = UpdateStatus("main", "origin/main", behind=2, new_commits=("abc1234 исправлена очередь", "def5678 музыка по уровню"))
    applied, asked = [], []
    panel = _panel(lambda repo: status, lambda repo: applied.append(1) or UpdateResult(True, "Обновлено до abc1234. Перезапустите приложение"))
    panel.check_now()
    assert _wait(panel, lambda: panel.buttons_enabled() == (True, True))
    assert "Доступно обновление: 2" in panel.label_text() and "abc1234 исправлена очередь" in panel.label_text()

    panel.confirm = lambda st: asked.append(st) or False                    # пользователь отказался
    panel.update_now()
    assert applied == [] and asked and "отменено" in panel.label_text()

    panel.confirm = lambda st: True
    panel.update_now()
    assert _wait(panel, lambda: "Обновлено до abc1234" in panel.label_text()) and applied == [1]
    assert panel.buttons_enabled() == (True, False)


def test_uncommitted_changes_keep_the_update_button_disabled_and_explain_why(qapp):
    status = UpdateStatus("main", "origin/main", behind=1, dirty_files=("ui/main_window.py",))
    applied = []
    panel = _panel(lambda repo: status, lambda repo: applied.append(1) or UpdateResult(True, "ok"))
    panel.check_now()
    assert _wait(panel, lambda: panel.buttons_enabled()[0])
    assert panel.buttons_enabled() == (True, False)
    assert "незакоммиченные изменения" in panel.label_text() and "ui/main_window.py" in panel.label_text()
    panel.update_now()
    assert applied == []


def test_update_is_refused_while_the_queue_is_running(qapp):
    panel = _panel(lambda repo: UpdateStatus("main", "origin/main", behind=1, new_commits=("a1 x",)))
    panel.check_now()
    assert _wait(panel, lambda: panel.buttons_enabled() == (True, True))
    panel.busy_check = lambda: True
    panel.confirm = lambda st: pytest.fail("подтверждение не должно спрашиваться, пока идёт очередь")
    panel.update_now()
    assert "дождитесь конца очереди" in panel.label_text()


def test_unexpected_git_failure_is_shown_not_swallowed(qapp):
    def boom(repo):
        raise RuntimeError("что-то сломалось")

    panel = _panel(boom)
    panel.check_now()
    assert _wait(panel, lambda: "Не удалось проверить обновления: что-то сломалось" in panel.label_text())


@pytest.mark.skipif(GIT is None, reason="git недоступен")
def test_real_git_flow_check_confirm_update_and_dirty_copy_is_protected(qapp, repos):
    app, dev = repos
    panel = UpdatePanel(app)
    publish(dev, "new_feature.txt", "новая функция")

    (app / "a.txt").write_text("незакоммиченная правка\n")               # 1) грязная копия: обновление запрещено
    panel.check_now()
    assert _wait(panel, lambda: panel.buttons_enabled()[0] and "незакоммиченные" in panel.label_text())
    assert panel.buttons_enabled() == (True, False) and "a.txt" in panel.label_text()
    assert not (app / "new_feature.txt").exists()

    git(app, "checkout", "--", "a.txt")                                  # 2) правку откатили: обновление доступно
    panel.check_now()
    assert _wait(panel, lambda: panel.buttons_enabled() == (True, True))
    assert "новая функция" in panel.label_text()
    panel.confirm = lambda st: True
    panel.update_now()
    assert _wait(panel, lambda: "Обновлено до" in panel.label_text())
    assert (app / "new_feature.txt").exists()
