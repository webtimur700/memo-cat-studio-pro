import subprocess
from pathlib import Path

import pytest

from updater.git_updater import apply_update, check_for_updates, git_command


GIT = git_command()
pytestmark = pytest.mark.skipif(GIT is None, reason="git недоступен")


def git(cwd: Path, *args: str) -> str:
    return subprocess.run([*GIT, "-c", "user.email=t@t", "-c", "user.name=t", *args], cwd=cwd, check=True,
                          capture_output=True, text=True).stdout.strip()


@pytest.fixture()
def repos(tmp_path):
    """Сервер (bare) и две копии: app — «приложение пользователя», dev — кто-то другой, кто публикует коммиты."""
    server = tmp_path / "server.git"
    subprocess.run([*GIT, "init", "--bare", "-b", "main", str(server)], check=True, capture_output=True)
    app, dev = tmp_path / "app", tmp_path / "dev"
    for path in (app, dev):
        subprocess.run([*GIT, "clone", str(server), str(path)], check=True, capture_output=True)
    (dev / "a.txt").write_text("1\n")
    (dev / "pyproject.toml").write_text("[project]\nname='x'\n")
    git(dev, "add", "."); git(dev, "commit", "-m", "первый коммит"); git(dev, "push", "-u", "origin", "HEAD:main")
    git(app, "pull", "origin", "main"); git(app, "branch", "--set-upstream-to=origin/main", "main")
    return app, dev


def publish(dev: Path, name: str = "b.txt", message: str = "новая функция"):
    (dev / name).write_text("x\n")
    git(dev, "add", "."); git(dev, "commit", "-m", message); git(dev, "push", "origin", "HEAD:main")


def test_up_to_date_and_new_commits_are_reported(repos):
    app, dev = repos
    status = check_for_updates(app)
    assert status.behind == 0 and not status.can_update and "последняя версия" in status.message

    publish(dev, message="исправлена очередь")
    publish(dev, "c.txt", "добавлена музыка")
    status = check_for_updates(app)                                    # fetch происходит внутри
    assert (status.behind, status.ahead) == (2, 0) and status.can_update
    assert [c.split(" ", 1)[1] for c in status.new_commits] == ["добавлена музыка", "исправлена очередь"]
    assert "Доступно обновление: 2" in status.message


def test_update_fast_forwards_only_after_a_call_and_reports_dependency_change(repos):
    app, dev = repos
    before = git(app, "rev-parse", "HEAD")
    publish(dev)
    assert check_for_updates(app).can_update and git(app, "rev-parse", "HEAD") == before      # проверка ничего не меняет
    (dev / "pyproject.toml").write_text("[project]\nname='x'\ndependencies=['y']\n")
    git(dev, "commit", "-am", "новая зависимость"); git(dev, "push", "origin", "HEAD:main")

    result = apply_update(app)
    assert result.ok and result.dependencies_changed and "pip install -e ." in result.message
    assert git(app, "rev-parse", "HEAD") == git(dev, "rev-parse", "HEAD") and (app / "b.txt").exists()
    assert check_for_updates(app).behind == 0


def test_uncommitted_changes_block_the_update_and_nothing_is_touched(repos):
    app, dev = repos
    publish(dev)
    (app / "a.txt").write_text("моя правка\n")
    status = check_for_updates(app)
    assert status.dirty_files == ("a.txt",) and not status.can_update and status.behind == 1
    assert "незакоммиченные изменения" in status.message and "a.txt" in status.message

    result = apply_update(app)
    assert not result.ok and "незакоммиченные" in result.message
    assert (app / "a.txt").read_text() == "моя правка\n" and not (app / "b.txt").exists()      # ни правка, ни коммиты не тронуты

    staged = repos[0]
    git(staged, "add", "a.txt")                                           # проиндексированное тоже считается
    assert check_for_updates(app).dirty_files == ("a.txt",)


def test_untracked_files_do_not_block_but_are_counted(repos):
    app, dev = repos
    publish(dev)
    (app / "notes.txt").write_text("личное\n")
    status = check_for_updates(app)
    assert status.untracked == 1 and status.can_update
    assert apply_update(app).ok and (app / "notes.txt").exists()


def test_untracked_file_that_would_be_overwritten_makes_git_refuse_safely(repos):
    app, dev = repos
    publish(dev, "same.txt")
    (app / "same.txt").write_text("мой файл с тем же именем\n")
    result = apply_update(app)
    assert not result.ok and (app / "same.txt").read_text() == "мой файл с тем же именем\n"


def test_diverged_branch_is_not_auto_updated(repos):
    app, dev = repos
    publish(dev)
    (app / "local.txt").write_text("l\n")
    git(app, "add", "."); git(app, "commit", "-m", "локальный коммит")
    status = check_for_updates(app)
    assert (status.ahead, status.behind) == (1, 1) and not status.can_update and "разошлась" in status.message
    assert not apply_update(app).ok


def test_errors_are_explained(tmp_path, repos):
    plain = tmp_path / "not_a_repo"
    plain.mkdir()
    assert "не является git-репозиторием" in check_for_updates(plain).message

    app, _ = repos
    git(app, "checkout", "-b", "feature")
    assert "нет удалённой ветки" in check_for_updates(app).message

    git(app, "checkout", "main")
    git(app, "remote", "set-url", "origin", str(tmp_path / "does_not_exist.git"))
    status = check_for_updates(app)
    assert status.error and "Не удалось проверить обновления" in status.message


def test_missing_git_is_explained(tmp_path, monkeypatch):
    import updater.git_updater as module

    monkeypatch.setattr(module.shutil, "which", lambda name: None)
    assert module.git_command() is None
    assert "git не найден" in check_for_updates(tmp_path).message
