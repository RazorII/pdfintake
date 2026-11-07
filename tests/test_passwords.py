"""passwords.py: saved passwords tried without asking (DESIGN §12.1).

Every password here is an invented string, and every PDF is built in the test with pikepdf.
"""

from __future__ import annotations

import logging
import os
import stat
import sys
from pathlib import Path

import pikepdf
import pytest

from pdfintake import passwords
from pdfintake.errors import EXIT_USAGE, UsageError

USER_PW = "violet-anchor-7731"
OWNER_PW = "quilt-harbor-0420"
DECOYS = ["lantern-moss-1", "pebble-orbit-2", "copper-fern-3"]
ALL_SECRETS = [USER_PW, OWNER_PW, *DECOYS]

posix_only = pytest.mark.skipif(sys.platform == "win32", reason="POSIX owner and mode checks")


# --------------------------------------------------------------------------------- helpers

@pytest.fixture
def home(tmp_path, monkeypatch):
    """A fake home folder, so default_path() never touches the real config folder."""
    h = tmp_path / "home"
    h.mkdir()
    monkeypatch.setenv("HOME", str(h))
    monkeypatch.setenv("USERPROFILE", str(h))
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    monkeypatch.delenv("LOCALAPPDATA", raising=False)
    return h


def _write(path: Path, data: bytes | str, mode: int = 0o600) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data.encode("utf-8") if isinstance(data, str) else data)
    os.chmod(path, mode)
    return path


def _pdf(path: Path, **enc) -> Path:
    pdf = pikepdf.new()
    pdf.add_blank_page()
    if enc:
        pdf.save(path, encryption=pikepdf.Encryption(**enc))
    else:
        pdf.save(path)
    return path


@pytest.fixture
def user_pdf(tmp_path):
    return _pdf(tmp_path / "locked.pdf", user=USER_PW, owner=OWNER_PW, R=6)


@pytest.fixture
def r4_pdf(tmp_path):
    return _pdf(tmp_path / "locked-r4.pdf", user=USER_PW, owner=OWNER_PW, R=4)


@pytest.fixture
def perm_pdf(tmp_path):
    return _pdf(tmp_path / "edit-lock.pdf", user="", owner=OWNER_PW, R=6,
                allow=pikepdf.Permissions(extract=False, modify_other=False))


def _no_secret(*texts: str) -> None:
    for text in texts:
        for secret in ALL_SECRETS:
            assert secret not in text


def _no_secret_in_frames(err: BaseException) -> None:
    """No exception in ``err``'s chain, and no local in any traceback frame, holds a secret."""
    seen = set()
    exc = err
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        _no_secret(str(exc), repr(exc))
        tb = exc.__traceback__
        while tb is not None:
            if Path(tb.tb_frame.f_code.co_filename).name != Path(__file__).name:
                _no_secret(repr(tb.tb_frame.f_locals))
            tb = tb.tb_next
        exc = exc.__cause__ or exc.__context__


def _assert_unsafe(excinfo, reason: str) -> None:
    err = excinfo.value
    assert isinstance(err, UsageError)
    assert err.exit_code == EXIT_USAGE == 2
    assert err.code == "passwords_file_unsafe"
    assert err.detail["reason"] == reason
    _no_secret(str(err), repr(err), err.message, repr(err.detail))


# --------------------------------------------------------------------------------- default path

def test_default_path_per_platform(home, monkeypatch):
    monkeypatch.setattr(passwords.sys, "platform", "darwin")
    assert passwords.default_path() == home / "Library" / "Application Support" / "pdfintake" / "passwords"

    monkeypatch.setattr(passwords.sys, "platform", "linux")
    assert passwords.default_path() == home / ".config" / "pdfintake" / "passwords"
    monkeypatch.setenv("XDG_CONFIG_HOME", "relative/ignored")
    assert passwords.default_path() == home / ".config" / "pdfintake" / "passwords"
    xdg = home / "xdg"
    monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg))
    assert passwords.default_path() == xdg / "pdfintake" / "passwords"

    monkeypatch.setattr(passwords.sys, "platform", "win32")
    assert passwords.default_path() == home / "AppData" / "Local" / "pdfintake" / "passwords"
    local = home / "Local"
    monkeypatch.setenv("LOCALAPPDATA", str(local))
    assert passwords.default_path() == local / "pdfintake" / "passwords"


def test_default_path_is_never_created(home):
    passwords.default_path()
    passwords.candidates(None)
    assert list(home.iterdir()) == []


# --------------------------------------------------------------------------------- candidates

def test_no_default_file_means_no_candidates(home):
    got = passwords.candidates(None)
    assert got == [] and repr(got) == "<0 saved passwords>"


def test_default_file_is_read_when_present(home):
    _write(passwords.default_path(), f"{USER_PW}\n")
    assert passwords.candidates(None) == [USER_PW]


@posix_only
def test_default_file_must_be_safe_too(home):
    _write(passwords.default_path(), f"{USER_PW}\n", mode=0o644)
    with pytest.raises(UsageError) as excinfo:
        passwords.candidates(None)
    _assert_unsafe(excinfo, "mode")


def test_order_kept_and_lines_parsed_exactly(tmp_path):
    body = "﻿" + f"{DECOYS[0]}\r\n\n {DECOYS[1]} \n{DECOYS[0]}\n{USER_PW}"
    f = _write(tmp_path / "pw", body)
    got = passwords.candidates(f)
    # BOM dropped, CRLF and empty lines handled, spaces kept, repeats tried once, last line without a break kept.
    assert list(got) == [DECOYS[0], f" {DECOYS[1]} ", USER_PW]


def test_slices_copies_and_sums_stay_hidden(tmp_path):
    got = passwords.candidates(_write(tmp_path / "pw", "\n".join(ALL_SECRETS)))
    for derived in (got[:2], got[1:], got[::-1], got.copy(), got + [], [] + got, got * 1, 1 * got):
        assert isinstance(derived, passwords.Candidates)
        _no_secret(repr(derived), str(derived), repr({"c": derived}))
    assert got[0] == ALL_SECRETS[0] and type(got[0]) is str
    assert list(got[:2]) == ALL_SECRETS[:2]


def test_candidates_repr_and_str_hide_the_passwords(tmp_path):
    got = passwords.candidates(_write(tmp_path / "pw", "\n".join(ALL_SECRETS)))
    assert repr(got) == str(got) == f"<{len(ALL_SECRETS)} saved passwords>"
    assert f"{got}" == repr(got) and f"{got!r}" == repr(got)
    assert repr(passwords.Candidates(["x"])) == "<1 saved password>"
    _no_secret(repr(got), str(got), repr({"c": got}))
    assert isinstance(got, list)


def test_explicit_file_must_exist(tmp_path):
    with pytest.raises(UsageError) as excinfo:
        passwords.candidates(tmp_path / "nope")
    _assert_unsafe(excinfo, "missing")


@posix_only
@pytest.mark.parametrize("mode", [0o644, 0o640, 0o604, 0o660, 0o666, 0o602, 0o620])
def test_group_or_other_access_is_refused(tmp_path, mode):
    f = _write(tmp_path / "pw", f"{USER_PW}\n", mode=mode)
    with pytest.raises(UsageError) as excinfo:
        passwords.candidates(f)
    _assert_unsafe(excinfo, "mode")
    assert "chmod 600" in str(excinfo.value)


@posix_only
@pytest.mark.parametrize("mode", [0o600, 0o400])
def test_owner_only_modes_pass(tmp_path, mode):
    f = _write(tmp_path / "pw", f"{USER_PW}\n", mode=mode)
    assert passwords.candidates(f) == [USER_PW]


@posix_only
def test_another_owner_is_refused(tmp_path, monkeypatch):
    f = _write(tmp_path / "pw", f"{USER_PW}\n")
    monkeypatch.setattr(passwords.os, "geteuid", lambda: os.stat(f).st_uid + 1)
    with pytest.raises(UsageError) as excinfo:
        passwords.candidates(f)
    _assert_unsafe(excinfo, "owner")


def test_a_folder_is_refused(tmp_path):
    d = tmp_path / "pw"
    d.mkdir()
    with pytest.raises(UsageError) as excinfo:
        passwords.candidates(d)
    assert excinfo.value.detail["reason"] in {"not_a_file", "unreadable"}   # Windows can't open a folder


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="needs a FIFO")
def test_a_fifo_is_refused_without_blocking(tmp_path):
    fifo = tmp_path / "pw"
    os.mkfifo(fifo, 0o600)
    with pytest.raises(UsageError) as excinfo:
        passwords.candidates(fifo)
    _assert_unsafe(excinfo, "not_a_file")


def test_at_most_100_entries(tmp_path):
    hundred = [f"invented-{i:03d}" for i in range(100)]
    assert passwords.candidates(_write(tmp_path / "ok", "\n".join(hundred) + "\n")) == hundred
    f = _write(tmp_path / "many", "\n".join(hundred + ["invented-100"]) + "\n")
    with pytest.raises(UsageError) as excinfo:
        passwords.candidates(f)
    _assert_unsafe(excinfo, "too_many")
    assert "invented-0" not in str(excinfo.value)


def test_too_large_is_refused_before_reading(tmp_path):
    f = _write(tmp_path / "pw", b"a" * (passwords.MAX_BYTES + 1))
    with pytest.raises(UsageError) as excinfo:
        passwords.candidates(f)
    _assert_unsafe(excinfo, "too_large")


def test_not_utf8_is_refused_without_quoting_bytes(tmp_path):
    f = _write(tmp_path / "pw", USER_PW.encode() + b"\xff\xfe\n")
    with pytest.raises(UsageError) as excinfo:
        passwords.candidates(f)
    _assert_unsafe(excinfo, "not_utf8")
    # No UnicodeDecodeError chained to it at all: its .object would be the file's raw bytes (decided 2026-11-06).
    assert excinfo.value.__cause__ is None and excinfo.value.__context__ is None


@pytest.mark.parametrize("case", ["not_utf8", "too_many"])
def test_a_refusal_keeps_no_line_of_the_file(tmp_path, case):
    """A run keeps this error for its whole batch, so neither it, anything chained to it, nor any frame
    of its traceback may hold a line of the file (DESIGN §12.1, decided 2026-11-06)."""
    body = "\n".join(ALL_SECRETS) + "\n"
    if case == "too_many":
        body += "".join(f"invented-{i:03d}\n" for i in range(100))
    data = body.encode() + (b"\xff\xfe\n" if case == "not_utf8" else b"")
    f = _write(tmp_path / "pw", data)
    with pytest.raises(UsageError) as excinfo:
        passwords.candidates(f)
    _assert_unsafe(excinfo, case)
    err = excinfo.value
    assert err.__context__ is None and err.__cause__ is None
    _no_secret_in_frames(err)


@posix_only
def test_message_names_the_file_not_the_contents(tmp_path):
    f = _write(tmp_path / "pw", "\n".join(ALL_SECRETS), mode=0o644)
    with pytest.raises(UsageError) as excinfo:
        passwords.candidates(f)
    assert str(f) in str(excinfo.value)
    _no_secret(str(excinfo.value), repr(excinfo.value))


# --------------------------------------------------------------------------------- check

def _verdict_of_candidates(path) -> str | None:
    try:
        passwords.candidates(path)
    except UsageError as exc:
        return exc.detail["reason"]
    return None


def _check_cases(tmp_path: Path) -> dict[str, tuple[Path, str | None]]:
    hundred = "".join(f"invented-{i:03d}\n" for i in range(100))
    folder = tmp_path / "a-folder"
    folder.mkdir()
    cases = {
        "ok": (_write(tmp_path / "ok", "\n".join(ALL_SECRETS) + "\n"), None),
        "hundred": (_write(tmp_path / "hundred", hundred), None),
        "hundred_with_repeats": (_write(tmp_path / "repeats", hundred + "invented-000\n" * 50), None),
        "too_many": (_write(tmp_path / "many", hundred + "invented-100\n"), "too_many"),
        "too_large": (_write(tmp_path / "large", b"a" * (passwords.MAX_BYTES + 1)), "too_large"),
        "not_utf8": (_write(tmp_path / "bytes", USER_PW.encode() + b"\xff\xfe\n"), "not_utf8"),
        "missing": (tmp_path / "nope", "missing"),
        "folder": (folder, "not_a_file" if sys.platform != "win32" else "unreadable"),
    }
    if sys.platform != "win32":
        cases["mode"] = (_write(tmp_path / "shared", f"{USER_PW}\n", mode=0o644), "mode")
        cases["read_only"] = (_write(tmp_path / "ro", f"{USER_PW}\n", mode=0o400), None)
        if os.geteuid() != 0:
            cases["unreadable"] = (_write(tmp_path / "locked", f"{USER_PW}\n", mode=0o000), "unreadable")
    if hasattr(os, "mkfifo"):
        os.mkfifo(tmp_path / "fifo", 0o600)
        cases["fifo"] = (tmp_path / "fifo", "not_a_file")
    return cases


def test_check_gives_the_verdict_candidates_gives(tmp_path):
    for name, (path, reason) in _check_cases(tmp_path).items():
        got = passwords.check(path)
        assert got == reason, name
        assert got == _verdict_of_candidates(path), name          # one code path, one answer


def test_check_returns_only_a_reason_never_a_password(tmp_path, caplog, capsys):
    import gc
    import secrets

    token = "invented-" + secrets.token_hex(8)           # made up here, so only check could hold them
    good = _write(tmp_path / "ok", "".join(f"{token}-{i:03d}\n" for i in range(5)))
    bad = _write(tmp_path / "many", "".join(f"{token}-{i:03d}\n" for i in range(101)))
    assert passwords.check(good) is None
    reason = passwords.check(bad)
    assert reason == "too_many" and reason in passwords._REASONS
    assert token not in repr(reason)
    out = capsys.readouterr()
    assert token not in out.out + out.err + caplog.text
    gc.collect()
    held = [o for o in gc.get_objects()
            if isinstance(o, (list, set, tuple)) and any(isinstance(x, str) and x.startswith(token) for x in o)]
    assert held == [], "check kept the passwords it read"


def test_check_never_calls_candidates(tmp_path, monkeypatch):
    f = _write(tmp_path / "ok", f"{USER_PW}\n")
    monkeypatch.setattr(passwords, "candidates", lambda *a, **k: pytest.fail("check went through candidates"))
    assert passwords.check(f) is None


def test_check_of_the_default_file(home):
    assert passwords.check(None) is None                 # no file: nothing a run would refuse
    assert list(home.iterdir()) == []                    # and nothing was created
    _write(passwords.default_path(), f"{USER_PW}\n")
    assert passwords.check(None) is None
    if sys.platform != "win32":
        os.chmod(passwords.default_path(), 0o644)
        assert passwords.check(None) == "mode"


@posix_only
def test_check_sees_the_owner_as_candidates_does(tmp_path, monkeypatch):
    f = _write(tmp_path / "pw", f"{USER_PW}\n")
    monkeypatch.setattr(passwords.os, "geteuid", lambda: os.stat(f).st_uid + 1)
    assert passwords.check(f) == "owner" == _verdict_of_candidates(f)


def test_check_on_the_windows_branch(home, tmp_path, monkeypatch):
    monkeypatch.setattr(passwords, "_is_windows", lambda: True)
    inside = _write(home / "AppData" / "Local" / "pdfintake" / "passwords", f"{USER_PW}\n", mode=0o644)
    outside = _write(tmp_path / "shared" / "passwords", f"{USER_PW}\n")
    assert passwords.check(inside) is None
    assert passwords.check(outside) == "outside_profile" == _verdict_of_candidates(outside)


# --------------------------------------------------------------------------------- the Windows branch

def test_windows_branch_accepts_a_file_inside_the_profile(home, monkeypatch):
    monkeypatch.setattr(passwords, "_is_windows", lambda: True)
    f = _write(home / "AppData" / "Local" / "pdfintake" / "passwords", f"{USER_PW}\n", mode=0o644)
    assert passwords.candidates(f) == [USER_PW]          # no POSIX mode to read on Windows


def test_windows_branch_refuses_a_file_outside_the_profile(home, tmp_path, monkeypatch):
    monkeypatch.setattr(passwords, "_is_windows", lambda: True)
    f = _write(tmp_path / "shared" / "passwords", f"{USER_PW}\n")
    with pytest.raises(UsageError) as excinfo:
        passwords.candidates(f)
    _assert_unsafe(excinfo, "outside_profile")


def test_windows_branch_refuses_a_link_out_of_the_profile(home, tmp_path, monkeypatch):
    monkeypatch.setattr(passwords, "_is_windows", lambda: True)
    real = _write(tmp_path / "shared" / "passwords", f"{USER_PW}\n")
    link = home / "passwords"
    try:
        link.symlink_to(real)
    except OSError:
        pytest.skip("no symlinks here")
    with pytest.raises(UsageError) as excinfo:
        passwords.candidates(link)
    _assert_unsafe(excinfo, "outside_profile")


# --------------------------------------------------------------------------------- open_with

def test_opens_with_the_right_password_wherever_it_is(user_pdf):
    for k in range(len(DECOYS) + 1):
        cands = DECOYS[:k] + [USER_PW] + DECOYS[k:]
        pdf = passwords.open_with(user_pdf, cands)
        assert pdf is not None
        assert len(pdf.pages) == 1
        pdf.close()


def test_owner_password_opens_too(user_pdf):
    pdf = passwords.open_with(user_pdf, [DECOYS[0], OWNER_PW])
    assert pdf is not None
    pdf.close()


def test_r4_file(r4_pdf):
    pdf = passwords.open_with(r4_pdf, [*DECOYS, USER_PW])
    assert pdf is not None
    pdf.close()


def test_no_saved_password_opens_it(user_pdf):
    assert passwords.open_with(user_pdf, DECOYS) is None
    assert passwords.open_with(user_pdf, []) is None
    assert passwords.open_with(user_pdf, passwords.Candidates()) is None


def test_edit_lock_opens_even_when_every_saved_password_is_wrong(perm_pdf):
    # pikepdf refuses a wrong password on an edit-locked file, so the empty one is tried first.
    pdf = passwords.open_with(perm_pdf, DECOYS)
    assert pdf is not None
    pdf.close()


def test_unencrypted_file_opens(tmp_path):
    pdf = passwords.open_with(_pdf(tmp_path / "plain.pdf"), DECOYS)
    assert pdf is not None and not pdf.is_encrypted
    pdf.close()


def test_works_with_any_iterable_and_stops_at_the_first_that_opens(user_pdf):
    seen = []

    def gen():
        for p in [DECOYS[0], USER_PW, DECOYS[1]]:
            seen.append(p)
            yield p

    pdf = passwords.open_with(user_pdf, gen())
    assert pdf is not None
    pdf.close()
    assert seen == [DECOYS[0], USER_PW]


def test_empty_candidates_are_skipped(user_pdf):
    pdf = passwords.open_with(user_pdf, ["", USER_PW])
    assert pdf is not None
    pdf.close()


def test_a_non_string_candidate_is_a_type_error_without_the_value(user_pdf):
    with pytest.raises(TypeError) as excinfo:
        passwords.open_with(user_pdf, [DECOYS[0], USER_PW.encode()])
    _no_secret(str(excinfo.value), repr(excinfo.value))
    assert "violet" not in str(excinfo.value)


def test_a_missing_file_raises_without_a_password(tmp_path):
    with pytest.raises(Exception) as excinfo:
        passwords.open_with(tmp_path / "gone.pdf", DECOYS)
    _no_secret(str(excinfo.value), repr(excinfo.value))


def test_a_damaged_file_raises_without_a_password(tmp_path, user_pdf):
    broken = tmp_path / "broken.pdf"
    broken.write_bytes(b"not a pdf at all " * 10)
    with pytest.raises(pikepdf.PdfError) as excinfo:
        passwords.open_with(broken, DECOYS)
    _no_secret(str(excinfo.value), repr(excinfo.value))


def test_an_error_quoting_a_password_is_replaced(user_pdf, monkeypatch):
    real_open = pikepdf.open

    def leaky(path, password=""):
        if password == DECOYS[1]:
            raise pikepdf.PdfError(f"something went wrong with {password}")
        return real_open(path, password=password)

    monkeypatch.setattr(pikepdf, "open", leaky)
    with pytest.raises(pikepdf.PdfError) as excinfo:
        passwords.open_with(user_pdf, [DECOYS[0], DECOYS[1], USER_PW])
    _no_secret(str(excinfo.value), repr(excinfo.value))
    err = excinfo.value
    assert err.__cause__ is None and err.__context__ is None
    assert "locked.pdf" in str(err)
    _no_secret_in_frames(err)


def test_an_error_whose_chain_quotes_a_password_is_replaced(user_pdf, monkeypatch):
    real_open = pikepdf.open

    def leaky(path, password=""):
        if password == DECOYS[1]:
            try:
                raise ValueError(f"inner detail {password}")
            except ValueError:
                raise pikepdf.PdfError("outer message without it")
        return real_open(path, password=password)

    monkeypatch.setattr(pikepdf, "open", leaky)
    with pytest.raises(pikepdf.PdfError) as excinfo:
        passwords.open_with(user_pdf, [DECOYS[0], DECOYS[1], USER_PW])
    err = excinfo.value
    assert err.__context__ is None and "locked.pdf" in str(err)
    _no_secret_in_frames(err)


def test_a_reraised_error_keeps_no_password_in_its_frames(user_pdf, monkeypatch):
    real_open = pikepdf.open

    def failing(path, password=""):
        if password == DECOYS[1]:
            raise pikepdf.PdfError("a damaged object stream")
        return real_open(path, password=password)

    monkeypatch.setattr(pikepdf, "open", failing)
    with pytest.raises(pikepdf.PdfError) as excinfo:
        passwords.open_with(user_pdf, [DECOYS[0], DECOYS[1], USER_PW])
    assert str(excinfo.value) == "a damaged object stream"      # not a quoting error: passed on as is
    _no_secret_in_frames(excinfo.value)


def test_open_with_logs_and_prints_nothing(user_pdf, caplog, capsys):
    caplog.set_level(logging.DEBUG)
    pdf = passwords.open_with(user_pdf, [*DECOYS, USER_PW])
    pdf.close()
    passwords.open_with(user_pdf, DECOYS)
    out = capsys.readouterr()
    assert caplog.records == [] and out.out == "" and out.err == ""


def test_open_with_returns_only_the_pdf(user_pdf):
    # Which password worked isn't returned or attached to the result.
    pdf = passwords.open_with(user_pdf, [DECOYS[0], USER_PW])
    assert type(pdf) is pikepdf.Pdf
    _no_secret(repr(pdf))
    pdf.close()


# --------------------------------------------------------------------------------- add

def _from(tmp_path: Path, text: str, name: str = "incoming") -> Path:
    return _write(tmp_path / "skill-temp" / name, text, mode=0o644)   # the skill's file tool may not set 0600


@posix_only
def test_add_creates_the_default_file_private(home, tmp_path):
    passwords.add(_from(tmp_path, USER_PW + "\n"))
    target = passwords.default_path()
    assert target.read_text("utf-8") == USER_PW + "\n"
    assert stat.S_IMODE(os.stat(target).st_mode) == 0o600
    assert stat.S_IMODE(os.stat(target.parent).st_mode) == 0o700
    assert passwords.candidates(None) == [USER_PW]


@posix_only
def test_add_is_0600_whatever_the_umask(home, tmp_path):
    src = _from(tmp_path, USER_PW)
    passwords.default_path().parent.mkdir(parents=True)    # a umask this strict would block new folders
    old = os.umask(0o277)
    try:
        passwords.add(src)
    finally:
        os.umask(old)
    assert stat.S_IMODE(os.stat(passwords.default_path()).st_mode) == 0o600


def test_add_appends_in_order_and_skips_duplicates(home, tmp_path):
    passwords.add(_from(tmp_path, DECOYS[0]))
    passwords.add(_from(tmp_path, f"{DECOYS[1]}\n{DECOYS[0]}\n{DECOYS[1]}\n"))
    passwords.add(_from(tmp_path, DECOYS[0] + "\n"))
    assert passwords.default_path().read_text("utf-8") == f"{DECOYS[0]}\n{DECOYS[1]}\n"


def test_add_mends_a_missing_final_line_break(home, tmp_path):
    _write(passwords.default_path(), DECOYS[0])
    passwords.add(_from(tmp_path, USER_PW))
    assert passwords.candidates(None) == [DECOYS[0], USER_PW]


def test_add_to_an_explicit_file(tmp_path):
    target = tmp_path / "wrapper" / "pw"
    passwords.add(_from(tmp_path, USER_PW), to=target)
    assert passwords.candidates(target) == [USER_PW]


def test_add_with_an_empty_file_writes_nothing(home, tmp_path):
    passwords.add(_from(tmp_path, "\n\n"))
    assert not passwords.default_path().exists()


@posix_only
def test_add_refuses_an_unsafe_list_and_writes_nothing(home, tmp_path):
    target = _write(passwords.default_path(), DECOYS[0] + "\n", mode=0o644)
    with pytest.raises(UsageError) as excinfo:
        passwords.add(_from(tmp_path, USER_PW))
    _assert_unsafe(excinfo, "mode")
    assert target.read_text("utf-8") == DECOYS[0] + "\n"


def test_add_refuses_past_100(home, tmp_path):
    hundred = [f"invented-{i:03d}" for i in range(100)]
    _write(passwords.default_path(), "\n".join(hundred) + "\n")
    passwords.add(_from(tmp_path, hundred[5]))           # a duplicate is still fine
    with pytest.raises(UsageError) as excinfo:
        passwords.add(_from(tmp_path, USER_PW))
    _assert_unsafe(excinfo, "full")
    assert passwords.candidates(None) == hundred


def test_repeats_count_once_in_candidates_and_add(home, tmp_path):
    # 100 lines, 60 different passwords: add() and candidates() must agree on the count.
    sixty = [f"invented-{i:03d}" for i in range(60)]
    _write(passwords.default_path(), "\n".join(sixty + sixty[:40]) + "\n")
    assert passwords.candidates(None) == sixty
    forty = [f"fresh-{i:02d}" for i in range(40)]
    passwords.add(_from(tmp_path, "\n".join(forty)))
    assert passwords.candidates(None) == sixty + forty      # the file add() wrote is still read
    with pytest.raises(UsageError) as excinfo:
        passwords.add(_from(tmp_path, USER_PW))
    _assert_unsafe(excinfo, "full")
    assert passwords.candidates(None) == sixty + forty


def test_many_repeats_of_few_passwords_are_read(tmp_path):
    f = _write(tmp_path / "pw", "\n".join(DECOYS * 50) + "\n")     # 150 lines, 3 passwords
    assert passwords.candidates(f) == DECOYS


def test_a_failed_add_leaves_no_new_file_behind(home, tmp_path, monkeypatch):
    # Windows branch with %LOCALAPPDATA% outside the profile: refused before anything is created.
    monkeypatch.setattr(passwords, "_is_windows", lambda: True)
    monkeypatch.setattr(passwords.sys, "platform", "win32")
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "elsewhere"))
    with pytest.raises(UsageError) as excinfo:
        passwords.add(_from(tmp_path, USER_PW))
    _assert_unsafe(excinfo, "outside_profile")
    assert not passwords.default_path().exists()
    assert not (tmp_path / "elsewhere").exists()
    assert passwords.candidates(None) == []                  # later runs aren't blocked


# The two tests below run the platform's own checks and, on POSIX too, the Windows branch. Their list
# sits inside the (fake) profile folder: on Windows a new list outside it is refused before it is
# created, so the failure after creation they test would never be reached.
@pytest.fixture(params=["native", "windows_branch"])
def branch(request, monkeypatch):
    if request.param == "windows_branch":
        monkeypatch.setattr(passwords, "_is_windows", lambda: True)
    return request.param


def test_a_check_failing_after_create_removes_the_new_file(home, tmp_path, monkeypatch, branch):
    real_check = passwords._check_open_file
    refused = []

    def refuse_the_list(path, st, *, private):
        if not private:                                       # the source file, read first: passes as usual
            return real_check(path, st, private=private)
        refused.append(os.path.lexists(path))
        raise passwords._unsafe(path, "owner")

    monkeypatch.setattr(passwords, "_check_open_file", refuse_the_list)
    target = home / "wrapper" / "pw"
    with pytest.raises(UsageError) as excinfo:
        passwords.add(_from(tmp_path, USER_PW), to=target)
    assert excinfo.value.detail["reason"] == "owner" and excinfo.value.__context__ is None
    assert refused == [True]                                  # refused after it was created ...
    assert not target.exists()                                # ... and removed again


def test_a_write_failing_after_create_removes_the_new_file(home, tmp_path, monkeypatch, branch):
    real_write = os.write

    def broken(fd, data):
        raise OSError("disk full")

    target = home / "wrapper" / "pw"
    src = _from(tmp_path, USER_PW)
    monkeypatch.setattr(passwords.os, "write", broken)
    with pytest.raises(UsageError) as excinfo:
        passwords.add(src, to=target)
    monkeypatch.setattr(passwords.os, "write", real_write)
    _assert_unsafe(excinfo, "unreadable")
    assert not target.exists()


def test_a_failed_add_never_removes_an_existing_list(home, tmp_path, monkeypatch):
    target = _write(passwords.default_path(), DECOYS[0] + "\n")
    src = _from(tmp_path, USER_PW)
    real_write = os.write

    def broken(fd, data):
        raise OSError("disk full")

    monkeypatch.setattr(passwords.os, "write", broken)
    with pytest.raises(UsageError):
        passwords.add(src)
    monkeypatch.setattr(passwords.os, "write", real_write)
    assert target.read_text("utf-8") == DECOYS[0] + "\n"


def test_add_needs_the_source_file(home, tmp_path):
    with pytest.raises(UsageError) as excinfo:
        passwords.add(tmp_path / "nope")
    _assert_unsafe(excinfo, "missing")


def test_add_logs_and_prints_nothing(home, tmp_path, caplog, capsys):
    caplog.set_level(logging.DEBUG)
    assert passwords.add(_from(tmp_path, USER_PW)) is None
    out = capsys.readouterr()
    assert caplog.records == [] and out.out == "" and out.err == ""


def test_add_then_open(home, tmp_path, user_pdf):
    passwords.add(_from(tmp_path, "\n".join(DECOYS)))
    assert passwords.open_with(user_pdf, passwords.candidates(None)) is None
    passwords.add(_from(tmp_path, USER_PW))
    pdf = passwords.open_with(user_pdf, passwords.candidates(None))
    assert pdf is not None
    pdf.close()


# --------------------------------------------------------------------------------- hygiene

def test_the_module_never_prompts_or_logs():
    src = Path(passwords.__file__).read_text("utf-8")
    for banned in ("input(", "getpass", "sys.stdin", "import logging", "print(", "keyring", "security "):
        assert banned not in src
