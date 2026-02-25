from Agent.ui import help_text, make_thread_id, normalize_user


def test_normalize_user():
    assert normalize_user("Pawan Gunjan") == "Pawan_Gunjan"
    assert normalize_user("P@wan!!!") == "Pwan"
    assert normalize_user("") == "anonymous"
    assert normalize_user("   ") == "anonymous"


def test_make_thread_id():
    tid = make_thread_id("PawanGunjan")
    parts = tid.split(":")
    assert len(parts) == 3
    assert parts[0] == "PawanGunjan"
    assert len(parts[2]) == 8  # uuid hex slice


def test_help_text_contains_commands():
    txt = help_text()
    assert "/help" in txt
    assert "/sync" in txt
    assert "/exit" in txt
