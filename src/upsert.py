from src.process import NEW_SOURCE_FILE, QUARANTINE_FILE, ROOT, load_contract, upsert


def plural(count, word):
    return f"{count} {word}" + ("" if count == 1 else "s")


if __name__ == "__main__":
    summary = upsert(load_contract())

    since = "first load" if summary["watermark"] is None else f"signed up after {summary['watermark']:%Y-%m-%d %H:%M} UTC"
    reasons = ", ".join(f"{count} {reason}" for reason, count in summary["recorded"].items()) or "nothing new"

    print(f"bronze      {plural(summary['arrived'], 'row')} in {NEW_SOURCE_FILE.name} | {summary['landed']} new")
    print(f"silver      {plural(summary['added'], 'new student')} ({since}) | {summary['already_in']} already in")
    print(f"quarantine  {reasons} -> {QUARANTINE_FILE.relative_to(ROOT).as_posix()}")
    print(f"gold        {plural(summary['total'], 'new student')} encoded for the model")
