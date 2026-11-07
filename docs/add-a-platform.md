# Add a platform

A platform is one row in `PLATFORMS` in `src/pdfintake/platforms.py`. Every pack has a `default` reader set, so a new platform works with every pack as it is: no pack needs an edit.

A row looks like this:

```python
Platform("linux-aarch64", "linux", ("aarch64", "arm64"), True, ("ubuntu-24.04-arm",), "manylinux", "CI row optional"),
```

| Field | What it is |
|---|---|
| `key` | the platform's name in packs and records: `<os>-<arch>`, lowercase |
| `sys_platform` | `sys.platform` on it (`darwin`, `win32`, `linux`) |
| `machines` | the architecture names Python reports for it, as in `sysconfig.get_platform()` |
| `supported` | whether pdfintake runs there |
| `ci_runners` | the GitHub Actions runners that test it |
| `min_os` | the oldest OS version it runs on |
| `notes` | for an unsupported row, why: the message the user sees |

The platform is read from the running interpreter, not the machine underneath: an x86_64 Python under Rosetta on an Apple silicon Mac is `darwin-x86_64`.

## Turning on an unsupported row

Two platforms are listed but unsupported, because a dependency has no build for them:

- `darwin-x86_64`: pikepdf 10.17 has no Intel-Mac wheel.
- `win-arm64`: tesserocr has no win_arm64 wheel.

When the missing build appears:

1. **Check every runtime dependency installs there** with `uv tool install`, from wheels, without a compiler. pdfintake never builds a dependency from source on a user's machine.
2. **Run the full suite there**, including `pdfintake doctor --json --canary`, which reads every installed pack's test image with every engine.
3. **Flip `supported` to `True`**, set `min_os`, clear `notes`, and add the runner to `ci_runners`.
4. **Add the CI job** in `.github/workflows/test.yml` with the same runner.
5. **Update the platform table** in [README.md](../README.md) and the requirements in [claude-plugin/README.md](../claude-plugin/README.md). A test checks that the README table and the CI matrix match `PLATFORMS`.

## Adding a new row

The same steps, plus:

- `tests/test_platforms.py` gets the platform's `sysconfig` tag and machine name in its mapping cases.
- `platforms.cache_root()` must give the right cache folder for the OS. A new OS (not just a new processor) needs its own case there and in the tests.
- An engine that runs only on some platforms lists its keys in its `platforms` set; an engine that should run on the new platform gets the key added, and its contract suite must pass there.

## Platform-specific reader sets

A pack may add a reader set for one platform (Apple Vision on `darwin-arm64`, for example), keyed by the platform key. That's pack data, chosen by the kit's bench on that platform ([add-a-language.md](add-a-language.md)), and it comes after the platform is supported, in a pack PR.

## Open the PR

Use the **code** pull-request template, and paste the suite's output from the new platform.
