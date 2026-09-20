# Credential lock identity and upgrade safety

Python, TypeScript and Rust now derive the same lock filename from a resolved
credential path, including missing files reached through directory aliases or
dangling symlinks. Symlinks are followed before a subsequent `..` is applied.
Inaccessible, non-directory, looping and non-Unicode paths fail explicitly;
there is no guessed fallback lock. Resolution permits at most 40 link expansions.

Windows identities remove the ordinary verbatim prefix, normalize separators
and use whole-string Unicode lowercase. Case-sensitive Windows directories
may therefore be locked more conservatively than necessary. Missing non-ASCII
components and non-ASCII UNC roots are refused rather than risking different
locks for filesystem-equivalent spellings. Provision such local files first or
use an ASCII missing suffix. Device names, alternate streams, drive-relative
paths and trailing-dot/space components are not accepted. Ordinary POSIX
path hashes do not change.

**Upgrade participants together.** Stop processes using old credential refresh
or store-writing code before restarting them on the new versions, especially
on Windows and for previously unresolved POSIX aliases. Mixed versions can
choose different lock filenames. Use the same `LM15_LOCK_DIR`. Never delete a
live lock file to resolve contention.

This is path identity, not universal file identity: hardlinks, bind mounts,
mapped-drive/UNC aliases, changing symlinks and foreign CLI writers remain
outside the guarantee. The filesystem namespace must remain stable. Runtime
Unicode-table differences and actual cross-platform exclusion still need
verification. Explicit absolute paths avoid differences in home discovery;
only current-user `~` expansion is supported by the shared resolver.

Regression sources were added in the September 20 implementation pass.
No tests or platform builds were run in that pass.
