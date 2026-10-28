# Pre-registration of the replication

- `CONFIRM2_SPEC.md` is the replication protocol (SBA 7(a) loans and federal civil cases), unchanged since it was fixed:
  - SHA-256 `ce72a7d3fa2e6ae9cee96ebe06af0d4ee53a37a224e2976415b0bd9eece1e1c9`;
  - committed on 2026-10-25 (commit `b7d4f3d` of the private development repository), before any data were downloaded;
  - time-stamped with OpenTimestamps on 2026-10-25 between 18:44:49 and 18:45:02 UTC. `CONFIRM2_SPEC.md.ots` is the proof; it is attested in Bitcoin blocks 973110 and 973112.
- Verify: `pip install opentimestamps-client`, then `ots verify CONFIRM2_SPEC.md.ots` (next to `CONFIRM2_SPEC.md`). `ots info CONFIRM2_SPEC.md.ots` shows the attestations.
- The protocol names two roles: "CD", who wrote it, and "CC", who ran it. Both were large language models (Claude) working under the authors' direction, as the paper's statement on the use of AI tools says. The file is kept byte for byte, so that the proof verifies.
- `confirm2_data_manifest.json`: URL, download time, size and SHA-256 of every raw file the replication used. `reproduce.py` compares a new download with it.
- Every change after the protocol was fixed is listed, with its effect, in the paper's table of deviations (Appendix E).
