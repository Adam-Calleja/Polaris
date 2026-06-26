#!/usr/bin/env python3
"""Generate a small, fully synthetic ragas one-hop evaluation set for Polaris.

The queries and gold answers are fabricated and correspond to the synthetic
ticket corpus (scripts/generate_synthetic_corpus.py). They follow the same
schema as the real benchmark so the evaluation harness runs unchanged, and the
label axes (source_needed, docs_scope_needed, validity_sensitive,
attachment_dependent) mirror the real benchmark's distribution.

No real query, answer, ticket, or person is represented.

Usage:
    python scripts/generate_synthetic_eval.py
"""
from __future__ import annotations

import json
from pathlib import Path

# (category, user_input, reference, source_needed, docs_scope_needed,
#  validity_sensitive, attachment_dependent)
EVAL: list[tuple[str, str, str, str, str, str, str]] = [
    ("Schedulers and Runtime",
     "Why does my Slurm job get killed with an oom-kill event and how do I fix it?",
     "The node ran out of memory. Request a high-memory (-himem) partition, reduce the per-task memory footprint, or set --mem explicitly (e.g. --mem=64G) and tune from there.",
     "both", "local_official", "yes", "no"),
    ("Schedulers and Runtime",
     "My job has been PENDING for hours with reason Priority. Is something wrong?",
     "A Priority reason usually means the partition is busy, not that the request is invalid. The job starts when nodes free up; request fewer nodes or a less busy partition for shorter waits.",
     "tickets", "none", "yes", "no"),
    ("Schedulers and Runtime",
     "sbatch rejects my script with 'Invalid partition name specified'. What should I use?",
     "The named partition does not exist. List valid partitions with 'sinfo -s' and use a current one.",
     "docs", "local_official", "yes", "no"),
    ("Schedulers and Runtime",
     "My array job only runs the first task and the rest produce no output. Why?",
     "Output paths likely used %j instead of %A_%a, so tasks overwrote each other. Use -o logs/%A_%a.out so each array task logs separately.",
     "both", "local_official", "no", "no"),
    ("Schedulers and Runtime",
     "How do I request a GPU so it is visible inside my job?",
     "Request the gpu resource and QOS (e.g. --gres=gpu:1) and load the cuda module; nvidia-smi then lists the allocated device.",
     "both", "local_official", "yes", "no"),
    ("Schedulers and Runtime",
     "My job was cancelled DUE TO TIME LIMIT. How do I give it more time?",
     "Increase --time up to the partition maximum (see 'scontrol show partition'), or checkpoint the run so it can resume across shorter jobs.",
     "both", "local_official", "yes", "no"),
    ("Storage, Filesystems, and Data Services",
     "I get 'Disk quota exceeded' writing to my project area. How do I get more space?",
     "The project is at its storage limit. Clear unneeded data or request a quota increase; large intermediate files can go in the scratch area, which is not quota-limited.",
     "both", "local_official", "yes", "no"),
    ("Storage, Filesystems, and Data Services",
     "Output files my job wrote are missing on the login node. Where did they go?",
     "They were likely written to node-local /tmp, which is cleared when the job ends. Write outputs to your project or scratch directory so they persist.",
     "tickets", "none", "yes", "no"),
    ("Storage, Filesystems, and Data Services",
     "Reading millions of small files is extremely slow. Is the filesystem broken?",
     "Many tiny files are an anti-pattern on parallel filesystems. Pack them into larger archives (tar/HDF5) or stage them to local scratch at job start for much better throughput.",
     "both", "local_official", "no", "no"),
    ("Storage, Filesystems, and Data Services",
     "What is the recommended way to transfer several hundred GB to the cluster?",
     "Use rsync over SSH for resumable transfers, or the dedicated data-transfer nodes for large moves. Avoid the login nodes for bulk copies as they are rate-limited.",
     "docs", "local_official", "yes", "no"),
    ("Storage, Filesystems, and Data Services",
     "A colleague added me to a project but I get Permission denied on the shared folder.",
     "Group membership may not have propagated to the filesystem ACL yet. Once refreshed, log out and back in and access works.",
     "tickets", "none", "yes", "no"),
    ("Access, Authentication, and Identity",
     "SSH suddenly fails with 'Permission denied (publickey)'. What should I check?",
     "Confirm your public key is present in your home directory and that your private key has 600 permissions. After account migrations the key sometimes needs re-adding.",
     "both", "local_official", "yes", "no"),
    ("Access, Authentication, and Identity",
     "I lost my MFA device and cannot log in. How do I reset it?",
     "After identity verification your MFA enrolment is reset; re-enrol a new device at first login via the self-service portal.",
     "tickets", "none", "yes", "no"),
    ("Access, Authentication, and Identity",
     "How does a new group member get an account on the HPC service?",
     "The PI adds the member to the project on the resource portal, which provisions an account automatically; onboarding instructions follow by email.",
     "docs", "local_official", "yes", "no"),
    ("Access, Authentication, and Identity",
     "I cannot reach the login nodes from off-site. Do I need a VPN?",
     "Off-campus access requires the institutional VPN or the documented jump host. Connect to the VPN first, then SSH as normal.",
     "docs", "local_official", "yes", "no"),
    ("Access, Authentication, and Identity",
     "Login is refused saying my account is inactive, but my grant is still running.",
     "The account likely reached its review date and can be extended in line with your active allocation, which restores access.",
     "tickets", "none", "yes", "no"),
    ("Platforms and Systems",
     "Which partition should I use for a memory-heavy workload?",
     "Use a high-memory (-himem) partition for memory-bound work; standard partitions suit normal runs. Per-node memory figures are in the platform documentation.",
     "docs", "local_official", "yes", "no"),
    ("Platforms and Systems",
     "Which is the most recent CPU partition and how do I benefit from it?",
     "Use the newest CPU partition in service and rebuild your code with an architecture-tuned flag to exploit its instruction set.",
     "docs", "local_official", "yes", "no"),
    ("Platforms and Systems",
     "How do I pick a specific compiler and MPI version rather than the defaults?",
     "Defaults are shown by 'module list'. Use 'module avail' to see versions and 'module load name/version' to pin a specific one.",
     "docs", "local_official", "yes", "no"),
    ("Platforms and Systems",
     "What GPU model do the accelerator nodes have, and how do I request them?",
     "The accelerator nodes carry data-centre GPUs; exact model and memory are in the platform documentation. Request them with --gres=gpu:N.",
     "docs", "local_official", "yes", "yes"),
    ("Operations, Reliability, and Incidents",
     "I could not log in for an hour this morning. Was there an outage?",
     "A login node was likely under high load and rebalanced; service returns to normal. Check the status page for confirmed incidents.",
     "tickets", "none", "yes", "no"),
    ("Operations, Reliability, and Incidents",
     "Is any maintenance scheduled that could interrupt a long run?",
     "Check the service status page for maintenance windows. Jobs that would overlap a window are held until after it.",
     "docs", "local_official", "yes", "no"),
    ("Finance, Procurement, and Admin",
     "How do I request more compute allocation for my grant?",
     "Submit a resource request referencing your grant code; allocations are reviewed against capacity and funding.",
     "docs", "local_official", "yes", "no"),
    ("Secure and Trusted Computing",
     "My project handles sensitive data. How do I get the trusted research environment?",
     "Access requires an approved data-handling agreement and additional onboarding; the team guides you through the governance steps once it is in place.",
     "docs", "local_official", "yes", "no"),
    ("Research and Organisational Context",
     "What acknowledgement text should I use to cite the HPC facility in a paper?",
     "Use the acknowledgement wording on the service's documentation site, including the relevant facility grant references.",
     "docs", "external_official", "no", "no"),
]


def main() -> None:
    out = Path("data/synthetic/eval")
    out.mkdir(parents=True, exist_ok=True)

    records = []
    for i, (cat, q, ref, src, scope, valid, attach) in enumerate(EVAL, start=1):
        # 70/30 dev/test split, deterministic by index.
        split = "test" if i % 10 >= 7 else "dev"
        records.append({
            "id": f"SYN-{i:03d}",
            "user_input": q,
            "reference": ref,
            "reference_contexts": [ref],
            "split": split,
            "category": cat,
            "source_needed": src,
            "docs_scope_needed": scope,
            "validity_sensitive": valid,
            "attachment_dependent": attach,
        })

    with (out / "ragas_one_hop_synthetic.jsonl").open("w") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    for sp in ("dev", "test"):
        with (out / f"ragas_one_hop_synthetic.{sp}.jsonl").open("w") as fh:
            for r in records:
                if r["split"] == sp:
                    fh.write(json.dumps(r) + "\n")

    manifest = {
        "synthetic": True,
        "note": "Fully synthetic ragas one-hop evaluation set over the synthetic ticket corpus. No real query, answer, or ticket is represented.",
        "total": len(records),
        "dev": sum(r["split"] == "dev" for r in records),
        "test": sum(r["split"] == "test" for r in records),
        "validity_sensitive_yes": sum(r["validity_sensitive"] == "yes" for r in records),
        "attachment_dependent_yes": sum(r["attachment_dependent"] == "yes" for r in records),
    }
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {len(records)} eval items ({manifest['dev']} dev / {manifest['test']} test)")
    print(f"  validity_sensitive=yes: {manifest['validity_sensitive_yes']}/{len(records)}")
    print(f"  attachment_dependent=yes: {manifest['attachment_dependent_yes']}/{len(records)}")


if __name__ == "__main__":
    main()
