#!/usr/bin/env python3
"""Generate a fully synthetic HPC support-ticket corpus for Polaris.

This produces fabricated tickets in the same Jira issue schema that the
ingestion pipeline consumes, so they are drop-in ingestable. Nothing in the
output is derived from real support tickets: all identities, identifiers,
and scenarios are invented. Real *public* infrastructure facts (CSD3 partition
names, public software packages) are used only to keep the tickets realistic.

The category mix mirrors the real benchmark's topic distribution so the
synthetic corpus is representative without containing any real content.

Usage:
    python scripts/generate_synthetic_corpus.py [--count 120] [--seed 42]
"""
from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timedelta
from pathlib import Path

# Topic distribution mirrors the real 100-item benchmark (counts -> weights).
CATEGORY_WEIGHTS = {
    "Schedulers and Runtime": 24,
    "Storage, Filesystems, and Data Services": 20,
    "Access, Authentication, and Identity": 19,
    "Platforms and Systems": 18,
    "Operations, Reliability, and Incidents": 8,
    "Finance, Procurement, and Admin": 6,
    "Secure and Trusted Computing": 4,
    "Research and Organisational Context": 1,
}

PARTITIONS = ["icelake", "icelake-himem", "cclake", "sapphire", "ampere", "ampere-himem"]
SOFTWARE = ["GROMACS", "OpenFOAM", "PyTorch", "TensorFlow", "GCC", "OpenMPI", "R", "LAMMPS", "Quantum ESPRESSO"]
STATUSES = ["Resolved", "Closed", "Resolved", "Closed", "Open", "In Progress"]

# Per-category scenario pools: (summary, problem, resolution). Placeholders in
# braces are filled with randomised synthetic values.
SCENARIOS: dict[str, list[tuple[str, str, str]]] = {
    "Schedulers and Runtime": [
        ("Job {job} pending for hours on {part}",
         "My job {job} has been stuck in PENDING on the {part} partition for over six hours. squeue shows reason '(Priority)'. Is the queue just busy or have I requested something invalid?",
         "The {part} partition was heavily subscribed at the time. Your resource request was valid; the job started once nodes freed up. For shorter waits, request fewer nodes or use a less busy partition."),
        ("Job killed by OOM on {part}",
         "Job {job} keeps dying with 'slurmstepd: error: Detected 1 oom-kill event'. I requested 1 node on {part}. How do I give it more memory?",
         "The node ran out of RAM. Either request a -himem partition, lower the per-task memory footprint, or set --mem explicitly. We recommend starting with --mem=64G and tuning from there."),
        ("sbatch script rejected: invalid partition",
         "When I run sbatch I get 'Invalid partition name specified'. My script has #SBATCH -p {part}old. What's the correct name?",
         "The partition you named does not exist. The current valid partitions are listed by 'sinfo -s'. Use {part} instead."),
        ("GPU not visible inside job on {part}",
         "I submitted to {part} with --gres=gpu:1 but nvidia-smi reports no devices inside the job. The {sw} run falls back to CPU.",
         "You also need to load the cuda module and ensure your submission requests the gpu QOS. Add 'module load cuda' and resubmit; nvidia-smi then lists the allocated device."),
        ("Array job {job} only runs first task",
         "My array job {job} with --array=1-50 only ever executes task 1. The rest disappear without output.",
         "Your output path used %j rather than %A_%a, so tasks overwrote each other. Use #SBATCH -o logs/%A_%a.out and the array tasks log separately."),
        ("Job exceeded time limit on {part}",
         "Job {job} was cancelled with 'DUE TO TIME LIMIT'. I set --time=12:00:00 but the {sw} simulation needs longer.",
         "Request a longer wall time up to the partition maximum (see 'scontrol show partition {part}'), or checkpoint the {sw} run so it can resume across multiple shorter jobs."),
    ],
    "Storage, Filesystems, and Data Services": [
        ("Quota exceeded on project directory",
         "I can no longer write to my project area; everything fails with 'Disk quota exceeded'. quota shows I'm at 100%. How do I get more space?",
         "Your project is at its storage limit. Either clear unneeded data or request a quota increase via the resource request form. Large intermediate files can go in the scratch area, which is not quota-limited."),
        ("Files missing after job on {part}",
         "Output files my {sw} job wrote on {part} are not visible on the login node. Did the job lose them?",
         "The job wrote to node-local /tmp, which is cleared when the job ends. Write outputs to your project or scratch directory instead so they persist."),
        ("Very slow I/O reading many small files",
         "Reading a dataset of ~2 million small files is extremely slow and stalls my {sw} job. Is the filesystem degraded?",
         "Many tiny files are a known anti-pattern on parallel filesystems. Pack them into a few larger archives (tar/HDF5) or stage them to local scratch at job start for much better throughput."),
        ("Permission denied on shared project folder",
         "A colleague added me to the project but I get 'Permission denied' opening the shared directory.",
         "Group membership had not propagated to the filesystem ACL. We refreshed it; please log out and back in, after which access works."),
        ("How do I transfer 500GB to the cluster?",
         "What's the recommended way to move ~500GB from my institution to my project area?",
         "Use rsync over SSH for resumable transfers, or the data-transfer nodes for large moves. Avoid the login nodes for bulk copies as they are rate-limited."),
    ],
    "Access, Authentication, and Identity": [
        ("SSH fails with Permission denied (publickey)",
         "I can't log in: 'Permission denied (publickey)'. My key worked last week. Nothing changed on my side.",
         "Your public key was missing from the new home directory after a migration. We re-added it; please retry. Ensure your private key has 600 permissions."),
        ("MFA device lost, locked out",
         "I lost the phone with my authenticator app and can't complete login. How do I reset MFA?",
         "We reset your MFA enrolment after identity verification. Re-enrol a new device at first login via the self-service portal."),
        ("Account appears to have expired",
         "Login is refused entirely and I'm told my account is inactive. My grant is still running.",
         "The account had reached its review date. We extended it in line with your active allocation; access is restored."),
        ("New user needs cluster access",
         "I've just joined the group and need an account on the HPC service. What do I do?",
         "Your PI should add you to the project on the resource portal; an account is then provisioned automatically. You'll receive onboarding instructions by email."),
        ("VPN required to reach login nodes?",
         "From off-site I can't reach the login nodes. Do I need a VPN?",
         "Off-campus access requires the institutional VPN or the documented jump host. Connect to the VPN first, then SSH as normal."),
    ],
    "Platforms and Systems": [
        ("Which partition should I use for {sw}?",
         "I'm running {sw} and unsure which partition fits best. My jobs are memory-heavy.",
         "For memory-bound {sw} workloads use an -himem partition; for standard runs {part} is appropriate. See the platform docs for per-node memory figures."),
        ("Is there a newer CPU partition available?",
         "My code benefits from newer instruction sets. Which is the most recent CPU partition?",
         "The {part} partition is the most recent CPU generation in service. Rebuild your code with an architecture-tuned flag to take advantage of it."),
        ("GPU type on {part}?",
         "What GPU model do the {part} nodes have, and how much memory per GPU?",
         "Those nodes carry data-centre GPUs; exact model and memory are listed in the platform documentation. Request them with --gres=gpu:N."),
        ("Default software stack version",
         "Which compiler and MPI are loaded by default, and how do I pick a specific version?",
         "Defaults are shown by 'module list' on login. Use 'module avail' to see all versions and 'module load name/version' to pin one."),
    ],
    "Operations, Reliability, and Incidents": [
        ("Login nodes unresponsive this morning",
         "I couldn't log in for about an hour this morning; connections hung. Was there an outage?",
         "A login node was under unusually high load and was rebalanced. Service is back to normal; apologies for the disruption."),
        ("Is there scheduled maintenance this week?",
         "I'm planning a long run. Is any downtime scheduled that might interrupt it?",
         "A maintenance window is planned; see the service status page for exact dates. Jobs that would overlap the window will be held until after it."),
        ("Filesystem felt degraded during my run",
         "My {sw} job on {part} ran far slower than usual and I suspect a filesystem issue.",
         "There was a transient storage performance incident in that period, now resolved. You may wish to rerun the affected job."),
    ],
    "Finance, Procurement, and Admin": [
        ("How do I request more compute allocation?",
         "My project is running low on compute credits. How do I request more for the current grant?",
         "Submit a resource request referencing your grant code; allocations are reviewed against available capacity and your funding."),
        ("Can my group buy dedicated nodes?",
         "We'd like guaranteed capacity. Is buying into the cluster possible and how is it billed?",
         "Yes, the service offers a buy-in model with dedicated access. The team can provide costs and the procurement process on request."),
        ("Which grant code should jobs be charged to?",
         "I work across two projects. How do I make sure jobs are charged to the right grant?",
         "Set the account in your submission with '#SBATCH -A <project>'. Each project maps to its grant; jobs without an account use your default."),
    ],
    "Secure and Trusted Computing": [
        ("Access to the secure data environment",
         "My project handles sensitive data and needs the trusted research environment. How do I get access?",
         "Access requires an approved data-handling agreement and additional onboarding. We'll guide you through the governance steps once the agreement is in place."),
        ("Handling a restricted dataset on the cluster",
         "Are restricted datasets allowed on the standard filesystem or must they stay in the secure enclave?",
         "Restricted data must remain within the secure environment and may not be copied to standard storage. The enclave provides compliant compute and storage."),
    ],
    "Research and Organisational Context": [
        ("How should I acknowledge the HPC service in papers?",
         "What's the correct acknowledgement text to cite the HPC facility in a publication?",
         "Please use the acknowledgement wording on the service's documentation site, including the relevant grant references for the facility."),
    ],
}


def _adf(text: str) -> dict:
    """Wrap plain text in a minimal Atlassian Document Format paragraph."""
    return {"type": "doc", "version": 1,
            "content": [{"type": "paragraph", "content": [{"type": "text", "text": text}]}]}


def _fill(template: str, rng: random.Random) -> str:
    return template.format(
        job=f"{rng.randint(1000000, 9999999)}",
        part=rng.choice(PARTITIONS),
        sw=rng.choice(SOFTWARE),
    )


def _counts(total: int) -> dict[str, int]:
    weight_sum = sum(CATEGORY_WEIGHTS.values())
    counts = {c: max(1, round(total * w / weight_sum)) for c, w in CATEGORY_WEIGHTS.items()}
    return counts


def generate(total: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    counts = _counts(total)
    tickets: list[dict] = []
    next_key = 1000
    base_date = datetime(2024, 1, 1)
    for category, n in counts.items():
        for _ in range(n):
            summary_t, problem_t, resolution_t = rng.choice(SCENARIOS[category])
            summary = _fill(summary_t, rng)
            problem = _fill(problem_t, rng)
            resolution = _fill(resolution_t, rng)

            created = base_date + timedelta(days=rng.randint(0, 600), minutes=rng.randint(0, 1439))
            updated = created + timedelta(hours=rng.randint(1, 240))
            status = rng.choice(STATUSES)
            resolved = status in {"Resolved", "Closed"}

            reporter = f"researcher{rng.randint(1, 400):03d}@example.org"
            ticket = {
                "key": f"DEMO-{next_key}",
                "_synthetic": True,
                "_category": category,
                "fields": {
                    "key": f"DEMO-{next_key}",
                    "summary": summary,
                    "description": _adf(problem),
                    "status": {"name": status},
                    "created": created.strftime("%Y-%m-%dT%H:%M:%S.000+0000"),
                    "updated": updated.strftime("%Y-%m-%dT%H:%M:%S.000+0000"),
                    "resolutionDate": updated.strftime("%Y-%m-%dT%H:%M:%S.000+0000") if resolved else None,
                    "reporter": {"emailAddress": reporter},
                    "creator": {"emailAddress": reporter},
                    "assignee": {"emailAddress": "support.analyst@example.org"},
                    "customfield_10042": [{"emailAddress": "pi.lead@example.org"}],
                    "comment": {"comments": [
                        {
                            "author": {"emailAddress": "support.analyst@example.org"},
                            "body": _adf(resolution),
                            "created": updated.strftime("%Y-%m-%dT%H:%M:%S.000+0000"),
                        }
                    ] if resolved else []},
                },
            }
            tickets.append(ticket)
            next_key += 1
    rng.shuffle(tickets)
    return tickets


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--count", type=int, default=120, help="approximate number of tickets")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data/synthetic", help="output directory")
    args = ap.parse_args()

    out = Path(args.out)
    (out).mkdir(parents=True, exist_ok=True)
    tickets = generate(args.count, args.seed)

    with (out / "tickets.jsonl").open("w") as fh:
        for t in tickets:
            fh.write(json.dumps(t) + "\n")

    counts = _counts(args.count)
    manifest = {
        "synthetic": True,
        "note": "Fully synthetic, fabricated HPC support tickets. No real ticket, user, or content is represented. Generated by scripts/generate_synthetic_corpus.py.",
        "seed": args.seed,
        "total_tickets": len(tickets),
        "category_distribution": counts,
        "schema": "Jira issue (key, fields.{summary, description[ADF], status, created, updated, resolutionDate, comment.comments[], reporter/assignee/creator/customfield_10042})",
    }
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {len(tickets)} synthetic tickets to {out/'tickets.jsonl'}")
    for c, n in counts.items():
        print(f"  {n:>3}  {c}")


if __name__ == "__main__":
    main()
