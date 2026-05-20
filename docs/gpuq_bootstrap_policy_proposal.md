# GPU Bootstrap Policy Proposal

## Goal

Improve fairness on `gpuq`/`gpudev` without leaving GPUs idle.

Desired behavior:

- If a user currently has no running GPU usage, they should be able to get at least `n` GPU units started quickly.
- Once a user already has `n` bootstrap-eligible GPU units in use, the rest of their queue should compete normally.
- If nobody needs a bootstrap start, all GPUs should remain available for normal backfill and high-throughput use.
- Users should not need to learn a new workflow.

## Current State

As of `2026-04-29`:

- `gpuq`:
  - `DefaultTime=06:00:00`
  - `MaxTime=1-00:00:00`
- `gpudev`:
  - `DefaultTime=06:00:00`
  - `MaxTime=1-00:00:00`

Observed behavior looks like standard fairshare/priority scheduling. There is no visible policy that ensures a user with zero running GPU usage gets their next GPU allocation started before users who already hold multiple GPUs.

## Why Not Hard Per-User Caps

Hard per-user GPU caps are simple, but they waste capacity:

- if only a few users are active, GPUs can sit idle
- users with urgent single-GPU debug jobs still wait behind long queues
- they reduce throughput even when the cluster is lightly loaded

That is the wrong tradeoff here.

## Recommended Policy

Use a **bootstrap QoS** with **higher priority**, assigned automatically to a user's first `n` GPU units when they currently have fewer than `n` running GPU units.

Recommended starting values:

- bootstrap target: `n = 8 GPUs`
- bootstrap QoS name: `gpu_bootstrap`

Separately, shorten walltimes for **all** GPU jobs from the current 24-hour max:

- recommended GPU default: keep `6:00:00`
- recommended GPU max: `12:00:00`

If there is a real need for longer GPU runs, those can move to a separate long-GPU QoS/partition instead of letting the main shared GPU queues stretch to 24 hours.

## Files In This Repo

This repo now contains a deployable sample bundle:

- Lua hook:
  - [job_submit.lua](/gpfs/data/huo-lab/Image/annawoodard/prima/ops/slurm/job_submit.lua)
- state updater:
  - [update_bootstrap_gpu_running.py](/gpfs/data/huo-lab/Image/annawoodard/prima/ops/slurm/update_bootstrap_gpu_running.py)
- systemd service:
  - [bootstrap-gpu-running.service](/gpfs/data/huo-lab/Image/annawoodard/prima/ops/slurm/bootstrap-gpu-running.service)
- systemd timer:
  - [bootstrap-gpu-running.timer](/gpfs/data/huo-lab/Image/annawoodard/prima/ops/slurm/bootstrap-gpu-running.timer)

## Why This Works

This policy gives a starved user a fast on-ramp without reserving capacity:

- users with zero running GPU usage get a temporary priority boost
- heavy users can still fill the cluster when nobody else needs the boost
- no GPUs are reserved or held idle
- the cluster remains more backfill-friendly because long GPU jobs are capped uniformly

## Implementation Shape

### 1. Add a higher-priority QoS

Create a QoS such as `gpu_bootstrap` with:

- higher priority than `normal`
- the same walltime policy as normal GPU jobs
- optionally `MaxJobsPU` or `MaxTRESPU` for this QoS only, but keep it generous enough that it does not create idle capacity

This QoS should be used only for bootstrap-eligible jobs.

### 2. Keep the normal queue broad

Keep normal GPU jobs eligible for the full partition. Do not reserve nodes. Do not use a strict per-user cap as the main control.

### 3. Assign bootstrap QoS automatically

Use `job_submit.lua` to apply the bootstrap QoS behind the scenes when:

- the job is a GPU job
- the target partition is `gpuq` or `gpudev`
- the user currently has fewer than `n` running GPU units
- the user currently has fewer than `n` bootstrap-tagged GPU units already in flight
- the current submission still fits within the remaining bootstrap GPU budget
- the job does not already request a special QoS

### 4. Do not shell out from `job_submit.lua`

`job_submit.lua` runs inside `slurmctld` under internal locks. It must stay fast.

Do **not** call `squeue`, `sacct`, or other shell commands from the Lua hook.

Instead, maintain a tiny local state file, for example:

`/var/spool/slurm/bootstrap_gpu_running.tsv`

Format:

```text
1001|0|0
1002|1|1
1003|4|2
```

Columns:

- `uid`
- `running_gpu_gpus`
- `bootstrap_inflight_gpus`

Update it every 30-60 seconds from a root-owned systemd timer or cron job that computes:

- running GPU count per user
- bootstrap-tagged pending/running/completing GPU count per user
- filtered to `gpuq` and `gpudev`

Then let `job_submit.lua` do one cheap file read and a simple integer lookup.

## Suggested Policy Details

### Bootstrap threshold

Start with:

- bootstrap target `n = 8 GPUs`

That is usually enough to let a user start meaningful work without letting large multi-GPU jobs consume the entire bootstrap allowance.

### Walltimes

Recommended:

- `gpuq` / `gpudev`: keep `DefaultTime=06:00:00`, reduce `MaxTime=12:00:00`
- `gpu_bootstrap`: same walltime policy as `gpuq` / `gpudev`

Rationale:

- shorter maximum limits improve backfill
- keeping the same walltime policy across QoS avoids breaking user expectations
- 24 hours is too long for the cluster behavior you want

### Eligibility

Only bootstrap jobs that:

- request GPU resources
- land on `gpuq` or `gpudev`
- do not already request a special QoS

This avoids surprising users who intentionally choose a different QoS.

## Caveat

This is an approximation of "give a user their first `n` GPU units if they currently have none running."

Because `job_submit.lua` runs at submit time, not continuously, it does not perfectly reclassify old pending jobs as running counts change. That is usually acceptable in practice if:

- the state file updates frequently
- the QoS boost is only for the first `n` jobs

If the site eventually wants exact dynamic behavior, that requires a deeper scheduler-side policy than a simple submit hook.

This GPU-based design is preferable to a job-count design because it prevents a single large job from consuming the entire startup allowance. For example, with an `8 GPU` bootstrap target, a `16 GPU` job would not receive the bootstrap QoS.

## Minimal Admin Rollout

1. Create `gpu_bootstrap` QoS with higher priority.
2. Reduce GPU walltime limits from `24h` to `12h`, while keeping the same walltime policy for both normal and bootstrap GPU jobs.
3. Install a root-owned updater that writes cached GPU state to a local TSV file every minute.
4. Enable `JobSubmitPlugins=lua`.
5. Install the Lua hook and the updater service/timer.
6. Restart or reconfigure Slurm and enable the timer.
7. Watch queue behavior for 1-2 weeks and adjust:
   - bootstrap target in GPUs
   - bootstrap priority
   - partition-wide GPU walltime limit

## Exact Deployment Steps

### 1. Create the bootstrap QoS

If `normal` is currently the only QoS and has priority `0`, a reasonable starting point is:

```bash
sacctmgr add qos gpu_bootstrap Priority=1000 Description="Priority boost for first eight GPU units"
```

If the site already uses a broader priority scale, set this high enough to outrank `normal` but not so high that it overwhelms all other policy.

If accounting/QoS enforcement requires explicit association access, also grant the QoS to the relevant GPU accounts or users. For example:

```bash
sacctmgr modify account where Name=<GPU_ACCOUNT> set qos+=gpu_bootstrap
```

or, if needed at user scope:

```bash
sacctmgr modify user where Account=<GPU_ACCOUNT> set qos+=gpu_bootstrap
```

### 2. Shorten GPU walltime limits

For immediate live testing:

```bash
scontrol update PartitionName=gpuq DefaultTime=06:00:00 MaxTime=12:00:00
scontrol update PartitionName=gpudev DefaultTime=06:00:00 MaxTime=12:00:00
```

Then make the same change persistent in `slurm.conf` for the partition definitions.

Concretely, change the `gpuq` and `gpudev` partition stanzas so they contain:

```ini
PartitionName=gpuq
DefaultTime=06:00:00
MaxTime=12:00:00
```

```ini
PartitionName=gpudev
DefaultTime=06:00:00
MaxTime=12:00:00
```

If the site keeps each partition on a single line, the equivalent persistent edit is:

```ini
PartitionName=gpuq   ... DefaultTime=06:00:00 MaxTime=12:00:00 ...
PartitionName=gpudev ... DefaultTime=06:00:00 MaxTime=12:00:00 ...
```

Only those two fields need to change for this walltime policy update.

Current values are:

- `gpuq`: `DefaultTime=06:00:00`, `MaxTime=1-00:00:00`
- `gpudev`: `DefaultTime=06:00:00`, `MaxTime=1-00:00:00`

Recommended persistent values:

- `gpuq`: `DefaultTime=06:00:00`, `MaxTime=12:00:00`
- `gpudev`: `DefaultTime=06:00:00`, `MaxTime=12:00:00`

### 3. Install the Lua hook

If no other job submit plugins are configured:

```bash
# in slurm.conf
JobSubmitPlugins=lua
```

If other job submit plugins are already configured:

```bash
# in slurm.conf
JobSubmitPlugins=<existing_plugins>,lua
```

Install the file:

```bash
install -m 0644 /gpfs/data/huo-lab/Image/annawoodard/prima/ops/slurm/job_submit.lua /etc/slurm/job_submit.lua
```

Important:

- Slurm expects the file to be named exactly `job_submit.lua`
- it must live in the same configuration directory as `slurm.conf`
- if the script is invalid, `slurmctld` can fail to start

### 4. Install the updater script and timer

Install the updater:

```bash
install -d -m 0755 /usr/local/sbin
install -m 0755 /gpfs/data/huo-lab/Image/annawoodard/prima/ops/slurm/update_bootstrap_gpu_running.py /usr/local/sbin/update_bootstrap_gpu_running.py
```

Install the systemd unit and timer:

```bash
install -m 0644 /gpfs/data/huo-lab/Image/annawoodard/prima/ops/slurm/bootstrap-gpu-running.service /etc/systemd/system/bootstrap-gpu-running.service
install -m 0644 /gpfs/data/huo-lab/Image/annawoodard/prima/ops/slurm/bootstrap-gpu-running.timer /etc/systemd/system/bootstrap-gpu-running.timer
```

Prepare the state directory:

```bash
install -d -m 0755 /var/spool/slurm
```

Enable the timer:

```bash
systemctl daemon-reload
systemctl enable --now bootstrap-gpu-running.timer
systemctl start bootstrap-gpu-running.service
```

The timer writes:

```text
/var/spool/slurm/bootstrap_gpu_running.tsv
```

Format:

```text
uid|running_gpu_gpus|bootstrap_inflight_gpus
```

### 5. Reload Slurm

After the config and Lua script are in place:

```bash
scontrol reconfigure
```

If the site prefers a daemon restart path for plugin changes, use the local admin standard instead.

### 6. Validate

Check that the updater is working:

```bash
systemctl status bootstrap-gpu-running.timer
systemctl status bootstrap-gpu-running.service
cat /var/spool/slurm/bootstrap_gpu_running.tsv
```

Check that Slurm loaded the Lua hook cleanly:

```bash
grep -i job_submit /var/log/slurm/slurmctld.log | tail -50
```

Suggested functional smoke test:

1. Pick a user with zero running GPU usage.
2. Submit one short GPU job to `gpuq`.
3. Run `systemctl start bootstrap-gpu-running.service` or wait 60-90 seconds for the timer refresh.
4. Submit a second short GPU job.
5. Refresh the state file again.
6. Submit a third GPU job.
7. Verify the first two receive `QOS=gpu_bootstrap`.
8. Verify the third stays on normal QoS.
9. Verify that a job over `12h` is rejected on `gpuq`/`gpudev`.

Do not use three immediate back-to-back submissions for the first smoke test. Because the state file is refreshed out of band, a burst of submissions inside one refresh interval can temporarily over-assign bootstrap QoS. That behavior is acceptable for an approximate bootstrap policy, but it makes the first validation test ambiguous.

Also test the GPU-budget behavior directly:

1. Pick a user with zero running GPU usage.
2. Submit a short `8 GPU` job to `gpuq`.
3. Verify it can receive `QOS=gpu_bootstrap`.
4. Pick a user with zero running GPU usage.
5. Submit a short `16 GPU` job to `gpuq`.
6. Verify it does **not** receive `QOS=gpu_bootstrap`.

For example:

```bash
scontrol show job <jobid> | egrep 'QOS=|Partition=|TimeLimit='
```

If the site later wants exact per-submit accounting even under burst submission, that is the point where a deeper scheduler-side implementation becomes worthwhile.

## Expected Outcome

Compared with the current setup, this should:

- reduce starvation for users with zero running GPU usage
- preserve high utilization
- improve backfill
- keep user workflow unchanged

Users still submit to Slurm normally. The policy stays behind the hood.

## Possible Follow-On Simplification

If this policy works well, the site may be able to retire the separate debug GPU queue as a scheduling workaround.

That is one of the practical advantages of this design:

- it tries to deliver "fast first GPU job" behavior directly on the shared GPU queues
- that reduces the need for users to choose between a throughput queue and a turnaround queue
- if it performs well enough, `gpudev` may no longer need to exist as a separate queue for startup fairness

That simplification should be treated as a later operational decision, not a prerequisite for rollout.
