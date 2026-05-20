-- Sample Slurm job_submit.lua for hidden GPU bootstrap priority.
--
-- Intended deployment model:
-- 1. Set JobSubmitPlugins=lua in slurm.conf.
-- 2. Install this file as /etc/slurm/job_submit.lua.
-- 3. Maintain a fast local state file with:
--      uid|running_gpu_gpus|bootstrap_inflight_gpus
--    where bootstrap_inflight_gpus counts PD/R/CG jobs already tagged with the
--    bootstrap QoS.
--
-- This script intentionally does not shell out to squeue/sacct. job_submit.lua
-- runs inside slurmctld while internal locks are held, so it must stay cheap.

local log_prefix = "gpu_bootstrap"

local GPU_PARTITIONS = {
  gpuq = true,
  gpudev = true,
}

local BOOTSTRAP_QOS = "gpu_bootstrap"
local BOOTSTRAP_TARGET_GPUS = 8
local GPU_MAX_MIN = 720

local STATE_FILE = "/var/spool/slurm/bootstrap_gpu_running.tsv"

local function log_debug(fmt, ...)
  slurm.log_debug(string.format("%s: "..fmt, log_prefix, ...))
end

local function log_user(fmt, ...)
  slurm.log_user(string.format("%s: "..fmt, log_prefix, ...))
end

local function find_in_str(s, needle)
  if s == nil then
    return false
  end
  return string.find(s, needle, 1, true) ~= nil
end

local function requests_gpu(job_desc)
  return find_in_str(job_desc["tres_per_job"], "gpu")
      or find_in_str(job_desc["tres_per_node"], "gpu")
      or find_in_str(job_desc["tres_per_task"], "gpu")
      or find_in_str(job_desc["tres_per_socket"], "gpu")
      or find_in_str(job_desc["gres"], "gpu")
end

local function gpu_count_in_str(s)
  if s == nil then
    return 0
  end

  local total = 0
  for count in string.gmatch(s, "gpu[^,:=]*:(%d+)") do
    total = total + tonumber(count)
  end
  return total
end

local function requested_gpu_count(job_desc)
  -- Be conservative here: different request fields can represent overlapping
  -- views of the same GPU request, so taking the max is safer than summing.
  local counts = {
    gpu_count_in_str(job_desc["tres_per_job"]),
    gpu_count_in_str(job_desc["tres_per_node"]),
    gpu_count_in_str(job_desc["tres_per_task"]),
    gpu_count_in_str(job_desc["tres_per_socket"]),
    gpu_count_in_str(job_desc["gres"]),
  }

  local max_count = 0
  for _, count in ipairs(counts) do
    if count > max_count then
      max_count = count
    end
  end

  if max_count > 0 then
    return max_count
  end

  if requests_gpu(job_desc) then
    return 1
  end

  return 0
end

local function chosen_partition(job_desc)
  return job_desc["partition"]
end

local function is_gpu_partition(partition)
  return partition ~= nil and GPU_PARTITIONS[partition] == true
end

local function parse_running_counts(path)
  local counts = {}
  local fh = io.open(path, "r")
  if fh == nil then
    return counts
  end

  for line in fh:lines() do
    local uid_str, running_str, inflight_str = string.match(line, "^(%d+)|(%d+)|(%d+)$")
    if uid_str ~= nil and running_str ~= nil and inflight_str ~= nil then
      counts[tonumber(uid_str)] = {
        running_gpu_gpus = tonumber(running_str),
        bootstrap_inflight_gpus = tonumber(inflight_str),
      }
    end
  end

  fh:close()
  return counts
end

local function gpu_state(uid)
  local counts = parse_running_counts(STATE_FILE)
  return counts[uid] or {
    running_gpu_gpus = 0,
    bootstrap_inflight_gpus = 0,
  }
end

local function current_time_limit(job_desc)
  local time_limit = job_desc["time_limit"]
  if time_limit == nil or time_limit == slurm.NO_VAL then
    return nil
  end
  return time_limit
end

local function reject_if_over_limit(job_desc, max_minutes, message)
  local time_limit = current_time_limit(job_desc)
  if time_limit ~= nil and time_limit > max_minutes then
    log_user(message)
    return slurm.ESLURM_INVALID_TIME_LIMIT
  end
  return slurm.SUCCESS
end

local function maybe_assign_bootstrap_qos(job_desc, submit_uid)
  -- Respect explicit non-bootstrap QoS choices.
  if job_desc["qos"] ~= nil and job_desc["qos"] ~= BOOTSTRAP_QOS then
    return slurm.SUCCESS
  end

  local state = gpu_state(submit_uid)
  local requested_gpus = requested_gpu_count(job_desc)
  if requested_gpus <= 0 then
    return slurm.SUCCESS
  end

  if state["running_gpu_gpus"] >= BOOTSTRAP_TARGET_GPUS then
    return slurm.SUCCESS
  end

  if state["bootstrap_inflight_gpus"] >= BOOTSTRAP_TARGET_GPUS then
    return slurm.SUCCESS
  end

  if state["running_gpu_gpus"] + requested_gpus > BOOTSTRAP_TARGET_GPUS then
    return slurm.SUCCESS
  end

  if state["bootstrap_inflight_gpus"] + requested_gpus > BOOTSTRAP_TARGET_GPUS then
    return slurm.SUCCESS
  end

  job_desc["qos"] = BOOTSTRAP_QOS
  log_debug(
    "uid=%d assigned qos=%s requested_gpus=%d running_gpu_gpus=%d bootstrap_inflight_gpus=%d",
    submit_uid,
    BOOTSTRAP_QOS,
    requested_gpus,
    state["running_gpu_gpus"],
    state["bootstrap_inflight_gpus"]
  )
  return slurm.SUCCESS
end

local function enforce_gpu_time_policy(job_desc)
  return reject_if_over_limit(
    job_desc,
    GPU_MAX_MIN,
    "GPU jobs on gpuq/gpudev must request 12h or less"
  )
end

function slurm_job_submit(job_desc, part_list, submit_uid)
  if submit_uid == 0 then
    return slurm.SUCCESS
  end

  if not requests_gpu(job_desc) then
    return slurm.SUCCESS
  end

  local partition = chosen_partition(job_desc)
  if not is_gpu_partition(partition) then
    return slurm.SUCCESS
  end

  local rc = maybe_assign_bootstrap_qos(job_desc, submit_uid)
  if rc ~= slurm.SUCCESS then
    return rc
  end

  return enforce_gpu_time_policy(job_desc)
end

function slurm_job_modify(job_desc, job_rec, part_list, modify_uid)
  if modify_uid == 0 then
    return slurm.SUCCESS
  end

  local partition = job_desc["partition"]
  if partition == nil and job_rec ~= nil then
    partition = job_rec["partition"]
  end

  if not is_gpu_partition(partition) then
    return slurm.SUCCESS
  end

  if not requests_gpu(job_desc) and job_rec ~= nil then
    local rec_gres = job_rec["gres"]
    if not find_in_str(rec_gres, "gpu") then
      return slurm.SUCCESS
    end
  end

  return enforce_gpu_time_policy(job_desc)
end
