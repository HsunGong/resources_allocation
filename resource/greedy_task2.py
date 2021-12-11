import random
from typing import List
import numpy as np
from resource.greedy import shortest_end_time
from resource.plot import plot

from resource.run import ResourceScheduler, Job, Host, Core, Block, list2int

global transmission_speed


def multi_time_schedule(times: List[int], num_cores: int):
    finish_time_per_core = [0] * num_cores
    ans_idx = [-1] * len(times)

    time_idx_sorted = list(reversed(np.argsort(times)))

    for i in time_idx_sorted:
        core_idx = np.argmin(finish_time_per_core)
        finish_time_per_core[core_idx] += times[i]
        ans_idx[i] = core_idx

    return ans_idx


def greedy(rs: ResourceScheduler):
    global transmission_speed
    transmission_speed = rs.St

    num_block = max(len(job.blocks) for job in rs.jobs)
    num_core = sum(host.num_core for host in rs.hosts)

    jobids = [i for i in range(rs.numJob)]
    jobids = sorted(
        jobids,
        key=lambda id: sum(block.data for block in rs.jobs[id].blocks) / rs.
        jobs[id].speed,
        reverse=True,
    )  # possible core is different. -> ~~speed

    for jid in jobids:
        job = rs.jobs[jid]
        accum_core = 0
        all_used = []

        max_start_time = 0
        for host in rs.hosts:
            block_in_host = list(
                filter(lambda block: host.hostid == block.host, job.blocks))
            # print(f"Job{jid} has {block_in_host} block in host {host}")
            print(
                host,
                [block.host for block in job.blocks],
            )
            if len(block_in_host) == 0: continue

            num_core = min(host.num_core, len(block_in_host))
            accum_core += num_core

            used_cores = sorted(host.cores,
                                key=lambda core: core.finish_time)[:num_core]
            print(f"Job{jid} use: {used_cores}")
            start_time = used_cores[-1].finish_time
            max_start_time = max(start_time, max_start_time)

            all_used.append([host, used_cores])

        # Sync Start Time
        for host, used_cores in all_used:
            for core in used_cores:
                print(host, core, "updpate to", max_start_time)
                core.finish_time = max_start_time

        # Speed computation
        speed = job.speed * (1 - rs.alpha * (accum_core - 1))

        for host, used_cores in all_used:
            block_in_host = list(
                filter(lambda block: host.hostid == block.host, job.blocks))
            if len(block_in_host) == 0: continue

            num_core = min(host.num_core, len(block_in_host))
            # estimate end time
            for block in block_in_host:
                # find ealiest core
                core = min(used_cores, key=lambda core: core.finish_time)
                # if core.finish_time == 0:
                #     print("WARN", core)

                # finish_time --> core.finish_time + block.data / speed
                core.add_block(block, add_finish_time=block.data / speed)

        finish_time_now = 0
        for _, cores in all_used:
            for core in cores:
                finish_time_now = max(finish_time_now, core.finish_time)

        for _, cores in all_used:
            for core in cores:
                core.finish_time = finish_time_now
        for host, _ in all_used:
            host.finish_time = max(host.finish_time, finish_time_now)
        # update job finish
        job.finish_time = finish_time_now
        # update host finish


def greedy_trans(rs: ResourceScheduler, max_allowed_core=2):
    global transmission_speed
    transmission_speed = rs.St

    num_block = max(len(job.blocks) for job in rs.jobs)
    # num_core = sum(host.num_core for host in rs.hosts)

    jobids = [i for i in range(rs.numJob)]
    jobids = sorted(
        jobids,
        key=lambda id: sum(block.data for block in rs.jobs[id].blocks) / rs.
        jobs[id].speed,
        reverse=True,
    )  # possible core is different. -> ~~speed

    all_cores = []
    for host in rs.hosts:
        all_cores.extend(host.cores)

    num_jobs, num_core = len(jobids), min(len(all_cores), max_allowed_core)
    end_time = np.zeros((num_jobs, num_core + 1))
    bubbles = np.zeros_like(end_time)
    end_time_coreid = np.zeros((num_jobs, num_block, num_core + 1))
    for i in range(num_jobs):
        for j in range(1, num_core + 1):
            end_time[i][j], bubbles[i][j] = shortest_end_time(
                rs, i, rs.jobs[i], j, end_time_coreid)

    for jid in jobids:
        job = rs.jobs[jid]
        # XXX
        max_core = min(
            len(job.blocks), len(all_cores), 
            np.argmin(end_time[jid][1:]) + 1,
            max_allowed_core)  # max core = 3
        block_by_host = [0 for _ in rs.hosts]
        for block in job.blocks:
            block_by_host[block.host] += block.data

        # penalty is : if data is large in this host, the core has high priority
        used_cores = sorted(all_cores,
                            key=lambda core: core.finish_time 
                            # - block_by_host[core.hostid] / transmission_speed
                                )[:max_core]
        used_hostids = set(core.hostid for core in used_cores)
        speed = job.speed * (1 - rs.alpha * (len(used_cores) - 1))

        # Sync Start Time
        start_time = max(used_cores,
                         key=lambda core: core.finish_time).finish_time
        for core in used_cores:
            core.finish_time = start_time

        blocks = sorted(job.blocks, key=lambda block: block.data / speed, reverse=True)
        for block in blocks:
            # assign block -> min finish core
            # XXX: 考虑下传输
            core = min(used_cores,
                       key=lambda core: core.finish_time 
                    #    + block.data / transmission_speed if core.hostid != block.host else 0
            )
            core.add_block(block, block.data / speed,
                           block.data / transmission_speed)

        finish_time_now = max(used_cores,
                              key=lambda core: core.finish_time).finish_time

        # update job finish
        job.finish_time = finish_time_now
        # update host finish
        for core in used_cores:
            _hid = core.hostid
            rs.hosts[_hid].finish_time = max(rs.hosts[_hid].finish_time,
                                             finish_time_now)
    # 把所有传输的时间放到最前面。
    for host in rs.hosts:
        for core in host.cores:
            i = len(core.blocks) - 1
            while True:
                if i <= 0:
                    break
                if i >= 1 and core.blocks[i].jobid == core.blocks[i - 1].jobid:
                    core.blocks[i - 1].start_time = core.blocks[
                        i - 1].start_time + core.blocks[
                            i].start_time - core.blocks[i - 1].end_time
                    core.blocks[i - 1].end_time = core.blocks[i].start_time
                i = i - 1


def single_core(rs):
    """
    a job use a single core, no parallel

    The main change compared to task1 is in the selection of the host
    only consider the hosts that the block started on , use 
    exist_in_host[i][j]: to judge whether job i can start at host j
    num_host_in: to record the core number of each host
    finish_time_per_core[i][j]:     has changed to record core j in some hosts i, the array can be smaller 
        (sum(len(host.cores) for host in rs.hosts) to max num_host_in)

    (another way to think is that if the job dont have any block in the earlist core's host, then add the smallest block's transmission time )
    (this still works beacuse the according to util the max transimission time is 200/500 and the mininal run time is 50/80
    it is impossible we need to wait for more than 1 block's transimission)
    """

    job_time_single_core = [
        sum(blk.data for blk in job.blocks) / job.speed for job in rs.jobs
    ]

    #allocated_cores = multi_time_schedule(job_time_single_core,
    #                                      sum(len(host.cores) for host in rs.hosts)) #sum of all cores
    #hid = 0
    #cur_host = rs.hosts[hid]
    num_host = len(rs.hosts)
    num_host_in = np.zeros(num_host)
    for i in range(len(rs.hosts)):
        #print(rs.hosts[i].cores)
        num_host_in[i] = len(rs.hosts[i].cores)  #record each host's corenum
    finish_time_per_core = np.zeros(
        (num_host, sum(len(host.cores) for host in rs.hosts)))
    exist_in_host = np.zeros((len(rs.jobs), num_host))
    ans_host_idx = [-1] * len(job_time_single_core)
    ans_core_idx = [-1] * len(job_time_single_core)

    time_idx_sorted = list(reversed(np.argsort(job_time_single_core)))

    for i in time_idx_sorted:
        for block in rs.jobs[i].blocks:
            exist_in_host[i][block.host] = True
        core_idx = 0
        core_time = np.inf
        hid = 0
        for j in range(num_host):
            if exist_in_host[i][j]:
                for k in range(int(num_host_in[j])):
                    if finish_time_per_core[j][k] < core_time:
                        core_time = finish_time_per_core[j][k]
                        hid = j
                        core_idx = k
        #core_idx = np.argmin(finish_time_per_core)
        finish_time_per_core[hid][core_idx] += job_time_single_core[i]
        ans_host_idx[i] = hid
        ans_core_idx[i] = core_idx

    for i, i_core in enumerate(ans_core_idx):
        job = rs.jobs[i]
        hid = ans_host_idx[i]
        cid = ans_core_idx[i]
        #print(hid)
        #print(cid)
        cur_host = rs.hosts[hid]
        core = cur_host.cores[cid]
        for block in job.blocks:
            # update start/end
            block.hostid = hid
            block.coreid = cid
            block.start_time = core.finish_time
            block.end_time = core.finish_time + block.data / job.speed
            # update core start/end
            core.add_block(block, block.data / job.speed)
        job.finish_time = core.finish_time
        cur_host.finish_time = max(cur_host.finish_time, core.finish_time)


def multi_time_schedule(times: List[int], num_cores: int):
    """Schedule for multi time-blocks
    times: a list of int
    num_cores: how many cores to use

    Return: a list of len(times) which is each time-block's index of core
    """

    finish_time_per_core = [0] * num_cores
    ans_idx = [-1] * len(times)

    time_idx_sorted = list(reversed(np.argsort(times)))

    for i in time_idx_sorted:
        core_idx = np.argmin(finish_time_per_core)
        finish_time_per_core[core_idx] += times[i]
        ans_idx[i] = core_idx

    return ans_idx
