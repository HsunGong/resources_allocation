import random
from typing import List
import numpy as np

from resource.run import ResourceScheduler, Job, Host, Core, Block, list2int

def shortest_end_time(rs, job_i, job, cores, end_time_coreid):
    """
    Return the shortest time when job_i can only use j cores
    and record the index of each block on those cores.
    """
    if cores == 0:
        return 0
    max_time = 0
    speed = job.speed * (1 - rs.alpha * (cores - 1))
    if len(job.blocks) <= cores:
        for i in range(len(job.blocks)):
            max_time = max(max_time, job.blocks[i].data)
            end_time_coreid[job_i][i][cores] = i
        ans = max_time / speed
        return ans
    core_time = np.zeros(cores)
    for block in job.blocks:
        short_time = 0xFFFFFF
        cid = 0
        for i in range(cores):
            if core_time[i] < short_time:
                short_time = core_time[i]
                cid = i
        core_time[cid] = core_time[cid] + block.data
        end_time_coreid[job_i][block.blockid][cores] = cid
    for i in range(cores):
        max_time = max(max_time, core_time[i])
    ans = max_time / speed
    return ans

def greedy_schedule(rs: ResourceScheduler):
        ## Only task 1 is supported
        assert rs.taskID == 1

        taskType = rs.taskID
        if taskType == 1:
            hid = 0
            cur_host = rs.hosts[hid]
            num_jobs = len(rs.jobs)
            num_core = cur_host.num_core
            num_block = 0
            for job in rs.jobs:
                num_block = max(num_block, len(job.blocks))
            end_time = np.zeros((num_jobs, num_core + 1))
            end_time_coreid = np.zeros((num_jobs, num_block, num_core + 1))
            for job in rs.jobs:
                job.blocks = sorted(job.blocks, key=lambda x: x.data, reverse=True)

            for i in range(num_jobs):
                for j in range(num_core + 1):
                    end_time[i][j] = shortest_end_time(rs, i, rs.jobs[i], j, end_time_coreid)

            i = 0
            for job in rs.jobs:
                num_use = min(num_core, len(job.blocks))
                core_use = 0
                j = 0
                finish_time_now = 0
                speed = job.speed * (1 - rs.alpha * (num_use - 1))
                for block in job.blocks:
                    block.hostid = hid
                    block.coreid = end_time_coreid[i][j][num_use] + core_use
                    core = cur_host.cores[int(block.coreid)]
                    block.start_time = core.finish_time
                    block.end_time = core.finish_time + block.data / speed
                    finish_time_now = max(finish_time_now, block.end_time)
                    core.add_block(block, block.data / speed)

                    j = j + 1
                for core in cur_host.cores:
                    core.finish_time = finish_time_now
                # update job finish
                job.finish_time = core.finish_time
                # update host finish
                cur_host.finish_time = max(cur_host.finish_time, core.finish_time)

                i = i + 1
                
                
def single_core(rs):
    job_time_single_core = [
        sum(blk.data for blk in job.blocks) / job.speed
        for job in rs.jobs
    ]

    allocated_cores = multi_job_schedule(
        job_time_single_core,
        len(rs.hosts[0].cores)
    )
    hid = 0
    cur_host = rs.hosts[hid]
    for i, i_core in enumerate(allocated_cores):
        job = rs.jobs[i]
        core = cur_host.cores[i_core]
        for block in job.blocks:
            # update start/end
            block.hostid = hid
            block.coreid = i_core
            block.start_time = core.finish_time
            block.end_time = core.finish_time + block.data / job.speed
            # update core start/end
            core.add_block(block, block.data / job.speed)
        job.finish_time = core.finish_time
        cur_host.finish_time = max(cur_host.finish_time, core.finish_time)

def multi_job_schedule(times: List[int], num_cores: int):
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

