import random
from typing import List
import numpy as np

from resource.run import ResourceScheduler, Job, Host, Core, Block, list2int

def shortest_end_time(rs, job_i, job, num_core, end_time_coreid):
    """
    Return the shortest time when job_i can only use j cores
    and record the index of each block on those cores.
    """
    assert num_core > 0
    schedule_res = multi_time_schedule(
        [blk.data for blk in job.blocks],
        num_core
    )
    print(f'Schedule for Job {job_i}:', schedule_res)
    core_data_size = [0] * num_core
    for core_idx, blk in zip(schedule_res, job.blocks):
        # Assign end_time_coreid.
        end_time_coreid[job_i][blk.blockid][num_core] = core_idx
        core_data_size[core_idx] += blk.data
    num_core_used = len(set(schedule_res))
    return max(core_data_size) / num_core_used

    # max_time = 0
    # speed = job.speed * (1 - rs.alpha * (num_core - 1))
    # if len(job.blocks) <= num_core:
    #     for i in range(len(job.blocks)):
    #         max_time = max(max_time, job.blocks[i].data)
    #         end_time_coreid[job_i][i][num_core] = i
    #     ans = max_time / speed
    #     return ans
    # core_time = np.zeros(num_core)
    # for block in job.blocks:
    #     short_time = 0xFFFFFF
    #     cid = 0
    #     for i in range(num_core):
    #         if core_time[i] < short_time:
    #             short_time = core_time[i]
    #             cid = i
    #     core_time[cid] = core_time[cid] + block.data
    #     end_time_coreid[job_i][block.blockid][num_core] = cid
    # for i in range(num_core):
    #     max_time = max(max_time, core_time[i])
    # ans = max_time / speed


def greedy_schedule(rs: ResourceScheduler):
        ## Only task 1 is supported
        assert rs.taskID == 1

        taskType = rs.taskID
        if taskType == 1:
            hid = 0
            cur_host = rs.hosts[hid]
            num_jobs = len(rs.jobs)
            num_core = cur_host.num_core

            num_block = max(len(job.blocks) for job in rs.jobs)
            end_time = np.zeros((num_jobs, num_core + 1))
            end_time_coreid = np.zeros((num_jobs, num_block, num_core + 1))
            for job in rs.jobs:
                job.blocks = sorted(job.blocks, key=lambda x: x.data, reverse=True)

            for i in range(num_jobs):
                for j in range(1, num_core + 1):
                    end_time[i][j] = shortest_end_time(rs, i, rs.jobs[i], j, end_time_coreid)

            job_block_sort_idx = reversed(np.argsort([len(job.blocks) for job in rs.jobs]))
            # job_block_sort_idx = range(len(rs.jobs))
            for job_idx in job_block_sort_idx:
                print('-='*10, job_idx)
                job = rs.jobs[job_idx]
                num_use = min(num_core, len(job.blocks))
                core_use = 0
                finish_time_now = 0
                speed = job.speed * (1 - rs.alpha * (num_use - 1))
                for j, block in enumerate(job.blocks):
                    block.hostid = hid
                    block.coreid = end_time_coreid[job_idx][j][num_use] + core_use
                    core = cur_host.cores[int(block.coreid)]
                    block.start_time = core.finish_time
                    block.end_time = core.finish_time + block.data / speed
                    finish_time_now = max(finish_time_now, block.end_time)
                    core.add_block(block, block.data / speed)
                for core in cur_host.cores:
                    core.finish_time = finish_time_now
                # update job finish
                job.finish_time = core.finish_time
                # update host finish
                cur_host.finish_time = max(cur_host.finish_time, core.finish_time)
                
                
def single_core(rs):
    job_time_single_core = [
        sum(blk.data for blk in job.blocks) / job.speed
        for job in rs.jobs
    ]

    allocated_cores = multi_time_schedule(
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

