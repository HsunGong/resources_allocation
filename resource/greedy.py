import random
from typing import List
import numpy as np
from resource.plot import plot

from resource.run import ResourceScheduler, Job, Host, Core, Block, list2int

class A():
    def __init__(self, jobs, hosts) -> None:
        self.jobs = jobs
        self.hosts = hosts

def test_job(job, schedule_res, num_core, speed):
    from copy import deepcopy
    fake_blocks = deepcopy(job.blocks)
    core_times = [0] * num_core
    for i, (core_idx, blk) in enumerate(zip(schedule_res, fake_blocks)):
        blk.start_time = core_times[core_idx]
        core_times[core_idx] += blk.data / speed
        blk.end_time = core_times[core_idx]
        blk.hostid = 0
        blk.coreid= core_idx
        print(f"Block {blk.blockid}, cid:{core_idx}, st:{blk.start_time}, ed:{blk.end_time}")

    fake_job = Job(None, None)
    fake_job.jobid = 0
    fake_hosts = Host(None, num_core)
    fake_hosts.hostid = 0
    fake_hosts.cores = [Core(0, i) for i in range(num_core)]
    fake_job.blocks = fake_blocks
    fake_rs = A([fake_job], [fake_hosts])
    print(f'Plotting job {job.jobid}, num_core={num_core}')
    plot(fake_rs)
    input()

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
    core_data_size = [0] * num_core
    for core_idx, blk in zip(schedule_res, job.blocks):
        # Assign end_time_coreid.
        end_time_coreid[job_i][blk.blockid][num_core] = core_idx
        core_data_size[core_idx] += blk.data
    num_core_used = len(set(schedule_res))
    speed = job.speed * (1 - rs.alpha * (num_core_used - 1))
    
    # test_job(job, schedule_res, num_core, speed)

    return max(core_data_size) / speed

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
            for job in rs.jobs:
                job.blocks.sort(key=lambda x: x.data, reverse=True) # Last Finsh First

            # estimate end time
            # end_time = np.zeros((num_jobs, num_core + 1))
            # end_time_coreid = np.zeros((num_jobs, num_block, num_core + 1))
            # for i in range(num_jobs):
            #     for j in range(1, num_core + 1):
            #         end_time[i][j] = shortest_end_time(rs, i, rs.jobs[i], j, end_time_coreid)


            # job_block_sort_idx = reversed(np.argsort([len(job.blocks) for job in rs.jobs]))
            # job_block_sort_idx = range(len(rs.jobs))
            jobids = [i for i in range(rs.numJob)]
            jobids = sorted(jobids, key=lambda id: sum(block.data for block in rs.jobs[id].blocks) / rs.jobs[id].speed, reverse=True)
            print(jobids)
            for jid in jobids:
                job = rs.jobs[jid]
                num_use = min(num_core, len(job.blocks))
                speed = job.speed * (1 - rs.alpha * (num_use - 1))
                
                # used_cores = set()
                used_cores = sorted(cur_host.cores, key=lambda core: core.finish_time)[:num_use]
                # print(used_cores)
                start_time = used_cores[-1].finish_time
                
                # estimate end time
                for block in job.blocks:
                    # find ealiest core
                    core = min(cur_host.cores, key=lambda core: core.finish_time)
                    core.finish_time = start_time
                    
                    # finish_time --> core.finish_time + block.data / speed
                    core.add_block(block, add_finish_time=block.data / speed)
                    print(f"Assign Block {block} to Core {core}")
                    # used_cores.add(core)

                finish_time_now = max(used_cores, key=lambda core: core.finish_time).finish_time
                for core in used_cores:
                    core.finish_time = finish_time_now
                
                # update job finish
                job.finish_time = finish_time_now
                # update host finish
                cur_host.finish_time = max(cur_host.finish_time, finish_time_now)
                
                
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
        print(job)
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

