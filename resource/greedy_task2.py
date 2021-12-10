import random
from typing import List
import numpy as np
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
    global transmission_speed;transmission_speed = rs.St

    num_block = max(len(job.blocks) for job in rs.jobs)
    num_core = sum(host.num_core for host in rs.hosts)

    jobids = [i for i in range(rs.numJob)]
    jobids = sorted(
        jobids,
        key=lambda id: sum(block.data for block in rs.jobs[id].blocks) / rs.jobs[id].speed,
        reverse=True,
    ) # possible core is different. -> ~~speed

    for jid in jobids:
        job = rs.jobs[jid]
        accum_core = 0
        all_used = []
        
        max_start_time = 0
        for host in rs.hosts:
            block_in_host = list(filter(lambda block: host.hostid == block.host, job.blocks))
            # print(f"Job{jid} has {block_in_host} block in host {host}")
            if len(block_in_host) == 0: continue
            
            num_core = min(host.num_core, len(block_in_host))
            accum_core += num_core

            used_cores = sorted(host.cores, key=lambda core: core.finish_time)[:num_core]
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
            block_in_host = list(filter(lambda block: host.hostid == block.host, job.blocks))
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
        for host,_ in all_used:
            host.finish_time = max(host.finish_time, finish_time_now)
        # update job finish
        job.finish_time = finish_time_now
        # update host finish
