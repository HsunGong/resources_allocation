import random
import numpy as np

from resource.run import ResourceScheduler, Job, Host, Core, Block, list2int

import pickle
# from copy import deepcopy
def deepcopy(obj):
    return pickle.loads(pickle.dumps(obj))

def try_assign_job_core(rs: ResourceScheduler, num_core) -> ResourceScheduler:
    rs = deepcopy(rs)
    host = rs.hosts[0]
    cores = host.cores # START at finish_time = 0
    jobs = rs.jobs

    # jobids = [i for i in range(rs.numJob)]
    # for all possible jobs? or sort by best?
    for job in jobs:
        blockids = [i for i in range(job.num_block)]
        # sort by least time
        # sorted(job.num_block, key=lambda block: block.data * rs.speedup(num_core) / job.speed)[]
        # last_blockid[0] = sorted(blockids, key=lambda blockid: job.blocks[blockid].end_time, reverse=True)
        # job.blocks[blockid].data * rs.speedup(num_core) / job.speed,
        
        ## Local Scheduling
        while any(block.end_time != np.inf for block in job.blocks):
            flag = False
            block = max(job.blocks, key=lambda block: block.end_time)

            _time = block.data * rs.speedup(num_core) / job.speed
            core = min(host.cores, key=lambda core: core.finish_time)
            if core.finish_time < block.end_time - _time:
                # try allocate
                core.add_block(block, _time)
                flag = True
            if not flag: # can not allocate
                break

    return rs

def list_schedule(rs: ResourceScheduler):
    # sort jobs by max of different cores
    host:Host = rs.hosts[0]
    for num_core in range(host.num_core):
        rs = try_assign_job_core(rs, num_core)
        rs.outputSolutionFromBlock()
        rs.plot()
