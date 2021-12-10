import random
from resource.run import ResourceScheduler, Job, Host, Core, Block, list2int

def rand_schedule(rs:ResourceScheduler):
    # naive
    # end_time = np.zeros(100)
    # for i in range(len(rs.jobs)):
    #     end_time[i] = 0
    for job in rs.jobs:
        hid = random.randint(0, rs.numHost - 1)
        cur_host = rs.hosts[hid]
        cid = random.randint(0, cur_host.num_core - 1)
        short_time = 0xFFFFF
        cid = 0
        for i in range(cur_host.num_core):
            if cur_host.cores[i].finish_time < short_time:
                short_time = cur_host.cores[i].finish_time
                cid = i
        core = cur_host.cores[cid]

        for block in job.blocks:
            # update start/end
            block.hostid = hid
            block.coreid = cid
            block.start_time = core.finish_time
            block.end_time = core.finish_time + block.data / job.speed
            # update core start/end
            core.add_block(block, block.data / job.speed)
        # update job finish
        job.finish_time = core.finish_time
        # update host finish
        cur_host.finish_time = max(cur_host.finish_time, core.finish_time)
