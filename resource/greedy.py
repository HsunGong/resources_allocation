import random
import numpy as np

from resource.run import ResourceScheduler, Job, Host, Core, Block, list2int

def shortest_end_time(rs, job, cores, end_time_coreid):
    if cores == 0:
        return 0
    max_time = 0
    speed = job.speed * (1 - rs.alpha * (cores - 1))
    if len(job.blocks) <= cores:
        for i in range(len(job.blocks)):
            max_time = max(max_time, job.blocks[i].data)
            end_time_coreid[i][cores] = i
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
        end_time_coreid[block.blockid][cores] = cid
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
                    end_time[i][j] = rs.shortest_end_time(rs.jobs[i], j, end_time_coreid[i])

            i = 0
            for job in rs.jobs:
                num_use = min(num_core, len(job.blocks))
                core_use = 0
                j = 0
                speed = job.speed * (1 - rs.alpha * (num_use - 1))
                for block in job.blocks:
                    # block.hostid = hid
                    # block.coreid = end_time_coreid[i][j][num_use] + core_use
                    core = cur_host.cores[int(block.coreid)]
                    # block.start_time = core.finish_time
                    # block.end_time = core.finish_time + block.data / speed
                    core.add_block(block, block.data / speed)

                    j = j + 1
                for k in range(num_core):
                    core = cur_host.cores[k]
                    core.finish_time = end_time[i][num_use]
                # update job finish
                job.finish_time = core.finish_time
                # update host finish
                cur_host.finish_time = max(cur_host.finish_time, core.finish_time)

                i = i + 1
                
                
def rand_schedule(self):# need imporvement
        # naive
#       end_time = np.zeros(100)
#       for i in range(len(self.jobs)):
#       end_time[i] = 0
    for job in self.jobs:
        hid = random.randint(0, self.numHost - 1)
        cur_host = self.hosts[hid]
        #cid = random.randint(0, cur_host.num_core - 1)
        short_time = 0xfffff
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
