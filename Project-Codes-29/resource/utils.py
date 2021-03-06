import random
from resource.run import ResourceScheduler, Job, Host, Core, Block, list2int
# random.seed(11)

# def generator(rs: ResourceScheduler, task, numJob=15, numCore=30, numBlock=80):
def generator(rs: ResourceScheduler, task, numJob=3, numCore=(20,30), numBlock=(20,80)):
    rs.numJob = numJob
    rs.alpha = 0.08
    if task == 2:
        rs.numHost = 5
        rs.St = 500
    else:
        rs.numHost = 1
        rs.St = None

    core_range = [i for i in range(*numCore)] # 20-30
    block_range = [i for i in range(*numBlock)] # 20-80
    size_range = [i for i in range(50, 200)]
    speed_range = [i for i in range(20, 80)]
    host_range = [i for i in range(rs.numHost)]

    hosts = []

    prev = 0
    for idx, num_core in enumerate(random.choices(core_range, k=rs.numHost)):
        hosts.append(Host(hostid=idx, num_core=num_core, prev_core=prev))
        prev += num_core

    rs.hosts = hosts

    jobs = []
    for idx, num_block in enumerate(random.choices(block_range, k=rs.numJob)):
        jobs.append(Job(jobid=idx, num_block=num_block))
    rs.jobs = jobs
    for idx, speed in enumerate(random.choices(speed_range, k=rs.numJob)):
        rs.jobs[idx].speed = speed

    job_blocks = []
    for job_idx in range(rs.numJob):
        cur_job = rs.jobs[job_idx]
        cur_job.blocks = []  # deinit all blocks
        blocks = []
        for block_idx, data in enumerate(list2int(random.choices(size_range, k=cur_job.num_block))):
            blocks.append(data)
        job_blocks.append(blocks)

    # Job-> block number-> block location (host number)
    for job_idx in range(rs.numJob):
        cur_job = rs.jobs[job_idx]
        blocks = job_blocks[job_idx]
        for block_idx, host in enumerate(
                list2int(random.choices(host_range, k=cur_job.num_block))):
            cur_job.add_block(data=blocks[block_idx], host=host)

    rs.init_task()
