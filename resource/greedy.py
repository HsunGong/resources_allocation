import random
from typing import List
import numpy as np
from resource.plot import plot

from resource.run import ResourceScheduler, Job, Host, Core, Block, list2int


def greedy2(rs: ResourceScheduler, pp=5):
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
            job.blocks.sort(key=lambda x: x.data,
                            reverse=True)  # Last Finsh First

        # estimate end time
        end_time = np.zeros((num_jobs, num_core + 1))
        bubbles = np.zeros_like(end_time)
        end_time_coreid = np.zeros((num_jobs, num_block, num_core + 1))
        for i in range(num_jobs):
            for j in range(1, num_core + 1):
                end_time[i][j], bubbles[i][j] = shortest_end_time(
                    rs, i, rs.jobs[i], j, end_time_coreid)

        jobids = [i for i in range(rs.numJob)]
        jobids = sorted(jobids,
                        key=lambda id: sum(block.data for block in rs.jobs[id].
                                           blocks) / rs.jobs[id].speed,
                        reverse=True)
        # jobids = sorted(jobids,
        #                 key=lambda id: np.argmin(end_time[jid][1:]) + 1,
        #                 reverse=False)

        for i, jid in enumerate(jobids):
            job = rs.jobs[jid]
            opt_num_core = np.argmin(end_time[jid][1:]) + 1
            # if opt_num_core < 0.75 * num_core:
            opt_num_core = min((num_core + i % pp) // pp, opt_num_core)
            if i % 2 == 1 and i == num_jobs - 1:
                opt_num_core = np.argmin(end_time[jid][1:]) + 1

            used_cores = sorted(
                cur_host.cores,
                key=lambda core: core.finish_time)[:opt_num_core]

            start_time = used_cores[-1].finish_time
            for core in used_cores:
                core.finish_time = start_time

            speed = job.speed * (1 - rs.alpha * (opt_num_core - 1))
            assert speed >= 0

            # estimate end time
            for block in job.blocks:
                # find ealiest core
                core = min(used_cores, key=lambda core: core.finish_time)
                core.add_block(block, add_finish_time=block.data / speed)

            finish_time_now = max(
                used_cores, key=lambda core: core.finish_time).finish_time
            
            for core in used_cores:
                core.finish_time = finish_time_now

            # update job finish
            job.finish_time = finish_time_now
            # update host finish
            cur_host.finish_time = max(cur_host.finish_time, finish_time_now)


def greedy_schedule_enum_core(rs: ResourceScheduler):
    MAX_SERCH_CORE = 3
    MAX_COMB_JOB = 2
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
            job.blocks.sort(key=lambda x: x.data,
                            reverse=True)  # Last Finsh First

        # estimate end time
        end_time = np.zeros((num_jobs, num_core + 1))
        bubbles = np.zeros_like(end_time)
        end_time_coreid = np.zeros((num_jobs, num_block, num_core + 1))
        for i in range(num_jobs):
            for j in range(1, num_core + 1):
                end_time[i][j], bubbles[i][j] = shortest_end_time(
                    rs, i, rs.jobs[i], j, end_time_coreid)

        def calculate_comb_bubble(jobs: List[Job], jobs_cores: List[int]):
            jobs_end_time = [
                end_time[job.jobid][job_core]
                for job, job_core in zip(jobs, jobs_cores)
            ]
            max_end_time = max(jobs_end_time)
            end_time_bubbles = sum((max_end_time - et) for et in jobs_end_time)
            single_job_bubbles = sum(
                bubbles[job.jobid][job_core]
                for job, job_core in zip(jobs, jobs_cores))
            return end_time_bubbles + single_job_bubbles

        jobids = [i for i in range(rs.numJob)]
        jobids = sorted(jobids,
                        key=lambda id: sum(block.data for block in rs.jobs[id].
                                           blocks) / rs.jobs[id].speed,
                        reverse=True)
        i = 0
        while i < num_jobs:
            assert MAX_COMB_JOB == 2
            num_use = min(MAX_SERCH_CORE, num_core)
            if i < num_jobs - 1:
                i0, i1 = jobids[i], jobids[i + 1]
                comb_jobs = [rs.jobs[i0], rs.jobs[i1]]
                all_idx = range(num_core)

                comb_bubbles = [
                    calculate_comb_bubble(
                        comb_jobs,
                        [len(all_idx[:k + 1]),
                         len(all_idx[k + 1:])]) for k in range(num_use - 1)
                ]
                min_k = np.argmin(comb_bubbles)
                comb_bubble_opt = comb_bubbles[min_k]

                if comb_bubble_opt < bubbles[i0][num_core] + bubbles[i1][
                        num_core]:
                    used_cores = sorted(
                        cur_host.cores,
                        key=lambda core: core.finish_time)[:num_use]

                    jobs_cores = [
                        used_cores[:min_k + 1], used_cores[min_k + 1:]
                    ]

                    # TODO: search for two choices.
                    for job_core in jobs_cores:
                        start_time = max(c.finish_time for c in job_core)
                        for c in job_core:
                            c.finish_time = max(c.finish_time, start_time)

                    for job_core, job, in zip(jobs_cores, comb_jobs):
                        speed = speed = job.speed * (1 - rs.alpha *
                                                     (len(job_core) - 1))
                        for block in job.blocks:
                            core = min(job_core,
                                       key=lambda core: core.finish_time)
                            core.add_block(block,
                                           add_finish_time=block.data / speed)
                        max_finish_time = max(core.finish_time
                                              for core in job_core)
                        # for core in job_core:
                        #     core.finish_time = max_finish_time
                        job.finish_time = max_finish_time

                    max_finish_time = max(
                        job.finish_time
                        for job, job_core in zip(comb_jobs, jobs_cores))
                    # assert abs(max_finish_time - max(core.finish_time for core in used_cores)) < 1e-5, \
                    #     (max_finish_time, max(core.finish_time for core in used_cores))
                    # for core in used_cores:
                    #     core.finish_time = max_finish_time
                    # for job in comb_jobs:
                    #     job.finish_time = max_finish_time
                    cur_host.finish_time = max(cur_host.finish_time,
                                               max_finish_time)

                    i += 2
                else:
                    pass
            else:
                job = rs.jobs[jobids[i]]
                num_use = min(num_core, len(job.blocks))
                speed = job.speed * (1 - rs.alpha * (num_use - 1))

                # used_cores = set()
                used_cores = sorted(
                    cur_host.cores,
                    key=lambda core: core.finish_time)[:num_use]
                # print(used_cores)
                start_time = used_cores[-1].finish_time

                # estimate end time
                for core in used_cores:
                    core.finish_time = start_time
                for block in job.blocks:
                    # find ealiest core
                    core = min(used_cores, key=lambda core: core.finish_time)

                    # finish_time --> core.finish_time + block.data / speed
                    core.add_block(block, add_finish_time=block.data / speed)
                    # print(f"Assign Block {block} to Core {core}")
                    # used_cores.add(core)

                finish_time_now = max(
                    used_cores, key=lambda core: core.finish_time).finish_time
                # for core in used_cores:
                #     core.finish_time = finish_time_now

                # update job finish
                job.finish_time = finish_time_now
                # update host finish
                cur_host.finish_time = max(cur_host.finish_time,
                                           finish_time_now)
                i += 1


def shortest_end_time(rs, job_i, job, num_core, end_time_coreid):
    """
    Return the shortest time when job_i can only use j cores, its bubble time
    and record the index of each block on those cores.
    """
    assert num_core > 0
    schedule_res = multi_time_schedule([blk.data for blk in job.blocks],
                                       num_core)
    core_data_size = [0] * num_core
    for core_idx, blk in zip(schedule_res, job.blocks):
        # Assign end_time_coreid.
        end_time_coreid[job_i][blk.blockid][num_core] = core_idx
        core_data_size[core_idx] += blk.data
    num_core_used = len(set(schedule_res))
    speed = job.speed * (1 - rs.alpha * (num_core_used - 1))
    if speed <= 0:
        return 0xffff, 0xffff
    max_data_size = max(core_data_size)

    # We will use all num_core CPUs even if we cannot use in fact but we will calculate
    # bubbles for it.
    bubble_data_size = sum(
        (max_data_size - s) for s in core_data_size)  # # if s != 0)

    # test_job(job, schedule_res, num_core, speed)

    return max_data_size / speed, bubble_data_size / speed


def greedy_schedule(rs: ResourceScheduler):
    ## Only task 1 is supported
    assert rs.taskID == 1

    taskType = rs.taskID
    if taskType == 1:
        hid = 0
        cur_host = rs.hosts[hid]
        num_core = cur_host.num_core

        for job in rs.jobs:
            job.blocks.sort(key=lambda x: x.data,
                            reverse=True)  # Last Finsh First

        jobids = [i for i in range(rs.numJob)]
        jobids = sorted(jobids,
                        key=lambda id: sum(block.data for block in rs.jobs[id].
                                           blocks) / rs.jobs[id].speed,
                        reverse=True)
        # print(jobids)
        for jid in jobids:
            job = rs.jobs[jid]
            num_use = min(num_core, len(job.blocks))
            speed = job.speed * (1 - rs.alpha * (num_use - 1))

            # used_cores = set()
            used_cores = sorted(cur_host.cores,
                                key=lambda core: core.finish_time)[:num_use]
            # print(used_cores)
            start_time = used_cores[-1].finish_time
            for core in used_cores:
                core.finish_time = start_time

            # estimate end time
            for block in job.blocks:
                # find ealiest core
                core = min(used_cores, key=lambda core: core.finish_time)

                # finish_time --> core.finish_time + block.data / speed
                core.add_block(block, add_finish_time=block.data / speed)
                # print(f"Assign Block {block} to Core {core}")
                # used_cores.add(core)

            finish_time_now = max(
                used_cores, key=lambda core: core.finish_time).finish_time
            for core in used_cores:
                core.finish_time = finish_time_now

            # update job finish
            job.finish_time = finish_time_now
            # update host finish
            cur_host.finish_time = max(cur_host.finish_time, finish_time_now)


def single_core(rs):
    job_time_single_core = [
        sum(blk.data for blk in job.blocks) / job.speed for job in rs.jobs
    ]

    allocated_cores = multi_time_schedule(job_time_single_core,
                                          len(rs.hosts[0].cores))
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
        blk.coreid = core_idx
        print(
            f"Block {blk.blockid}, cid:{core_idx}, st:{blk.start_time}, ed:{blk.end_time}"
        )

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
