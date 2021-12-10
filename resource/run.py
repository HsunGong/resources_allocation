import argparse
import sys, os
from typing import List, NamedTuple, Tuple
import typing
import numpy as np
import random
from plot import plot


def list2int(l):
    return [int(i) for i in l]


class Block:
    def __init__(self, data: int, host: int, blockid: int, jobid: int):
        self.blockid = blockid
        self.jobid = jobid
        self.data = data
        self.host = host

    def init(self):
        self.hostid = None
        self.coreid = None
        self.start_time = 0
        self.end_time = 0


class Job:
    def __init__(self, jobid, num_block) -> None:
        self.jobid = jobid
        self.blocks = []
        self.num_block = num_block

        self.speed = -1

    def add_block(self, data, host):
        block = Block(data, host, blockid=len(self.blocks), jobid=self.jobid)
        self.blocks.append(block)

    def init(self):
        # The finish time of each job
        self.finish_time = 0
        # The number of cores allocated to each job. init by 0
        self.used_cores = 0

        # Block perspective: job number->block number->(hostID, coreID,rank), rank=1 means that block is the first task running on that core of that host
        for idx in range(self.num_block):
            self.blocks[idx].init()


class Core:
    def __init__(self, hostid, coreid) -> None:
        self.hostid = hostid
        self.coreid = coreid

    def init(self):
        # host->core->finishTime : [num_host, num_core]
        self.finish_time = 0
        # Core perspective: host->core->task-> <job,block,startTime,endTime>
        # [num_host, num_core, task(job, block)]
        self.blocks = []

    def add_block(self, block, add_finish_time):
        self.finish_time += add_finish_time
        self.blocks.append(block)


class Host:
    def __init__(self, hostid, num_core) -> None:
        self.hostid = hostid
        self.num_core = num_core

    def init(self):
        self.finish_time = 0

        self.cores = [Core(self.hostid, idx) for idx in range(self.num_core)]
        for core in self.cores:
            core.init()


class ResourceScheduler:
    def __init__(self, task, file_in) -> None:
        self.taskID = task
        self.caseID = str(file_in)

        # numJob No. 0 ~ numJob-1
        # numHost No. 0 ~ numHost-1
        # alpha g(e)=1-alpha(e-1) alpha>0, e is the number of cores allocated to a single job
        # St, Speed of Transimision
        info = file_in.readline().strip().split(" ")
        if self.taskID == 1:
            self.numJob, self.numHost, self.alpha = list2int(
                info[:-1]) + [float(info[-1])]
            self.St = None
        else:
            self.numJob, self.numHost, self.alpha, self.St = (
                list2int(info[:-1]) + [float(info[-2])] + [int(info[-1])])

        ###### The number of cores for each host
        self.hosts = []
        info = file_in.readline().strip().split(" ")
        for idx, num_core in enumerate(list2int(info)):
            self.hosts.append(Host(hostid=idx, num_core=num_core))
        assert self.numHost == len(self.hosts)

        ###### The number of blocks for each job
        self.jobs = []
        info = file_in.readline().strip().split(" ")
        for idx, num_block in enumerate(list2int(info)):
            self.jobs.append(Job(jobid=idx, num_block=num_block))
        # Speed of calculation for each job
        info = file_in.readline().strip().split(" ")
        for idx, speed in enumerate(list2int(info)):
            self.jobs[idx].speed = speed

        ##### Job.Block init
        # dataSize: Job-> block number-> block size( speed of each block)
        job_blocks = []
        for job_idx in range(self.numJob):
            cur_job = self.jobs[job_idx]
            blocks = []
            info = file_in.readline().strip().split(" ")
            for block_idx, data in enumerate(list2int(info)):
                blocks.append(data)
            job_blocks.append(blocks)
            assert len(blocks) == cur_job.num_block

        # Job-> block number-> block location (host number)
        for job_idx in range(self.numJob):
            cur_job = self.jobs[job_idx]
            blocks = job_blocks[job_idx]
            info = file_in.readline().strip().split(" ")
            for block_idx, host in enumerate(list2int(info)):
                cur_job.add_block(data=blocks[block_idx], host=host)

            assert len(cur_job.blocks) == cur_job.num_block

        self.init_task()

    def init_task(self):
        for job in self.jobs:
            job.init()

        for host in self.hosts:
            host.init()

    def schedule(self, type="rand"):
        self.greedy_schedule()

    def greedy_schedule(self):
        ## Only task 1 is supported
        assert self.taskID == 1

        # sort by finish time
        job_parallel = []
        job_sequential = []

        taskType = self.taskID
        if taskType == 1:
            hid = 0
            cur_host = self.hosts[hid]
            num_jobs = len(self.jobs)
            num_core = cur_host.num_core
            num_block = 0
            for job in self.jobs:
                num_block = max(num_block, len(job.blocks))
            end_time = np.zeros((num_jobs, num_core))
            end_time_coreid = np.zeros((num_jobs, num_block, num_core))
            for job in self.jobs:
                job.blocks = sorted(job.blocks,  key=lambda x:x.data , reverse=True)

            for i in range(num_jobs):
                for j in range(num_core):
                    end_time[i][j] = self.shortest_end_time(self.jobs[i], j, end_time_coreid[i])
            
            i = 0
            for job in self.jobs:
                num_use = min(num_core, len(job.blocks))
                core_use = 0
                j = 0
                speed = job.speed * (1 - self.alpha * (num_use - 1))
                for block in job.blocks:
                    block.hostid = hid
                    block.coreid = end_time_coreid[i][j][num_use] + core_use
                    core = cur_host.cores[block.coreid]
                    block.start_time = core.finish_time
                    block.end_time = core.finish_time + block.data / speed
                    core.add_block(block, block.data/speed)

                    j = j + 1
                for k in num_core:
                    core = cur_host.cores[k]
                    core.finish_time = end_time[i][num_use]
                # update job finish
                job.finish_time = core.finish_time
                # update host finish
                cur_host.finish_time = max(cur_host.finish_time, core.finish_time)

                i = i + 1


    def shortest_end_time(job, cores, end_time_coreid, self):
        max_time = 0
        job = self.jobs[job_i]
        speed = job.speed * (1 - self.alpha * (cores - 1))
        if job.blocks <= cores:
            for i in len(job.blocks):
                max_time = max(max_time, job.blocks[i].data)
                end_time_coreid[i][cores] = i
            ans = max_time / speed
            return ans

        core_time = np.zeros(cores)
        for block in job.blocks:
            short_time = 0xffffff
            cid = 0
            for i in cores:
                if core_time[i] < short_time:
                    short_time = core_time[i]
                    cid = i
            core_time[cid] = core_time[cid] + block.data
            end_time_coreid[block.blockid][cores] = cid
        for i in cores:
            max_time = max(max_time, core_time[i])
        ans = max_time/speed
        return ans

    



    def rand_schedule(self):
        # naive
#        end_time = np.zeros(100)
#        for i in range(len(self.jobs)):
#            end_time[i] = 0
        for job in self.jobs:
            hid = random.randint(0, self.numHost - 1)
            cur_host = self.hosts[hid]
            cid = random.randint(0, cur_host.num_core - 1)
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

    def outputSolutionFromBlock(self):
        print("Task2 Solution (Block Perspective) of Teaching Assistant:")
        print("The maximum finish time:",
              max(job.finish_time for job in self.jobs))
        print("The total response time:",
              sum(job.finish_time for job in self.jobs))

    def outputSolutionFromCore(self):
        print("Task2 Solution (Core Perspective) of Teaching Assistant:")
        max_host_time = 0
        total_time = 0

        for host in self.hosts:
            max_host_time = max(max_host_time, host.finish_time)
            total_time += host.finish_time
            print(f"Host:{host.hostid} finishes at time {host.finish_time}:")
            for core in host.cores:
                print(
                    f"Core {core.coreid} has {len(core.blocks)} tasks and finishes at time {core.finish_time}"
                )
                for block in core.blocks:
                    print(
                        f"\tJob {block.jobid} Block {block.blockid}, runTime {block.start_time} to {block.end_time}"
                    )

        print("The maximum finish time:",
              max(job.finish_time for job in self.jobs))
        print("The total response time:",
              sum(job.finish_time for job in self.jobs))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter, )
    parser.add_argument("--task", default=1, type=int)
    parser.add_argument("--case", default="input/task1_case1.txt", type=str)
    args = parser.parse_args()

    if args.case is None:
        file_in = sys.stdin
    else:
        file_in = open(args.case, "r")

    rs = ResourceScheduler(args.task, file_in)

    # from utils import generator
    # generator(rs, args.task)

    rs.schedule()

    rs.outputSolutionFromBlock()
    rs.outputSolutionFromCore()
    plot(rs)
