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
        self.start_time = np.inf
        self.end_time = np.inf

    def __repr__(self) -> str:
        return f"J({self.jobid}) B({self.blockid}) D({self.data}) S({self.start_time:.1f}) E({self.end_time:.1f})"

class Job:
    def __init__(self, jobid, num_block) -> None:
        self.jobid = jobid
        self.blocks:List[Block] = []
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
        self.blocks:List[Block] = []

    def add_block(self, block:Block, add_finish_time):
        block.start_time = self.finish_time
        self.finish_time += add_finish_time
        block.end_time = self.finish_time

        block.coreid = self.coreid
        block.hostid = self.hostid
        self.blocks.append(block)
    def __repr__(self) -> str:
        return f"H({self.hostid}) C({self.coreid}) F({self.finish_time:.1f})"

class Host:
    def __init__(self, hostid, num_core) -> None:
        self.hostid = hostid
        self.num_core = num_core

    def init(self):
        self.finish_time = 0

        self.cores:List[Core] = [Core(self.hostid, idx) for idx in range(self.num_core)]
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
            self.numJob, self.numHost, self.alpha = list2int(info[:-1]) + [float(info[-1])]
            self.St = None
        else:
            self.numJob, self.numHost, self.alpha, self.St = (
                list2int(info[:-1]) + [float(info[-2])] + [int(info[-1])]
            )

        ###### The number of cores for each host
        self.hosts:List[Host] = []
        info = file_in.readline().strip().split(" ")
        for idx, num_core in enumerate(list2int(info)):
            self.hosts.append(Host(hostid=idx, num_core=num_core))
        assert self.numHost == len(self.hosts)

        ###### The number of blocks for each job
        self.jobs:List[Job] = []
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
    
    def plot(self):
        return plot(self)
    def speedup(self, num_core):
        return 1 - self.alpha(num_core - 1)

    def init_task(self):
        for job in self.jobs:
            job.init()

        for host in self.hosts:
            host.init()

    def outputSolutionFromBlock(self):
        print("Task2 Solution (Block Perspective) of Teaching Assistant:")
        for job in self.jobs:
            print(
                f"Job {job.jobid} obtains {job.used_cores} cores (speed={job.speed})"
                f"and finishes at time {job.finish_time}:"
            )
            for block in job.blocks:
                # TODO: computation???
                speed = block.data / job.speed * 1 #if job.finished == 1 else self.speed(core=len(job.blocks))
                print(f"Block{block.blockid}: H{block.hostid}, C{block.coreid}, R'TODO'(time={speed:.2f}),")
        print("The maximum finish time:", max(job.finish_time for job in self.jobs))
        print("The total response time:", sum(job.finish_time for job in self.jobs))
        print(
            "Utilization rate:",
            sum(job.finish_time for job in self.jobs) / sum(host.finish_time for host in self.hosts),
        )

    def outputSolutionFromCore(self):
        print("Task2 Solution (Core Perspective) of Teaching Assistant:")
        max_host_time = 0
        total_time = 0

        for host in self.hosts:
            max_host_time = max(max_host_time, host.finish_time)
            total_time += host.finish_time
            print(f"Host:{host.hostid} finishes at time {host.finish_time:.2f}:")
            for core in host.cores:
                print(
                    f"Core {core.coreid} has {len(core.blocks)} tasks and finishes at time {core.finish_time:.2f}"
                )
                for block in core.blocks:
                    print(
                        f"\tJob {block.jobid} Block {block.blockid}, runTime {block.start_time:.2f} to {block.end_time:.2f}"
                    )

        print("The maximum finish time:", max(job.finish_time for job in self.jobs))
        print("The total response time:", sum(job.finish_time for job in self.jobs))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--task", default=1, type=int)
    parser.add_argument("--type", default="greedy", type=str)
    parser.add_argument("--case", default="input/task1_case1.txt", type=str)
    args = parser.parse_args()

    if args.case is None:
        file_in = sys.stdin
    else:
        file_in = open(args.case, "r")

    rs = ResourceScheduler(args.task, file_in)

    # from utils import generator
    # generator(rs, args.task)

    def schedule(scheduler):
        # NOTE: block.hostid/coreid
        # NOTE: block.start_time/end_time
        from resource.greedy import greedy_schedule
        from resource.greedy import single_core
        from resource.rand import rand_schedule
        import copy

        best = None
        finish_time = np.inf
        for _type in ["rand_schedule", "single_core","greedy_schedule",]:
            sc = copy.deepcopy(scheduler)
            eval(_type)(sc)
            if max(host.finish_time for host in sc.hosts) < finish_time:
                best = (sc, _type)
                finish_time = max(host.finish_time for host in sc.hosts)

        return best

    best_rs, _type = schedule(rs)
    print(_type)

    # best_rs.outputSolutionFromBlock()
    # best_rs.outputSolutionFromCore()
    plot(best_rs)
