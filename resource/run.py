import argparse
import sys, os
from typing import List, NamedTuple, Tuple
import typing
import numpy as np
import random
import copy
from plot import plot


def list2int(l):
    return [int(i) for i in l]


class Block:
    def __init__(self, data: int, host: int, blockid: int, jobid: int, rank:int = 0):
        self.blockid = blockid
        self.jobid = jobid
        self.data = data
        self.host = host
        self.rank = rank

    def init(self):
        self.hostid = None
        self.coreid = None
        self.rank = 0
        self.start_time = np.inf
        self.end_time = np.inf

    def __repr__(self) -> str:
        return f"H[{self.host}] J{self.jobid}-B{self.blockid} D({self.data}) T({self.start_time:.1f}-{self.end_time:.1f}) H({self.hostid})/C({self.coreid})"


class Job:
    def __init__(self, jobid, num_block) -> None:
        self.jobid = jobid
        self.blocks: List[Block] = []
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
    def __repr__(self) -> str:
        return f"Job({self.jobid}) with N-block({self.num_block}) by Speed({self.speed})"

class Core:
    def __init__(self, hostid:int, coreid:int) -> None:
        self.hostid = hostid
        self.coreid = coreid

    def init(self):
        # host->core->finishTime : [num_host, num_core]
        self.finish_time = 0.
        # Core perspective: host->core->task-> <job,block,startTime,endTime>
        # [num_host, num_core, task(job, block)]
        self.blocks: List[Block] = []

    def add_block(self, block: Block, add_finish_time, transmission_time=0):
        """
        block_start_time: transmission_time
        block/core finish_time = add_finish_time + transmission_time
        """
        block.start_time = self.finish_time + transmission_time
        self.finish_time += add_finish_time + transmission_time
        block.end_time = self.finish_time

        block.coreid = self.coreid
        block.hostid = self.hostid
        self.blocks.append(block)

    def __repr__(self) -> str:
        return f"H({self.hostid}) C({self.coreid}) F({self.finish_time})"


class Host:
    def __init__(self, hostid, num_core, prev_core) -> None:
        self.hostid = hostid
        self.num_core = num_core
        self.prev_core = prev_core

    def init(self):
        self.finish_time = 0

        self.cores:List[Core] = [Core(self.hostid, idx + self.prev_core) for idx in range(self.num_core)]
        for core in self.cores:
            core.init()
    def __repr__(self) -> str:
        return f"H({self.hostid}) N({self.num_core})"

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
                list2int(info[:-2]) + [float(info[-2])] + [int(info[-1])]
            )

        ###### The number of cores for each host
        self.hosts: List[Host] = []
        info = file_in.readline().strip().split(" ")
        prev_core = 0
        for idx, num_core in enumerate(list2int(info)):
            self.hosts.append(Host(hostid=idx, num_core=num_core, prev_core=prev_core))
            prev_core += num_core
        
        assert self.numHost == len(self.hosts)

        ###### The number of blocks for each job
        self.jobs: List[Job] = []
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
            # print(cur_job.blocks)
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
        from collections import Counter
        global sum_job_time
        print("Task2 Solution (Block Perspective) of Teaching Assistant:")
        for host in self.hosts:
            for core in host.cores:
                for rank, block in enumerate(core.blocks):
                    block.rank = rank + 1
        for job in self.jobs:
            result = Counter(block.coreid for block in job.blocks)
            job.used_cores = len(result)
            g = 1 - self.alpha * (job.used_cores - 1)
            job.speed = g * job.speed
            #print(job.used_cores)
            print(
                f"Job {job.jobid} obtains {job.used_cores} cores (speed={job.speed})"
                f"and finishes at time {job.finish_time}:")
            for block in job.blocks:
                # TODO: computation???
                time = block.data / job.speed * 1  #if job.finished == 1 else self.speed(core=len(job.blocks))
                sum_job_time += time
                print(
                    f"Block{block.blockid}: H{block.hostid}, C{block.coreid}, R{block.rank}(time={time:.2f}),"
                )
        print("The maximum finish time:",
              max(job.finish_time for job in self.jobs))
        print("The total response time:",
              sum(job.finish_time for job in self.jobs))

    def outputSolutionFromCore(self):
        print("Task2 Solution (Core Perspective) of Teaching Assistant:")
        max_host_time = 0
        total_time = 0
        #sum_job_time = 0
        sum_core_time = 0
        for host in self.hosts:
            host.finish_time = 0
            for core in host.cores:
                host.finish_time = max(core.blocks[-1].end_time, host.finish_time)
            max_host_time = max(max_host_time, host.finish_time)
            total_time += host.finish_time
            print(
                f"Host:{host.hostid} finishes at time {host.finish_time:.2f}:")
            for core in host.cores:
                sum_core_time += core.finish_time
                print(
                    f"Core {core.coreid} has {len(core.blocks)} tasks and finishes at time {core.finish_time:.2f}"
                )
                for block in core.blocks:
                    print(
                        f"\tJob {block.jobid} Block {block.blockid}, runTime {block.start_time:.2f} to {block.end_time:.2f}"
                    )


        print("The maximum finish time:",
              max(job.finish_time for job in self.jobs))
        print("The total response time:",
              sum(job.finish_time for job in self.jobs))
        print(
            "Utilization rate:",
            sum_job_time / sum_core_time
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--task", default=2, type=int)
    parser.add_argument("--case", default="input/task2_case1.txt", type=str)
    args = parser.parse_args()

    if args.case is None:
        file_in = sys.stdin
    else:
        file_in = open(args.case, "r")
        
    global sum_job_time
    sum_job_time = 0

    rs = ResourceScheduler(args.task, file_in)

    def schedule_task1(scheduler):
        # NOTE: block.hostid/coreid
        # NOTE: block.start_time/end_time
        from resource.greedy import greedy_schedule, greedy_schedule_enum_core, greedy2
        from resource.greedy import single_core
        from resource.rand import rand_schedule

        best = None
        finish_time = np.inf
        for _type in [
                # "rand_schedule", 
                "single_core", 
                # "greedy_schedule",
                # "greedy_schedule_enum_core",
        ]:
            # for _type in ["greedy_schedule_enum_core"]:
            sc = copy.deepcopy(scheduler)
            eval(_type)(sc)
            if max(host.finish_time for host in sc.hosts) < finish_time:
                best = (sc, _type)
                finish_time = max(host.finish_time for host in sc.hosts)
            # print(_type)
            # plot(sc)
        numCore = 0
        for h in sc.hosts:
            numCore += len(h.cores)
        print(numCore)
        
        for npm1 in range(1, min(7, numCore)+1):
            sc = copy.deepcopy(scheduler)
            greedy2(sc, pp=npm1+1)
            # print(f'greedy2_{npm1+1}')
            # plot(sc)
            if max(host.finish_time for host in sc.hosts) < finish_time:
                best = (sc, f'greedy2_{npm1+1}')
                finish_time = max(host.finish_time for host in sc.hosts)
                
        return best, finish_time

    def schedule_task2(scheduler):
        # NOTE: block.hostid/coreid
        # NOTE: block.start_time/end_time
        from resource.greedy_task2 import greedy, greedy_trans, single_core

        best = None
        finish_time = np.inf
        for _type in ["greedy_trans"]:
            if _type == "greedy_trans":
                for max_core in range(1, 14):
                    kwargs = {"max_allowed_core": max_core}
                    sc = copy.deepcopy(scheduler)
                    eval(_type)(sc, **kwargs)
                    if max(host.finish_time for host in sc.hosts) < finish_time:
                        best = (sc, f'{_type}_{max_core}')
                        finish_time = max(host.finish_time for host in sc.hosts)
            else:
        # for _type in ["single_core","greedy"]:
                sc = copy.deepcopy(scheduler)
                eval(_type)(sc, **kwargs)
                if max(host.finish_time for host in sc.hosts) < finish_time:
                    best = (sc, _type)
                    finish_time = max(host.finish_time for host in sc.hosts)

        return best, finish_time
    # random.seed(103)
    from utils import generator
    generator(rs, args.task, numJob=50, numBlock=(20,80), numCore=(20,30))
    print(f'Generate random testcase.')

    if args.task == 1:
        (best_rs, _type), finish_time = schedule_task1(rs)
    else:
        (best_rs, _type), finish_time = schedule_task2(rs)

    print(_type, finish_time)
    best_rs.outputSolutionFromBlock()
    best_rs.outputSolutionFromCore()
    print('>' * 10, 'OPT scheduler:', _type, '<' * 10)
    plot(best_rs)

    # python resource/run.py --case input/task1_case1.txt --task 1
    # python resource/run.py --case input/task2_case1.txt --task 2
