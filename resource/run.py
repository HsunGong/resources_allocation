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
    def __repr__(self) -> str:
        return f"J({self.jobid}) B({self.blockid}) D({self.data})"
    
    def init(self):
        self.hostid = None
        self.coreid = None
        self.start_time = 0
        self.end_time = 0


class Job:
    def __init__(self, jobid, num_block) -> None:
        self.jobid = jobid
        self.blocks:List[Block] = []
        self.num_block = num_block

        self.speed = -1
        self.finished = 0
    def __repr__(self) -> str:
        return f"Job({self.jobid}) Speed:{self.speed} Finished:{self.finished}"

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

    def end_sequential(self):
        return sum(block.data for block in self.blocks) / self.speed

    def end_parallel(self):
        return max(block.data for block in self.blocks) / self.speed


class Core:
    def __init__(self, hostid, coreid) -> None:
        self.hostid = hostid
        self.coreid = coreid
    def __repr__(self) -> str:
        return f"Core({self.coreid})"

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
        self.cores:List[Core] = []

    def __repr__(self) -> str:
        return f"Host({self.hostid})"

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
            self.numJob, self.numHost, self.alpha = list2int(
                info[:-1]) + [float(info[-1])]
            self.St = None
        else:
            self.numJob, self.numHost, self.alpha, self.St = (
                list2int(info[:-1]) + [float(info[-2])] + [int(info[-1])])

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

    def init_task(self):
        for job in self.jobs:
            job.init()

        for host in self.hosts:
            host.init()

    def schedule(self, type="greedy"):
        if type == "rand":
            self.rand_schedule()
        elif type == "greedy":
            self.greedy_schedule()
        else:
            raise ValueError

    def speed(self, core=1):
        # g(e)=1-alpha(e-1)
        assert self.alpha * (core - 1)>= 0 and self.alpha * (core - 1) <= 1
        return 1 - self.alpha * (core - 1)

    def greedy_schedule(self):
        ## Only task 1 is supported
        assert self.taskID == 1
        jobids:List[int] = [i for i in range(self.numJob)]

        # sort by earily finish time
        job_parallel:List[int] = sorted(jobids, key=lambda jobid: self.speed(core=len(self.jobs[jobid].blocks)) * self.jobs[jobid].end_parallel())
        job_parallel:List[Tuple(int,float)] = [(jid, self.speed(core=len(self.jobs[jid].blocks)) * self.jobs[jid].end_parallel()) for jid in job_parallel]
        job_sequential:List[int] = sorted(jobids, key=lambda jobid: self.jobs[jobid].end_sequential())
        job_sequential:List[Tuple(int,float)] = [(i, self.jobs[i].end_sequential()) for i in job_sequential]
        print(job_parallel)
        print(job_sequential)
        
        accum_job = job_parallel + job_sequential
        # greedily allocate parallel job, then sequential
        host:Host = self.hosts[0]
        for jobid, max_time in job_parallel:
            job = self.jobs[jobid]
            if len(job.blocks) <= len(host.cores):
                # earilest_cores = sorted(host.cores, key=lambda core: core.finish_time)[:job.num_block]
                # NOTE: latest core first
                earilest_cores = sorted(host.cores, key=lambda core: core.finish_time, reverse=True)[:job.num_block]
                max_start_time = max(core.finish_time for core in earilest_cores)
                max_finish_time = max_start_time + max_time
                
                job.finish_time = max_finish_time
                for idx, block in enumerate(job.blocks):
                    block.start_time = max_start_time
                    block.end_time = max_finish_time
                    block.hostid = host.hostid
                    block.coreid = earilest_cores[idx].coreid
                    earilest_cores[idx].add_block(block, max_finish_time)
                    print(f"{host}:{earilest_cores[idx]} add block({block}) and finish by {earilest_cores[idx].finish_time}")
                job.finished = 1 # NOTE: Status 1 means is allocated by parallel
            
        host:Host = self.hosts[0]
        for jobid, max_time in job_sequential:
            job = self.jobs[jobid]
            if job.finished == 1: # Already allocated by parallel
                # Find a optimal solution?
                pass
            else:
                earilest_core = sorted(host.cores, key=lambda core: core.finish_time)[0]
                # earilest_cores = sorted(host.cores, key=lambda core: core.finish_time, reverse=True)[:job.num_block]
                
                for idx, block in enumerate(job.blocks):
                    block.start_time = earilest_core.finish_time
                    block.end_time = earilest_core.finish_time + block.data / job.speed
                    block.hostid = host.hostid
                    block.coreid = earilest_core.coreid
                    earilest_core.add_block(block, earilest_core.finish_time + block.data / job.speed)
                    print(f"2 {host}:{earilest_core} add block({block}) and finish by {earilest_core.finish_time}")
                    earilest_core = sorted(host.cores, key=lambda core: core.finish_time)[0]
                job.finish_time = block.end_time # last block end time

                job.finished = 2 # NOTE: Status 1 means is allocated by parallel
        
        for host in self.hosts:
            host.finish_time = max(core.finish_time for core in host.cores)

    def rand_schedule(self):
        # naive
        end_time = np.zeros(100)
        for i in range(len(self.jobs)):
            end_time[i] = 0
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
        for host in self.hosts:
            host.finish_time = max(core.finish_time for core in host.cores)

    def outputSolutionFromBlock(self):
        print("Task2 Solution (Block Perspective) of Teaching Assistant:")
        for job in self.jobs:
            print(f"Job {job.jobid} obtains {job.used_cores} cores (speed={job.speed})"
                f"and finishes at time {job.finish_time}:")
            for block in job.blocks:
                speed = block.data/job.speed * 1 if job.finished == 1 else self.speed(core=len(job.blocks))
                print(f"Block{block.blockid}: H{block.hostid}, C{block.coreid}, R'TODO'(time={speed:.2f}),")
        print("The maximum finish time:", max(job.finish_time for job in self.jobs))
        print("The total response time:", sum(job.finish_time for job in self.jobs))
        print("Utilization rate:", sum(job.finish_time for job in self.jobs) / sum(host.finish_time for host in self.hosts))

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

        print("The maximum finish time:",
              max(job.finish_time for job in self.jobs))
        print("The total response time:",
              sum(job.finish_time for job in self.jobs))

    def debug(self):
        for job in self.jobs:
            print(job)
            for block in job.blocks:
                print(block)
        
        for host in self.hosts:
            for core in host.cores:
                print(core)

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

    from utils import generator
    generator(rs, args.task)
    rs.debug()

    rs.schedule("greedy")
    # rs.schedule("rand")

    rs.outputSolutionFromBlock()
    rs.outputSolutionFromCore()
    plot(rs)
