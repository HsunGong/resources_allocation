import random
import numpy as np

from resource.run import ResourceScheduler, Job, Host, Core, Block, list2int

def list_schedule(rs: ResourceScheduler):
    # sort jobs by max of different cores
    host = rs.host[0]
    for num_core in host.
