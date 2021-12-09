from typing import List
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors

COLOR_TABLE = list(reversed(mcolors.XKCD_COLORS.values()))

def plot(scheduler):
    y_shift = [0] * len(scheduler.hosts)
    y_labels = []
    y_labels.extend(
        [f'h_{0}_c_{j}' for j in range(len(scheduler.hosts[0].cores))])
    for i, h in enumerate(scheduler.hosts[:-1]):
        y_shift[i+1] = y_shift[i] + len(h.cores)
        y_labels.extend([f'h_{i}_c_{j}' for j in range(len(h.cores))])
    
    def compute_yaxis(hostid, coreid):
            """Compute the Y axis of the figure"""
            return y_shift[hostid] + coreid

    plt.rcdefaults()
    fig, ax = plt.subplots()

    def plot_job(job):
        job_color = COLOR_TABLE[job.jobid]
        for blk in job.blocks:
            ax.barh(
                y=compute_yaxis(blk.hostid, blk.coreid),
                left=blk.start_time,
                width=blk.end_time - blk.start_time,
                color=job_color,
                edgecolor='k'
            )
    
    for job in scheduler.jobs:
        plot_job(job)
    
    ax.set_yticks(np.arange(len(y_labels)), labels=y_labels)
    ax.invert_yaxis()
    ax.set_xlabel('Time')
    ax.set_title('Visualization of running')
    plt.show()
