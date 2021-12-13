from typing import List
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
import random
def plot(scheduler):
    # COLOR_TABLE = plt.get_cmap('gist_rainbow')(np.linspace(0, 1.0, scheduler.numJob))
    COLOR_TABLE = list(mcolors.XKCD_COLORS.values())
    # random.shuffle(COLOR_TABLE)

    y_labels = []
    for host in scheduler.hosts:
        y_labels.extend([f'h[{core.hostid}-{core.coreid - host.prev_core}]' for core in host.cores])

    plt.rcdefaults()
    fig, ax = plt.subplots()

    handles = []
    def plot_job(job):
        job_color = COLOR_TABLE[job.jobid]
        for blk in job.blocks:
            ax.barh(y=blk.coreid,# compute_yaxis(blk.hostid, blk.coreid - scheduler.hosts[blk.hostid].prev_core),
                    left=blk.start_time,
                    width=blk.end_time - blk.start_time,
                    color=job_color,
                    label=job.jobid,
                    edgecolor='k')
            from matplotlib.patches import Patch
        handles.append(Patch(facecolor=job_color, label=job.jobid))

    for job in scheduler.jobs:
        plot_job(job)
    for host in scheduler.hosts:
        for core in host.cores:
            for (trans_start, trans_end) in core.transmission:
                ax.barh(y=core.coreid,# compute_yaxis(blk.hostid, blk.coreid - scheduler.hosts[blk.hostid].prev_core),
                        left=trans_start,
                        width=trans_end - trans_start,
                        linestyle="dotted",
                        color="#54545430",
                        label="-1", # label the transmission
                        edgecolor='k')

    ax.set_yticks(np.arange(len(y_labels)), labels=y_labels)
    ax.invert_yaxis()
    ax.set_xlabel('Time')
    ax.set_title('Visualization of running')
    # ax.legend(ncol=scheduler.numJob, bbox_to_anchor=(0, 1), labels=[i for i in range(scheduler.numJob)],
    #           loc='upper left', fontsize='small')
    # plt.legend(ncol=10,handles=handles[:], loc='upper left', fontsize='small', bbox_to_anchor=(0, -0.2))
    plt.tight_layout()
    plt.savefig(f'1.pdf')
    plt.show()
