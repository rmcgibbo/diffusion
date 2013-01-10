import os, sys
import shutil
pjoin = os.path.join
import numpy as np
import subprocess
from msmbuilder import arglib
from msmbuilder import Trajectory, Project
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as pp


class DiffusionProject(object):
    project_dir = None
    n_trajs = 10
    n_frames = 1000
    timestep = 1e-1
    bounds = np.array([-5,5])
    starting = np.array([0,0])
    diffusion_consts = [1, 10]
    conf_filename = 'null.pdb'
    trajectories = None


    def compute(self):
        "Simulate the trajectories, ans put them in self.trajectories"
        
        # container
        self.trajectories = [None for i in range(self.n_trajs)]

        for i in range(self.n_trajs):
            traj = np.zeros((self.n_frames, 2))
            for d in xrange(2):  # x and y coordinate
                traj[:, d] = self._simulate_traj(self.starting[d], self.diffusion_consts[d])

            # put it in the container
            self.trajectories[i] = traj
            print 'Got traj %d' % i


    def save(self):
        "Save the trajs as a n MSMBuilder project"
        
        traj_dir = pjoin(self.project_dir, 'Trajectories')
        if not os.path.exists(traj_dir):
            os.makedirs(traj_dir)

        t = Trajectory.load_trajectory_file(self.conf_filename)

        traj_paths = []
        for i, xyz in enumerate(self.trajectories):
            t['IndexList'] = None # bug in msmbuilder
            t['XYZList'] = xyz

            traj_paths.append(pjoin(traj_dir, 'trj%d.lh5' % i))
            t.save(traj_paths[-1])

        p = Project({'conf_filename': os.path.abspath(self.conf_filename),
            'traj_lengths': self.n_frames*np.ones(self.n_trajs),
            'traj_paths': [os.path.abspath(e) for e in traj_paths],
            'traj_converted_from': [[] for i in range(self.n_trajs)],
            'traj_errors': [None for i in range(self.n_trajs)],
            }, project_dir=self.project_dir, validate=True)
        p.save(pjoin(self.project_dir,'Project.yaml'))

        # just check again
        p = Project.load_from(pjoin(self.project_dir,'Project.yaml'))
        p._validate()
        assert np.all((p.load_traj(0)['XYZList'] - self.trajectories[0])**2 < 1e-6)

    def _simulate_traj(self, start, diffusion_const):
        "Simulate an individual x or y dimension"

        magic = 12345
        pos = magic*np.ones(self.n_frames, dtype=np.float64)
        pos[0] = start

        i = 1
        while i < self.n_frames:
            r = diffusion_const * self.timestep * np.random.randn()
            if np.min(self.bounds) < pos[i-1] + r < np.max(self.bounds):
                pos[i] = pos[i-1] + r
                i += 1
            else:
                #print 'Rejected, x=', pos[i-1], 'r=', r
                continue

        if magic in pos:
            raise RuntimeError('Something went wrong')
        return pos


def plot_raw_trajectory(i):
    from rmagic import r
    p = Project.load_from('project/Project.yaml')
    traj = p.load_traj(i)['XYZList']
    
    r.push(x=traj[:,0], y=traj[:,1], ts=np.arange(p.traj_lengths[i]))
    r.push(bounds=[-5,5])
    r.eval('''
    library(ggplot2)
    p = ggplot(data=data.frame(x=x, y=y, ts=ts), aes(x=x, y=y, color=ts))
    p = p + geom_path()
    #p = p + geom_point()
    p = p + scale_x_continuous(limits=bounds)
    p = p + scale_y_continuous(limits=bounds)
    p = p + scale_color_continuous(low='black', high='lightblue')
    p = p + ggtitle('One of the trajectories')
    ggsave('plot.png')
    system('open plot.png')
    ''')


def run_metric_learning(project_dir):
    # run the extract command
    subprocess.check_output(('LargeMargin2.py extract xyz -c 1 -f 10 -s 10 '
        '-p {pd}/Project.yaml -o {pd}/triplets.h5').format(pd=project_dir),
        shell=True)

    print subprocess.check_output(('LargeMargin2.py learn diagonal -a 0.1 -t project/triplets.h5 '
        '-m {pd}/metric.npy -M {pd}/metric.pickl').format(pd=project_dir),
        shell=True)

    print np.load(pjoin(project_dir, 'metric.npy'))

    print subprocess.check_output(('Cluster.py -a UData/Assignments.h5 -d UData/Assignments.h5.distances -g UData/Gens.lh5 -p {pd}/Project.yaml custom -i even_metric.pickl hybrid -k 40').format(pd=project_dir), shell=True)

    print subprocess.check_output(('Cluster.py -a Data/Assignments.h5 -d Data/Assignments.h5.distances -g Data/Gens.lh5 -p {pd}/Project.yaml custom -i {pd}/metric.pickl hybrid -k 40').format(pd=project_dir), shell=True)
    
    print subprocess.check_output('ipython delunay.ipy -- Data UData', shell=True)

def main():
    d = DiffusionProject()
    if os.path.exists('project'):
        shutil.rmtree('project')

    if os.path.exists('Data'):
        shutil.rmtree('Data')
    if os.path.exists('UData'):
        shutil.rmtree('UData')
    
    d.project_dir = 'project'
    d.compute()
    d.save()
    
    run_metric_learning(d.project_dir)

if __name__ == '__main__':
    main()
    #plot_raw_trajectory(3)
