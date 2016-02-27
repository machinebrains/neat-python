from multiprocessing import Pool
import multiprocessing as mp

no_progress_lib = 0
try:
    from tqdm import *
except ImportError:
    no_progress_lib = 1
    pass

import time

class ParallelEvaluator(object):
    def __init__(self, eval_function, num_workers=mp.cpu_count(), timeout=None, sleep_time=0.1,progress_bar=False,verbose=0):
        '''
        eval_function should take one argument (a genome object) and return
        a single float (the genome's fitness).
        '''
        self.num_workers = num_workers
        self.eval_function = eval_function
        self.timeout = timeout
        self.pool = Pool(num_workers)
        self.sleep_time = sleep_time
        if no_progress_lib == 0:
            self.progress_bar = progress_bar
        else:
            self.progress_bar = False
        self.verbose = verbose

    def evaluate(self, genomes):
        jobs = []
        if self.verbose != 0:
            print("## Dispatching all jobs")
        for genome in genomes:
            jobs.append(self.pool.apply_async(self.eval_function, (genome,)))
        if self.verbose != 0:
            print("## Done dispatching all jobs")
        
        if self.verbose != 0:
            print("## Evaluating Individuals")
        
        if self.progress_bar:
            pbar = tqdm(total=len(genomes))
            curr_incomplete = len(genomes)
        
        start_time = time.time()
        
        while True:
            incomplete_count = sum(1 for x in jobs if not x.ready())
            if incomplete_count == 0:
                if self.progress_bar:
                    pbar.close()
                break
            
            if self.progress_bar:
                diff_incomplete = curr_incomplete - incomplete_count
                if diff_incomplete > 0:
                    pbar.update(diff_incomplete)
            
            if self.progress_bar:
                curr_incomplete = incomplete_count
                
            if self.timeout != None:
                time_diff = start_time - time.time()
                if time_diff > self.timeout:
                    print("Evaluation time is more than the time allowed.")
                    exit(1)
            
            time.sleep(self.sleep_time)
        
    
        # assign the fitness back to each genome
        for job, genome in zip(jobs, genomes):
            genome.fitness = job.get(timeout=None)
        
        
