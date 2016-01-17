from multiprocessing import Pool
from tqdm import *
import time

class ParallelEvaluator(object):
    def __init__(self, num_workers, eval_function, timeout=None, sleep_time=0.1):
        '''
        eval_function should take one argument (a genome object) and return
        a single float (the genome's fitness).
        '''
        self.num_workers = num_workers
        self.eval_function = eval_function
        self.timeout = timeout
        self.pool = Pool(num_workers)
        self.sleep_time = sleep_time

    def evaluate(self, genomes):
        jobs = []
        print("## Dispatching all jobs")
        for genome in genomes:
            jobs.append(self.pool.apply_async(self.eval_function, (genome,)))
        print("## Done dispatching all jobs")
        
        print("## Evaluating Individuals")
        pbar = tqdm(total=len(genomes))
        curr_incomplete = len(genomes)
        start_time = time.time()
        
        while True:
            incomplete_count = sum(1 for x in jobs if not x.ready())
            if incomplete_count == 0:
                pbar.close()
                break
            
            diff_incomplete = curr_incomplete - incomplete_count
            if diff_incomplete > 0:
                pbar.update(diff_incomplete)
            
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
        
        
