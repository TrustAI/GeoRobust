import numpy as np
import logging
from abc import ABC, abstractmethod


class Recorder:
    '''
        A recorder that store all intermediate results.
    '''
    def __init__(self,
                 nb_var,
                 max_deep,
                 max_feval,
                 ):
        self.nb_var = nb_var
        # pre-comput trisection results to improve efficiency.
        self.thirds = tuple([(1./3)** i for i in range(1, max_deep + 1)])
        # initlize space for query results
        self.poses = [[] for i in range(max_feval)]
        self.lengths = np.zeros((max_feval, self.nb_var))
        self.levels = np.ones((max_feval, self.nb_var), dtype=int) * -1
        self.fc_vals = np.array([np.inf for i in range(max_feval)])
        self.sizes = np.zeros(max_feval)

        # initlize results
        self.minimum = np.inf
        self.best_idx = None

        # initilize the first centre point 
        self.last_center = -1
        self.po_idxs = [[0]]

        # lipschitz estimite
        self.local_slope = np.zeros(max_feval)

    def new_center(self, pos, length, level, surnd_size):
        '''
            Record several properties for a new point.
        '''
        self.last_center += 1
        self.poses[self.last_center] = pos
        self.lengths[self.last_center] = length
        self.levels[self.last_center] = level
        self.sizes[self.last_center] = surnd_size
    
    def update_func_val(self, f_values):
        '''
            Record query results and update the minimum found so far.
        '''
        nb_fval = len(f_values)
        start_idx = self.last_center + 1 - nb_fval
        self.fc_vals[start_idx : self.last_center + 1] = f_values

        tmp_min_idx = np.argmin(f_values)
        if f_values[tmp_min_idx] < self.minimum:
            self.minimum = f_values[tmp_min_idx].item()
            self.best_idx = start_idx + tmp_min_idx

    def update_func_val_and_slope(self, f_values, parent_idx, dist):
        '''
            Record query results;
            update the minimum found so far;
            Compute and record the local slope; 
            Return the largest one to update the corresponding centres' slope.
        '''
        nb_fval = len(f_values)
        start_idx = self.last_center + 1 - nb_fval
        self.fc_vals[start_idx : self.last_center + 1] = f_values

        tmp_min_idx = np.argmin(f_values)
        if f_values[tmp_min_idx] < self.minimum:
            self.minimum = f_values[tmp_min_idx].item()
            self.best_idx = start_idx + tmp_min_idx

        parent_f_value = self.fc_vals[parent_idx]
        tmp_slope = np.abs(f_values - parent_f_value) / dist
        self.local_slope[start_idx] = tmp_slope[0]
        self.local_slope[start_idx+1] = tmp_slope[1]
        return tmp_slope.max()

    def report(self, logger):
        logger.debug(f"###----------- Recorder Reports -----------###")
        nb_feval = self.last_center + 1
        logger.debug(f'[-] Current pos: {self.poses[:nb_feval]}')
        logger.debug(f'[-] Current slope: {self.local_slope[:nb_feval]}')
        logger.debug(f'[-] Current query results: {[float(item) for item in self.fc_vals[:nb_feval]]}')
        logger.debug(f'[-] Current lengths: {[item.tolist() for item in self.lengths[:nb_feval]]}')
        logger.debug(f'[-] Current levels: {[item.tolist() for item in self.levels[:nb_feval]]}')
        logger.debug(f'[-] Current sizes: {[float(item) for item in self.sizes[:nb_feval]]}')
        logger.debug(f'[-] Potential Optimal Points: {self.po_idxs[-1]}')
        logger.debug('##########----------------------------##########\n')

class DirectBase(ABC):
    '''
        A base class for DIRECT solver.
    '''
    def __init__(self, problem,
                       nb_var,
                       bounds,
                       max_iter,
                       max_deep,
                       max_feval,
                       tolerance,
                       debug = False,
                       **kwargs):

        self.problem = problem
        self.nb_var = nb_var
        self.max_iter = max_iter
        self.max_feval = max_feval + 1
        self.max_deep = max_deep
        self.tolerance = tolerance
        self.debug = debug

        self.lip_factor = 1

        self.logger = logging.getLogger(__name__)
        if self.debug:
            logging.basicConfig(
                format='[%(asctime)s] - %(message)s',
                datefmt='%Y/%m/%d %H:%M:%S',
                level=logging.DEBUG)
        else:
            logging.basicConfig(
                format='[%(asctime)s] - %(message)s',
                datefmt='%Y/%m/%d %H:%M:%S',
                level=logging.INFO)


        self.delta = lambda lst : np.max(lst)/2 # l infinit norm

        self.lb, self.ub = self._set_var_bound(bounds, self.nb_var)
        self.space_length = self.ub - self.lb

        for k,v in kwargs.items():
            exec(f'self.{k.lower()} = {v}')
        
        self.rcd = Recorder(self.nb_var, self.max_deep, self.max_feval)

    def solve(self):
        init_pos = [0.5]*self.nb_var
        init_length = [1.]*self.nb_var
        init_level = [0]*self.nb_var
        self.rcd.new_center(init_pos, init_length, init_level, self.delta(init_length))

        init_fval = self._query_func_val([init_pos])
        self.rcd.update_func_val(init_fval)

        self.nb_iter = 1
        self.runout_query = False
        self.reach_max_deep = False
        while self.nb_iter <= self.max_iter and not self.runout_query and not self.reach_max_deep:
            self._divide_space()
            self.local_low_bound = self.estimate_low_bound()
            self.logger.info(f'[-] {self.nb_iter} th iter: Global minimum: {self.rcd.minimum:.6f} (estimated lower bound: {self.local_low_bound:.6f}). Number of funcation evaluation: {self.rcd.last_center + 1}, found largest slope: {np.max(self.rcd.local_slope)}')
            cur_po_points = self._find_po()
            self.rcd.po_idxs.append(self._check_compute_resource(cur_po_points))
            self.rcd.report(self.logger)
            self.nb_iter += 1

    def get_opt_size(self):
        opt_size = np.sum(self.rcd.sizes[self.rcd.best_idx] * self.space_length)
        return opt_size
    
    def get_largest_po_size(self):
        if self.rcd.po_idxs[-1]:
            largest_size = np.sum(np.max(self.rcd.sizes[self.rcd.po_idxs[-1]]) * self.space_length)
        else:
            largest_size = np.sum(np.max(self.rcd.sizes[self.rcd.po_idxs[-2]]) * self.space_length)
        return largest_size

    def get_largest_slope(self):
        return np.max(self.rcd.local_slope[:self.rcd.last_center])

    def estimate_low_bound(self):
        cur_largest_slope = self.get_largest_slope()
        local_size = self.get_opt_size()
        low_bound = self.rcd.minimum - self.lip_factor * cur_largest_slope * local_size
        return low_bound

    @abstractmethod
    def _divide_space(self):
        pass

    @abstractmethod
    def _find_po(self):
        pass

    @abstractmethod
    def _check_compute_resource(self):
        pass

    def _calc_lbound(self, h, sizes):
        h_size = sizes[h]
        lb = []
        for pp in range(len(h)):
            tmp_rects = h_size < self.rcd.sizes[h[pp]]
            if True in tmp_rects:
                tmp_f = self.rcd.fc_vals[h[tmp_rects]]
                tmp_size = self.rcd.sizes[h[tmp_rects]]
                tmp_lbs = (self.rcd.fc_vals[h[pp]] - tmp_f) / (self.rcd.sizes[h[pp]] - tmp_size)
                lb.append(np.max(tmp_lbs))
            else:
                lb.append(-np.inf)
        return np.array(lb)

    def _calc_ubound(self, h, sizes):
        h_size = sizes[h]
        ub = []
        for pp in range(len(h)):
            tmp_rects = h_size > self.rcd.sizes[[h[pp]]]
            if True in tmp_rects:
                tmp_f = self.rcd.fc_vals[h[tmp_rects]]
                tmp_size = self.rcd.sizes[h[tmp_rects]]
                tmp_ubs = (tmp_f - self.rcd.fc_vals[h[pp]]) / (tmp_size - self.rcd.sizes[h[pp]])
                ub.append(np.min(tmp_ubs))
            else:
                ub.append(np.inf)
        return np.array(ub)

    def _query_func_val(self, centers):
        """
        Sequence query.
            centers : A list of evaluate points.
        """
        points = []
        for c in centers:
            points.append(self._to_actual_point(c, self.lb, self.space_length))
        ans = self.problem(points)
        return ans

    @staticmethod
    def _set_var_bound(bounds, nb_var):
        bounds = np.array(bounds, dtype=float)
        if not (bounds.shape == (nb_var,2)):
            raise AssertionError(
                  f'The shape of bounds should be ({nb_var},2). But got {bounds.shape}')
        lb = bounds[:,0]
        ub = bounds[:,1]
        if True in (lb - ub > 0):
            raise AssertionError('Low bound is larger than upper bound.'+
                  ' The lower bound should be with index 0,' +
                  ' the upper bound should be with index 1.')
        return lb, ub

    def optimal_result(self):
        optimal_idx = self.rcd.best_idx
        optimal_ans = self._to_actual_point(self.rcd.poses[optimal_idx], self.lb, self.space_length) 
        return optimal_ans

    @staticmethod
    def _to_unit_square(point, lb, space_length):
        return (point - lb)/space_length

    @staticmethod
    def _to_actual_point(pos, lb, space_length):
        return space_length * pos + lb

class LowBoundedDIRECT(DirectBase):

    def _check_compute_resource(self, po_points): 
        not_reach_max_deep = np.min(self.rcd.levels[po_points], axis=1) < self.max_deep
        if True in not_reach_max_deep:
            po_points = po_points[not_reach_max_deep]
            new_po_points = []
            available_nb_queries = self.max_feval - self.rcd.last_center - 1

            for pp in po_points:
                nb_divide_dim = np.sum(self.rcd.levels[pp] == np.min(self.rcd.levels[pp]))
                require_nb_queries = nb_divide_dim *2
                if available_nb_queries >= require_nb_queries:
                    new_po_points.append(pp)
                    available_nb_queries -= require_nb_queries
                else: 
                    self.runout_query = True
                    break
            return new_po_points

        else:
            self.reach_max_deep = True
            return None
    
    def _find_po(self):
        cur_sizes = self.rcd.sizes[:self.rcd.last_center+1]
        hull = []
        for sz in set(cur_sizes):
            sz_idx = np.where(cur_sizes == sz)[0]
            sz_min_fval = np.min(self.rcd.fc_vals[:self.rcd.last_center+1][sz_idx])
            po_points = np.where(self.rcd.fc_vals[:self.rcd.last_center+1] <= sz_min_fval)[0]

            for point in po_points:
                if point in sz_idx: hull.append(point)

        hull = np.array(hull)
        lbound = self._calc_lbound(hull, cur_sizes)
        ubound = self._calc_ubound(hull, cur_sizes)
        po_cond1 = lbound - ubound <= 0

        if self.rcd.minimum != 0:
            po_cond2 = (self.rcd.minimum - self.rcd.fc_vals[hull] +
                        self.rcd.sizes[hull] * ubound) / np.abs(self.rcd.minimum) >= self.tolerance
        else:
            po_cond2 = self.rcd.fc_vals[hull] - (self.rcd.sizes[hull] * ubound) <= 0

        po_cond = po_cond1 * po_cond2 

        return hull[po_cond]

    def _divide_space(self):
        for cur_idx in self.rcd.po_idxs[-1]:
            new_points = []
            new_slope = 0
            divide_dims = np.where(self.rcd.levels[cur_idx] == np.min(self.rcd.levels[cur_idx]))[0]
            tmp_third = self.rcd.thirds[self.rcd.levels[cur_idx, divide_dims[0]]]
            for d in divide_dims:

                tmp_left = self.rcd.poses[cur_idx][:]
                tmp_left[d] -= tmp_third
                new_points.append(tmp_left)

                tmp_right = self.rcd.poses[cur_idx][:]
                tmp_right[d] += tmp_third
                new_points.append(tmp_right)

            f_values = self._query_func_val(new_points)
            sorted_idx = np.argsort(f_values)

            divide_idx = sorted_idx//2
            divide_idx = sorted(np.unique(divide_idx), key=divide_idx.tolist().index)
            new_points = [[new_points[i*2],new_points[i*2+1]] for i in range(len(divide_dims))]

            for pair,pair_idx in enumerate(divide_idx):

                new_length = self.rcd.lengths[cur_idx].copy()
                new_length[divide_dims[divide_idx[:pair+1]]] /= 3.
                new_size = self.delta(new_length)
                new_level = self.rcd.levels[cur_idx].copy()
                new_level[divide_dims[divide_idx[:pair+1]]] += 1

                self.rcd.new_center(new_points[pair_idx][0],
                                    new_length,
                                    new_level,
                                    new_size)
                self.rcd.new_center(new_points[pair_idx][1],
                                    new_length,
                                    new_level,
                                    new_size)

                tmp_fvals = [f_values[pair_idx*2], f_values[pair_idx*2 + 1]] 
                tmp_dist = (self.space_length[divide_dims[pair]]) * tmp_third
                tmp_slope = self.rcd.update_func_val_and_slope(tmp_fvals, cur_idx, tmp_dist)
                if tmp_slope > new_slope: new_slope = tmp_slope

            self.rcd.lengths[cur_idx][divide_dims] /= 3.
            self.rcd.levels[cur_idx][divide_dims] += 1
            self.rcd.sizes[cur_idx] = self.delta(self.rcd.lengths[cur_idx])
            self.rcd.local_slope[cur_idx] = new_slope

class LowBoundedDIRECT_POset_full_parrallel(LowBoundedDIRECT):

    def __init__(self, problem,
                       nb_var,
                       bounds,
                       max_iter,
                       max_deep,
                       max_feval,
                       tolerance,
                       po_set_size = 1,
                       debug = False,
                       **kwargs):
        super(LowBoundedDIRECT_POset_full_parrallel,self).__init__(
            problem,
            nb_var,
            bounds,
            max_iter,
            max_deep,
            max_feval,
            tolerance,
            debug,
            **kwargs)
        self.po_size = po_set_size

    def _find_po(self):
        '''
        PO set
        '''
        cur_sizes = self.rcd.sizes[:self.rcd.last_center+1]
        hull = []
        for sz in set(cur_sizes):
            sz_idx = np.where(cur_sizes == sz)[0]
            if len(sz_idx) > self.po_size:
                tmp_k_minimum = np.partition(self.rcd.fc_vals[:self.rcd.last_center+1][sz_idx], self.po_size)[:self.po_size]
                sz_kth_min_fval = np.max(tmp_k_minimum)
            else:
                sz_kth_min_fval = np.max(self.rcd.fc_vals[:self.rcd.last_center+1][sz_idx])
            po_points = np.where(self.rcd.fc_vals[:self.rcd.last_center+1] <= sz_kth_min_fval)[0]

            for point in po_points:
                if point in sz_idx: hull.append(point)

        hull = np.array(hull)
        lbound = self._calc_lbound(hull, cur_sizes)
        ubound = self._calc_ubound(hull, cur_sizes)
        po_cond1 = lbound - ubound <= 0

        if self.rcd.minimum != 0:
            po_cond2 = (self.rcd.minimum - self.rcd.fc_vals[hull] +
                        self.rcd.sizes[hull] * ubound) / np.abs(self.rcd.minimum) >= self.tolerance
        else:
            po_cond2 = self.rcd.fc_vals[hull] - (self.rcd.sizes[hull] * ubound) <= 0

        po_cond = po_cond1 * po_cond2 

        return hull[po_cond]

    def _divide_space(self):
        dims_dict = {}
        all_new_points = []

        for cur_idx in self.rcd.po_idxs[-1]:
            new_slope = 0
            divide_dims = np.where(self.rcd.levels[cur_idx] == np.min(self.rcd.levels[cur_idx]))[0]
            tmp_third = self.rcd.thirds[self.rcd.levels[cur_idx, divide_dims[0]]]
            tmp_mark = len(all_new_points)
            for d in divide_dims:

                tmp_left = self.rcd.poses[cur_idx][:]
                tmp_left[d] -= tmp_third
                all_new_points.append(tmp_left)

                tmp_right = self.rcd.poses[cur_idx][:]
                tmp_right[d] += tmp_third
                all_new_points.append(tmp_right)

            dims_dict[cur_idx] = (divide_dims, tmp_mark, 2*len(divide_dims))

        all_f_values = self._query_func_val(all_new_points)

        for cur_idx in self.rcd.po_idxs[-1]:
            divide_dims, start_mark, num_results = dims_dict[cur_idx]
            f_values = all_f_values[start_mark:start_mark+num_results]
            new_points = all_new_points[start_mark:start_mark+num_results]
            tmp_third = self.rcd.thirds[self.rcd.levels[cur_idx, divide_dims[0]]]

            sorted_idx = np.argsort(f_values)
            divide_idx = sorted_idx//2
            divide_idx = sorted(np.unique(divide_idx), key=divide_idx.tolist().index)
            new_points = [[new_points[i*2],new_points[i*2+1]] for i in range(len(divide_dims))]

            for pair,pair_idx in enumerate(divide_idx):

                new_length = self.rcd.lengths[cur_idx].copy()
                new_length[divide_dims[divide_idx[:pair+1]]] /= 3.
                new_size = self.delta(new_length)
                new_level = self.rcd.levels[cur_idx].copy()
                new_level[divide_dims[divide_idx[:pair+1]]] += 1

                self.rcd.new_center(new_points[pair_idx][0],
                                    new_length,
                                    new_level,
                                    new_size)
                self.rcd.new_center(new_points[pair_idx][1],
                                    new_length,
                                    new_level,
                                    new_size)

                tmp_fvals = [f_values[pair_idx*2], f_values[pair_idx*2 + 1]] 
                tmp_dist = (self.space_length[divide_dims[pair]]) * tmp_third
                tmp_slope = self.rcd.update_func_val_and_slope(tmp_fvals, cur_idx, tmp_dist)
                if tmp_slope > new_slope: new_slope = tmp_slope

            self.rcd.lengths[cur_idx][divide_dims] /= 3.
            self.rcd.levels[cur_idx][divide_dims] += 1
            self.rcd.sizes[cur_idx] = self.delta(self.rcd.lengths[cur_idx])
            self.rcd.local_slope[cur_idx] = new_slope

if __name__ == "__main__":
    from autograd import grad

    def demo_problem(x_ins):
        values = []
        for x_in in x_ins:
            x1,x2 = x_in[0],x_in[1]
            # value = (1 + (x1 + x2 +1)**2 * (19 - 14 * x1 + 3 * x1**2 - 14 * x2 + 6 * x1* x2 + 3 * x2**2)) \
            #         * (30 + (2 * x1 - 3 * x2)**2 * (18 -32 * x1 + 12 * x1**2 + 48 * x2 - 36*x1*x2 + 27 * x2**2))
            value = (1.5 - x1 + x1*x2)**2 + (2.25 - x1 + x1*x2**2)**2 + (2.625 - x1 + x1*x2**3)**2
            values.append(value)
        return np.array(values)

    def func(x1,x2):
        # z = (1 + (x1 + x2 +1)**2 * (19 - 14 * x1 + 3 * x1**2 - 14 * x2 + 6 * x1* x2 + 3 * x2**2)) \
        #             * (30 + (2 * x1 - 3 * x2)**2 * (18 -32 * x1 + 12 * x1**2 + 48 * x2 - 36*x1*x2 + 27 * x2**2))
        z = (1.5 - x1 + x1*x2)**2 + (2.25 - x1 + x1*x2**2)**2 + (2.625 - x1 + x1*x2**3)**2
        return z

    fun_grad_x1 = grad(func,0)
    fun_grad_x2 = grad(func,1)
    max_grad = -np.inf
    max_fv = -np.inf
    min_fv = np.inf
    # for i in np.arange(-2,2.1,0.1):
    #     for j in np.arange(-2, 2.1, 0.1):
    for i in np.arange(-4.5,4.6,0.1):
        for j in np.arange(-4.5, 4.6, 0.1):
            tmp_grad_x1 = abs(fun_grad_x1(i,j))
            tmp_grad_x2 = abs(fun_grad_x2(i,j))
            tmp_grad = max(tmp_grad_x1,tmp_grad_x2)
            tmp_fv = demo_problem([(i,j)])
            if tmp_grad > max_grad:
                max_gidx = (i,j)
                max_grad = tmp_grad
            if tmp_fv.item() < min_fv:
                min_fidx = (i,j)
                min_fv = tmp_fv.item()
            if tmp_fv.item() > max_fv:
                max_fidx = (i,j)
                max_fv = tmp_fv.item()
    print(max_grad,max_gidx, max_fv, max_fidx, min_fv, min_fidx)
    
    b = [[-4.5,4.5],[-4.5,4.5]]
    solver = LowBoundedDIRECT(demo_problem, 2, b, max_deep=20, max_feval=4000, tolerance=1e-1, max_iter=100, debug=False)

    solver.solve()
    print(solver.rcd.best_idx)
    print(solver.optimal_result())
    print(solver.rcd.local_slope)

    solver = LowBoundedDIRECT_POset_full_parrallel(demo_problem, 2, b, max_deep=20, max_feval=4000, tolerance=1e-1, max_iter=100, po_set_size=2, debug=False)

    solver.solve()
    print(solver.rcd.best_idx)
    print(solver.optimal_result())
    # print(solver.rcd.local_slope[solver.rcd.best_idx])