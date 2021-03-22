import torch
import torch.autograd
import math 

from pytracking.libs import TensorList
from pytracking.utils.plotting import plot_graph
from pytracking.libs.optimization import MinimizationProblem, ConjugateGradientBase

from fsdet.modeling import GeneralizedRCNN

import logging
import time
import copy

from IPython import embed

# TODO self.problem(), how to load data properly to make it compatible with different functions
class DetectionLossProblem(MinimizationProblem):
    """
    Compute losses given the model and data
    """
    def __init__(self, proposals, box_features):
        self.proposals = proposals
        self.box_features = box_features
    def __call__(self, model: GeneralizedRCNN) -> TensorList:
        loss_dict = model.losses_from_features(self.box_features, self.proposals)
        losses = sum(loss_dict.values())
        return TensorList(losses)

class DetectionNewtonCG(ConjugateGradientBase):
    """Newton with Conjugate Gradient. Handels minimization problems in detection."""
    def __init__(self, problem: DetectionLossProblem, model: GeneralizedRCNN, init_hessian_reg = 0.0, hessian_reg_factor = 1.0,
                 cg_eps = 0.0, fletcher_reeves = True, standard_alpha = True, direction_forget_factor = 0,
                 debug = False, analyze = False, plotting = False, fig_num=(10, 11, 12)):
        super().__init__(fletcher_reeves, standard_alpha, direction_forget_factor, debug or analyze or plotting)

        self.problem = problem
        self.model = model
        self.model_eval = copy.deepcopy(model)
        self.x = TensorList([p.detach() for p in self.model.roi_heads.box_predictor.parameters()])
        self.x_eval = TensorList([p.detach() for p in self.model_eval.roi_heads.box_predictor.parameters()])

        self.analyze_convergence = analyze
        self.plotting = plotting
        self.fig_num = fig_num

        self.hessian_reg = init_hessian_reg
        self.hessian_reg_factor = hessian_reg_factor
        self.cg_eps = cg_eps
        self.f0 = None
        self.g = None

        self.residuals = torch.zeros(0)
        self.losses = torch.zeros(0)
        self.gradient_mags = torch.zeros(0)

    def clear_temp(self):
        self.f0 = None
        self.g = None

    # TODO set the num_cg_iter and num_newton_iter
    def run(self, num_cg_iter, num_newton_iter=None):

        if isinstance(num_cg_iter, int):
            if num_cg_iter == 0:
                return
            if num_newton_iter is None:
                num_newton_iter = 1
            num_cg_iter = [num_cg_iter] * num_newton_iter

        num_newton_iter = len(num_cg_iter)
        if num_newton_iter == 0:
            return

        # TODO related to the data loading, before iterations, should it evaluate the whole dataset or only a batch 
        # in analyze_convergence, it seems that the loss is evaluated only once?
        # if self.analyze_convergence:
        #     self.evaluate_CG_iteration(0)

        for cg_iter in num_cg_iter:
            self.run_newton_iter(cg_iter)
            self.hessian_reg *= self.hessian_reg_factor

        # TODO if self.debug, how to load the data properly (eval on whole dataset or on batch?)
        # why here calculate the loss again, duplicate with the last step run_CG
        # if self.debug:
        #     if not self.analyze_convergence:
        #         loss = self.problem(self.model)
        #         self.losses = torch.cat((self.losses, loss.detach().cpu().view(-1)))

        #     if self.plotting:
        #         plot_graph(self.losses, self.fig_num[0], title='Loss')
        #         plot_graph(self.residuals, self.fig_num[1], title='CG residuals')
        #         if self.analyze_convergence:
        #             plot_graph(self.gradient_mags, self.fig_num[2], 'Gradient magnitude')

        # TODO should the trained model be detached (no_grad) and returned?
        # self.x.detach_()

        self.clear_temp()

        return self.losses, self.residuals

    def run_newton_iter(self, num_cg_iter):

        assert self.model.training, "[SimpleTrainer] model was changed to eval mode!"

        # TODO check the autograd using TesnsorList datatype
        self.f0 = self.problem(self.model)
        embed()
        if self.debug and not self.analyze_convergence:
            self.losses = torch.cat((self.losses, self.f0.detach().cpu().view(-1)))
        
        # Gradient of loss
        self.g = TensorList(torch.autograd.grad(self.f0, self.model.roi_heads.box_predictor.parameters(), create_graph=True))
        embed()
        # Get the right hand side
        self.b = - self.g.detach()

        # Run CG
        delta_x, res = self.run_CG(num_cg_iter, eps=self.cg_eps)
        embed()
        # TODO partially done, check again if this works
        self.x.detach_()
        self.x += delta_x
        embed()
        # update or modify the weight of the model given the delta_x

        if self.debug:
            self.residuals = torch.cat((self.residuals, res))
        
        # TODO write metrics as in detectron2
        # self._write_metrics(loss_dict, data_time)

    # TODO check if the calculation is correct, also print the tensor shape in pytracking
    def A(self, x):
        return TensorList(torch.autograd.grad(self.g, self.model.roi_heads.box_predictor.parameters(), x, retain_graph=True)) + self.hessian_reg * x

    def ip(self, a, b):
        # Implements the inner product
        return self.problem.ip_input(a, b)
    
    def M1(self, x):
        return self.problem.M1(x)

    def M2(self, x):
        return self.problem.M2(x)

    def evaluate_CG_iteration(self, delta_x):
        # TODO load the same data as in Newton iteration
        # if self.analyze_convergence:
        #     self.x_eval = (self.x + delta_x).detach()

        #     loss = self.problem(self.model_eval)
        #     grad = TensorList(torch.autograd.grad(loss, self.model_eval.roi_heads.box_predictor.parameters()))

        #     # store in the vectors
        #     self.losses = torch.cat((self.losses, loss.detach().cpu().view(-1)))
        #     self.gradient_mags = torch.cat((self.gradient_mags, sum(grad.view(-1) @ grad.view(-1)).cpu().sqrt().detach().view(-1)))
        pass

    def _write_metrics(
        self,
        loss_dict: Dict[str, torch.Tensor],
        data_time: float,
        prefix: str = "",
        ):
        """
        Args:
            loss_dict (dict): dict of scalar losses
            data_time (float): time taken by the dataloader iteration
        """
        pass




    











