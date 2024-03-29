from detectron2.structures.boxes import Boxes
from detectron2.structures.instances import Instances
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
import os

# TODO self.problem(), how to load data properly to make it compatible with different functions
class DetectionLossProblem(MinimizationProblem):
    """
    Compute losses given the model and data
    """
    def __init__(self, proposals, box_features, regularization:str = None, mask:TensorList = None, base_params:dict = None, reg = None, augmentation:bool = False, bg_class_id = None, super_cat:str=None, pseudo_shots = None, noise_level = None, drop_rate = None):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.augmentation = augmentation
        # TODO this augmentation is now only for HDA, not TFA given the filtering of fg abd bg
        if self.augmentation:
            if super_cat == None or super_cat == "bg":
                self.fg_proposals, self.fg_box_features, self.bg_proposals, self.bg_box_features = self.fg_proposals_filter(proposals, box_features, bg_class_id=bg_class_id)
                del proposals, box_features
                torch.cuda.empty_cache()
            elif super_cat == 'animal' or super_cat == 'food':
                self.proposals, self.box_features = proposals, box_features
        else:
            self.proposals, self.box_features = proposals, box_features
            
        self.loss_dict = None
        self.regularization = regularization
        # self.regularization = False
        self.mask = mask
        self.base_params = base_params
        self.reg = reg
        # NOTE the param names are for box predictor in TFA (cos), not RTS (fc)
        # name of layers that were tuned, for CosineOutputLayers
        self.param_names = ['roi_heads.box_predictor.cls_score.weight', 
                            'roi_heads.box_predictor.bbox_pred.weight',
                            'roi_heads.box_predictor.bbox_pred.bias']
        self.super_cat = super_cat
        self.pseudo_shots = pseudo_shots
        self.noise_level = noise_level
        self.drop_rate = drop_rate

    def __call__(self, model: GeneralizedRCNN, weights: TensorList = None) -> TensorList:
        if self.augmentation:
            if self.super_cat == None or self.super_cat == "bg":
                fg_proposals, fg_box_features = self.feature_augmentation(self.fg_proposals, self.fg_box_features, pseudo_shots=self.pseudo_shots, noise_level=self.noise_level, drop_rate=self.drop_rate)
                proposals = [*self.bg_proposals, *fg_proposals]
                box_features = torch.cat((self.bg_box_features, fg_box_features), dim=0)
            elif self.super_cat == 'animal' or self.super_cat == 'food':
              proposals, box_features = self.feature_augmentation(self.proposals, self.box_features, pseudo_shots=self.pseudo_shots, noise_level=self.noise_level, drop_rate=self.drop_rate)
        else:
            proposals, box_features = self.proposals, self.box_features

        self.loss_dict = model.losses_from_features(box_features, proposals, weights, super_cat = self.super_cat)

        if self.regularization is not None:
            assert self.regularization in ['scalar', 'feature wise'], "Type of regularization is incorrect!"
            params = [p for p in model.roi_heads.box_predictor.parameters()] if weights is None else weights
            self.loss_dict.update({'loss_weight': self.regularization_loss(params)}) 

        losses = sum(self.loss_dict.values())
        return TensorList([losses])
    
    # NOTE this regularization_loss is used for constraining the base weights
    def regularization_loss(self, params: list):
        losses = 0.0

        if self.regularization == 'scalar':
            for idx, param_name in enumerate(self.param_names):
                base_weights_updated = params[idx][(self.mask[idx]==0).nonzero(as_tuple = True)]
                if len(params[idx].shape) == 2:
                    base_weights_updated = base_weights_updated.view(-1, params[idx].shape[1])
                losses = losses + torch.sum(torch.square(base_weights_updated - self.base_params[param_name]))
                # losses = losses + torch.sum(torch.square(self.reg*(base_weights_updated - self.base_params[param_name])))
            losses = self.reg*losses

        elif self.regularization == 'feature wise':
            for idx, param_name in enumerate(self.param_names):
                base_weights_updated = params[idx][(self.mask[idx]==0).nonzero(as_tuple = True)]
                if len(params[idx].shape) == 2:
                    base_weights_updated = base_weights_updated.view(-1, params[idx].shape[1])
                losses = losses + torch.sum(torch.square(self.reg[idx]*(base_weights_updated - self.base_params[param_name])))

        return losses

    def feature_augmentation(self, init_proposals, init_box_features, 
                             pseudo_shots=5, noise_level=0.1, drop_rate = 0.5):
        proposals = init_proposals*pseudo_shots
        box_features = init_box_features.repeat(pseudo_shots, 1)

        noise = torch.empty(box_features.size()).to(self.device)
        torch.nn.init.normal_(noise, mean=0, std=noise_level)

        box_features += noise
        torch.abs_(box_features)

        drop_out = torch.nn.Dropout(p=drop_rate, inplace=True)
        drop_out(box_features)
        return proposals, box_features

    def fg_proposals_filter(self, proposals, box_features, bg_class_id):
        # NOTE select the foreground proposals and corresponding box features for feature augmentation
        num_preds_per_image = torch.tensor([len(p) for p in proposals])
        gt_classes_per_image = [p.gt_classes for p in proposals]
        fg_inds = torch.tensor([]).to(self.device)
        fg_proposals = list([])
        bg_inds = torch.tensor([]).to(self.device)
        bg_proposals = list([])
        for i, (proposal, num_preds, gt_classes) in enumerate(zip(proposals, num_preds_per_image, gt_classes_per_image)):
            fg_inds_local = torch.arange(num_preds)[gt_classes!=bg_class_id].to(self.device)
            fg_inds = torch.cat((fg_inds, torch.sum(num_preds_per_image[:i])+fg_inds_local)).long()
            bg_inds_local = torch.arange(num_preds)[gt_classes==bg_class_id].to(self.device)
            bg_inds = torch.cat((bg_inds, torch.sum(num_preds_per_image[:i])+bg_inds_local)).long()

            if len(fg_inds_local)>0:
                fg_proposals.append(self.single_proposal_filter(proposal, fg_inds_local))

            if len(bg_inds_local)>0:
                bg_proposals.append(self.single_proposal_filter(proposal, bg_inds_local))
        
        fg_box_features = box_features[fg_inds]
        bg_box_features = box_features[bg_inds]

        return fg_proposals, fg_box_features, bg_proposals, bg_box_features

    def single_proposal_filter(self, proposal, inds):
        filtered_proposal = Instances(proposal.image_size)
        filtered_proposal.proposal_boxes = Boxes(proposal.proposal_boxes.tensor[inds])
        filtered_proposal.objectness_logits = proposal.objectness_logits[inds]
        filtered_proposal.gt_boxes = Boxes(proposal.gt_boxes.tensor[inds])
        filtered_proposal.gt_classes = proposal.gt_classes[inds]

        return filtered_proposal
        
    def bg_proposal_sampler(self, proposals, box_features, samples_num=10):
        num_preds = box_features.shape[0]
        num_preds_per_image = [len(p) for p in proposals]
        sample_inds, _ = torch.randperm(num_preds)[:samples_num].sort()
        sample_bool = torch.zeros(num_preds, dtype=torch.bool)
        sample_bool[sample_inds] = True
        sample_bool_list = sample_bool.split(num_preds_per_image, dim=0)
        sampled_proposals = list([])
        for i, (proposal, sample_bool_local) in enumerate(zip(proposals, sample_bool_list)):
            if torch.sum(sample_bool_local) == 0:
                continue
            sampled_proposals.append(self.single_proposal_filter(proposal, sample_bool_local))
        sampled_box_features = box_features[sample_inds]
        return sampled_proposals, sampled_box_features

            

class DetectionNewtonCG(ConjugateGradientBase):
    """Newton with Conjugate Gradient. Handels minimization problems in detection."""
    def __init__(self, problem: DetectionLossProblem, model: GeneralizedRCNN,
                 novel_init_weights=None, IDMAP = None, NOVEL_CLASSES = None, super_cat: str = None,
                 init_hessian_reg = 0.0, hessian_reg_factor = 1.0, cg_eps = 0.0, fletcher_reeves = True, standard_alpha = True, direction_forget_factor = 0,
                 debug = False, analyze = False, plotting = False, fig_num=(10, 11, 12)):
        super().__init__(fletcher_reeves, standard_alpha, direction_forget_factor, debug or analyze or plotting)

        self.problem = problem

        if novel_init_weights is not None:
            self.overwrite_novel_weights(model, novel_init_weights, IDMAP, NOVEL_CLASSES)
        
        self.model = model
        # NOTE currently model_eval is not available for hierarchical detection setting, as we are optimizing three modules at same time
        self.model_eval = copy.deepcopy(model) if debug is True else None
        
        if super_cat == 'bg':
            module = self.model.roi_heads.box_predictor_bg
        elif super_cat == 'animal':
            module = self.model.roi_heads.box_predictor_animal
        elif super_cat == 'food':
            module = self.model.roi_heads.box_predictor_food
        else:
            module = self.model.roi_heads.box_predictor
        self.super_cat = super_cat
        self.module = module

        # self.state_dict = self.model.roi_heads.box_predictor.state_dict()
        self.state_dict = self.module.state_dict()
        
        self.layer_names = list(self.state_dict.keys())
        self.x = TensorList([self.state_dict[k].detach().clone() for k in self.state_dict.keys()])
        self.x_eval = TensorList([self.state_dict[k].detach().clone() for k in self.state_dict.keys()]) if debug is True else None
        # self.augmentation = augmentation
        
        self.debug = debug
        self.analyze_convergence = analyze
        self.plotting = plotting
        self.fig_num = fig_num

        # TODO change this to a higher value to constrain it at the start, by giving a higher hessian reg factor, stable and converge quickly
        # TODO compare the regularizer with the meta learned weights
        # TODO rescaling or normalization of the meta weights
        self.hessian_reg = init_hessian_reg
        # TODO hessian decay 0.9
        self.hessian_reg_factor = hessian_reg_factor
        self.cg_eps = cg_eps
        self.f0 = None
        self.g = None

        self.residuals = torch.zeros(0)
        self.losses = torch.zeros(0)
        self.gradient_mags = torch.zeros(0)
    
    # NOTE this function is used to overwrite the params with the generated weights
    def overwrite_novel_weights(self, model, novel_init_weights, IDMAP, NOVEL_CLASSES):
        state_dict = model.roi_heads.box_predictor.state_dict()
        layer_names = list(state_dict.keys())
        x = TensorList([state_dict[k].detach().clone() for k in state_dict.keys()])

        for idx in range(len(x)):
            for i, c in enumerate(NOVEL_CLASSES):
                if idx == 0:
                    x[idx][IDMAP[c]] = novel_init_weights[idx][i]
                else:
                    x[idx][IDMAP[c]*4:(IDMAP[c]+1)*4] = novel_init_weights[idx][i*4:(i+1)*4]
        
        for i in range(len(layer_names)):
                state_dict[layer_names[i]] = x[i].detach()

        model.roi_heads.box_predictor.load_state_dict(state_dict) 

        # NOTE for test on TFA CG init weights to HDA novel layers only
        # state_dict = model.roi_heads.box_predictor.state_dict()
        # layer_names = list(state_dict.keys())
        # x = TensorList([state_dict[k].detach().clone() for k in state_dict.keys()])

        # x[0][:-1] = novel_init_weights[0][:]
        # x[2][:] = novel_init_weights[1][:]
        # x[3][:] = novel_init_weights[2][:] 

        # NOTE for test on TFA CG init weights to novel model only
        # for idx in range(len(x)):
        #     x[idx][:] = novel_init_weights[idx][:]

        # for i in range(len(layer_names)):
        #         state_dict[layer_names[i]] = x[i].detach()

        # model.roi_heads.box_predictor.load_state_dict(state_dict)

        # print('overwrite')
        # embed()
        
 
    def clear_temp(self):
        self.f0 = None
        self.g = None

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

        # in analyze_convergence, it seems that the loss is evaluated only once?
        # if self.analyze_convergence:
        #     self.evaluate_CG_iteration(0)

        for cg_iter in num_cg_iter:
            self.run_newton_iter(cg_iter)
            # TODO NOTE reg 1.0, factor 0.9 for decay, experiment with reg
            self.hessian_reg *= self.hessian_reg_factor

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

        # self.x.detach_()

        self.clear_temp()

        return self.losses, self.residuals

    def before_train(self):
        """Steps before running Newton iterations"""
        if self.analyze_convergence:
            self.evaluate_CG_iteration(0)

    def after_train(self):
        """Steps after running Newton iterations"""
        if self.debug:
            if not self.analyze_convergence:
                loss = self.problem(self.model)
                self.losses = torch.cat((self.losses, loss[0].detach().cpu().view(-1)))

            if self.plotting:
                plot_graph(self.losses, self.fig_num[0], title='Loss')
                plot_graph(self.residuals, self.fig_num[1], title='CG residuals')
                if self.analyze_convergence:
                    plot_graph(self.gradient_mags, self.fig_num[2], 'Gradient magnitude')

        self.x.detach_()
        self.clear_temp()

        return self.losses, self.residuals

    def run_newton_iter(self, num_cg_iter):
        self.x.requires_grad_(True)
        assert self.model.training, "[SimpleTrainer] model was changed to eval mode!"
        
        # Get the loss from the model
        self.f0 = self.problem(self.model)
        loss_dict = self.problem.loss_dict
        if self.debug and not self.analyze_convergence:
            self.losses = torch.cat((self.losses, self.f0[0].detach().cpu().view(-1)))
        
        # Gradient of loss
        # NOTE two stage test
        # self.g = TensorList(torch.autograd.grad(self.f0, self.model.roi_heads.box_predictor.parameters(), create_graph=True))
        self.g = TensorList(torch.autograd.grad(self.f0, self.module.parameters(), create_graph=True))   

        # Get the right hand side
        self.b = - self.g.detach()

        # Run CG
        delta_x, res = self.run_CG(num_cg_iter, eps=self.cg_eps)

        # delta_x = self.grad_clipping(delta_x)
        
        self.x.detach_()
        self.x += delta_x
        
        self.model_update()
        # print('{} loss update: {}'.format(self.super_cat, list(loss_dict.values())[0]), 'step norm: {}'.format([torch.sum(torch.square(delta_x[0])), torch.sum(torch.square(delta_x[1]))]))
        # embed()
        
        if self.debug:
            self.residuals = torch.cat((self.residuals, res))

        return loss_dict
    
    # TODO only for test 
    def grad_clipping(self, delta_x, norm = 1.0):
        norm0 = torch.sqrt(torch.sum(torch.square(delta_x[0])))
        norm1 = torch.sqrt(torch.sum(torch.square(delta_x[1])))
        if norm0 > norm:
            delta_x[0] = (delta_x[0]/norm0)*norm
        if norm1 > norm:
            delta_x[1] = (delta_x[1]/norm0)*norm

        return delta_x

    def A(self, x):
        # NOTE ts test
        # return TensorList(torch.autograd.grad(self.g, self.model.roi_heads.box_predictor.parameters(), x, retain_graph=True)) + self.hessian_reg * x
        return TensorList(torch.autograd.grad(self.g, self.module.parameters(), x, retain_graph=True)) + self.hessian_reg * x

    def ip(self, a, b):
        # Implements the inner product
        return self.problem.ip_input(a, b)
    
    def M1(self, x):
        return self.problem.M1(x)

    def M2(self, x):
        return self.problem.M2(x)

    def evaluate_CG_iteration(self, delta_x):
        if self.analyze_convergence:
            # mask out the gradient for the paramters of base classes
            # self.x_eval = (self.x + delta_x*self.mask).detach()
            self.x_eval = (self.x + delta_x).detach()
            self.model_update(eval=True)
            loss = self.problem(self.model_eval)
            grad = TensorList(torch.autograd.grad(loss, self.model_eval.roi_heads.box_predictor.parameters()))
            
            # store in the vectors
            self.losses = torch.cat((self.losses, loss[0].detach().cpu().view(-1)))
            self.gradient_mags = torch.cat((self.gradient_mags, sum(grad.view(-1) @ grad.view(-1)).cpu().sqrt().detach().view(-1)))

    def model_update(self, eval=False):
        if eval: 
            for i in range(len(self.layer_names)):
                self.state_dict[self.layer_names[i]] = self.x_eval[i].detach()
            self.model_eval.roi_heads.box_predictor.load_state_dict(self.state_dict)
        else:
            for i in range(len(self.layer_names)):
                self.state_dict[self.layer_names[i]] = self.x[i].detach()

            # self.model.roi_heads.box_predictor.load_state_dict(self.state_dict)  
            self.module.load_state_dict(self.state_dict)
            # print('updated') 
            # embed()        

class MetaDetectionNewtonCG(ConjugateGradientBase):
    """Newton with Conjugate Gradient. Handels minimization problems in detection."""
    def __init__(self, problem: DetectionLossProblem, model: GeneralizedRCNN,
                 novel_init_weights=None, IDMAP = None, NOVEL_CLASSES = None, 
                 init_hessian_reg = 0.0, hessian_reg_factor = 1.0, cg_eps = 0.0, fletcher_reeves = True, standard_alpha = True, direction_forget_factor = 0,
                 debug = False, analyze = False, plotting = False, fig_num=(10, 11, 12)):
        super().__init__(fletcher_reeves, standard_alpha, direction_forget_factor, debug or analyze or plotting)

        self.problem = problem
        self.model = model
        self.state_dict = self.model.roi_heads.box_predictor.state_dict()
        self.layer_names = list(self.state_dict.keys())
        self.x = TensorList([self.state_dict[k].detach().clone() for k in self.state_dict.keys()])
        # self.x.requires_grad_(True)
    
        if novel_init_weights is not None:
            self.overwrite_novel_weights(novel_init_weights, IDMAP, NOVEL_CLASSES)

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
    
    # TODO rts modify the function to overwrite the weights
    def overwrite_novel_weights(self, novel_init_weights, IDMAP, NOVEL_CLASSES):
        # NOTE tfa
        # for idx in range(len(self.x)):
        #     for i, c in enumerate(NOVEL_CLASSES):
        #         if idx == 0:
        #             self.x[idx][IDMAP[c]] = novel_init_weights[idx][i]
        #         else:
        #             self.x[idx][IDMAP[c]*4:(IDMAP[c]+1)*4] = novel_init_weights[idx][i*4:(i+1)*4]
        # NOTE rts novel
        for idx in range(len(self.x)):
            self.x[idx][:] = novel_init_weights[idx][:]
        # print('overwrite')
        # embed()

    def clear_temp(self):
        self.f0 = None
        self.g = None

    def before_train(self):
        """Steps before running Newton iterations"""
        if self.analyze_convergence:
            self.evaluate_CG_iteration(0)

    def after_train(self):
        """Steps after running Newton iterations"""
        if self.debug:
            if not self.analyze_convergence:
                loss = self.problem(self.model)
                self.losses = torch.cat((self.losses, loss[0].detach().cpu().view(-1)))

            if self.plotting:
                plot_graph(self.losses, self.fig_num[0], title='Loss')
                plot_graph(self.residuals, self.fig_num[1], title='CG residuals')
                if self.analyze_convergence:
                    plot_graph(self.gradient_mags, self.fig_num[2], 'Gradient magnitude')

        self.x.detach_()
        self.clear_temp()

        return self.losses, self.residuals

    def run_newton_iter(self, num_cg_iter):
        self.x.requires_grad_(True)
        assert self.model.training, "[SimpleTrainer] model was changed to eval mode!"
        # self.f0 = self.problem(self.model)
        self.f0 = self.problem(self.model, self.x)

        loss_dict = self.problem.loss_dict
        # Gradient of loss
        # self.g = TensorList(torch.autograd.grad(self.f0, self.model.roi_heads.box_predictor.parameters(), create_graph=True))
        self.g = TensorList(torch.autograd.grad(self.f0, self.x, create_graph=True))

        # Get the right hand side
        # self.b = - self.g.detach()
        self.b = - self.g
        
        # Run CG
        delta_x, res = self.run_CG(num_cg_iter, eps=self.cg_eps)
        # self.x.detach_()
        # self.x += delta_x
        self.x = self.x + delta_x
        # self.model_update()

        print('loss update: {}'.format(loss_dict['loss_cls'].item()), 'step norm: {}'.format([torch.sum(torch.square(delta_x[0])).item(), torch.sum(torch.square(delta_x[1])).item()]))
        # embed()

        return loss_dict

    def A(self, x):
        # return TensorList(torch.autograd.grad(self.g, self.model.roi_heads.box_predictor.parameters(), x, retain_graph=True, create_graph=True)) + self.hessian_reg * x
        return TensorList(torch.autograd.grad(self.g, self.x, x, retain_graph=True, create_graph=True)) + self.hessian_reg * x

    def ip(self, a, b):
        # Implements the inner product
        return self.problem.ip_input(a, b)
    
    def M1(self, x):
        return self.problem.M1(x)

    def M2(self, x):
        return self.problem.M2(x)

    # TODO find a solution to make the parameter refers to self.x (not a copy of value)
    # when the value of state_dict get changed, the model does not change
    # the value of the model only get change when the state_dict is loaded, which
    # only pass the value, but the grad_fn of model remains the original one (None)
    def model_update(self):
        for i in range(len(self.layer_names)):
            [attr_name, weight_name] = self.layer_names[i].split('.')
            getattr(getattr(self.model.roi_heads.box_predictor, attr_name), weight_name)[:] = self.x[i]
                    


        



    











