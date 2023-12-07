import numpy as np
import torch
from torch import nn
import ipdb
from cvxopt import matrix
import cvxopt


import diffuser.utils as utils
from .helpers import (
    cosine_beta_schedule,
    extract,
    apply_conditioning,
    Losses,
)

class GaussianDiffusion(nn.Module):
    def __init__(self, model, horizon, observation_dim, action_dim, n_timesteps=1000,
        loss_type='l1', clip_denoised=False, predict_epsilon=True,
        action_weight=1.0, loss_discount=1.0, loss_weights=None,
    ):
        super().__init__()
        self.horizon = horizon
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.transition_dim = observation_dim + action_dim
        self.model = model

        betas = cosine_beta_schedule(n_timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.n_timesteps = int(n_timesteps)
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)

        ## log calculation clipped because the posterior variance
        ## is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped',
            torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer('posterior_mean_coef1',
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))

        ## get loss coefficients and initialize objective
        loss_weights = self.get_loss_weights(action_weight, loss_discount, loss_weights)
        self.loss_fn = Losses[loss_type](loss_weights, self.action_dim)

    def get_loss_weights(self, action_weight, discount, weights_dict):
        '''
            sets loss coefficients for trajectory

            action_weight   : float
                coefficient on first action loss
            discount   : float
                multiplies t^th timestep of trajectory loss by discount**t
            weights_dict    : dict
                { i: c } multiplies dimension i of observation loss by c
        '''
        self.action_weight = action_weight

        dim_weights = torch.ones(self.transition_dim, dtype=torch.float32)

        ## set loss coefficients for dimensions of observation
        if weights_dict is None: weights_dict = {}
        for ind, w in weights_dict.items():
            dim_weights[self.action_dim + ind] *= w

        ## decay loss with trajectory timestep: discount**t
        discounts = discount ** torch.arange(self.horizon, dtype=torch.float)
        discounts = discounts / discounts.mean()
        loss_weights = torch.einsum('h,t->ht', discounts, dim_weights)

        ## manually set a0 weight
        loss_weights[0, :self.action_dim] = action_weight
        return loss_weights

    #------------------------------------------ sampling ------------------------------------------#

    def predict_start_from_noise(self, x_t, t, noise):
        '''
            if self.predict_epsilon, model output is (scaled) noise;
            otherwise, model predicts x0 directly
        '''
        if self.predict_epsilon:
            return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        else:
            return noise

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, cond, t):
        x_recon = self.predict_start_from_noise(x, t=t, noise=self.model(x, cond, t))

        if self.clip_denoised:
            x_recon.clamp_(-1., 1.)
        else:
            assert RuntimeError()

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
                x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, cond, t):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, cond=cond, t=t)
        noise = torch.randn_like(x)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
    
    @torch.no_grad()
    def p_sample_weight(self, x, cond, t, weight, s):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, cond=cond, t=t)
        noise = torch.randn_like(x)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + s * weight * nonzero_mask * (0.5 * model_log_variance).exp() + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, shape, cond, verbose=True, return_diffusion=False):
        device = self.betas.device

        batch_size = shape[0]
        x = torch.randn(shape, device=device)
        x = apply_conditioning(x, cond, self.action_dim)

        if return_diffusion: diffusion = [x]

        # progress = utils.Progress(self.n_timesteps) if verbose else utils.Silent()
        for i in reversed(range(0, self.n_timesteps)):
            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x = self.p_sample(x, cond, timesteps)
            x = apply_conditioning(x, cond, self.action_dim)

            # progress.update({'t': i})

            if return_diffusion: diffusion.append(x)

        # progress.close()

        if return_diffusion:
            return x, torch.stack(diffusion, dim=1)
        else:
            return x
        
        
    @torch.no_grad()
    def cbf_sample_loop_vec(self, shape, cond, cx, cy, radius, dtau, verbose=True, return_diffusion=False):
        device = self.betas.device

        batch_size = shape[0]
        x = torch.randn(shape, device=device)
        x = apply_conditioning(x, cond, self.action_dim)

        if return_diffusion: diffusion = [x]

        cvxopt.solvers.options['show_progress'] = False
        # cvxopt.solvers.options['abstol'] = 1e-2
        # cvxopt.solvers.options['reltol'] = 1e-2
        # cvxopt.solvers.options['feastol'] = 1e-2


        # progress = utils.Progress(self.n_timesteps) if verbose else utils.Silent()
        for i in reversed(range(0, self.n_timesteps)):
            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x = self.p_sample(x, cond, timesteps)

            # solve QP
            traj = x[:,:,2:4].flatten().detach().cpu().numpy()
            horizon = x.shape[1]

            # time_cur = 1. - self.betas[i]
            # time_cur = 1. - self.alphas_cumprod[i]
            if i!=(self.n_timesteps-1):
                # dtau = (time_past - time_cur).detach().cpu().numpy()
                P = matrix(np.eye(2*horizon))
                # dtau = 1/self.n_timesteps
                f = -1*(traj-traj_past)/dtau
                f = matrix(f.tolist())
                G = np.zeros((horizon, 2*horizon))
                for j in range(horizon):
                    G[j][2*j] = -2 * (traj[2*j]-cx)/radius[0]/radius[0]
                    G[j][2*j+1] = -2 * (traj[2*j+1]-cy)/radius[1]/radius[1]
                    # G[j][2*j] = -2 * (traj_past[2*j]-cx)
                    # G[j][2*j+1] = -2 * (traj_past[2*j+1]-cy)
                G = matrix(G)
                h = ((x[:,:,2]-cx)/radius[0])**2 + ((x[:,:,3]-cy)/radius[1])**2 - 1
                # h = (traj_past[0::2]-cx)**2 + (traj_past[1::2]-cy)**2 - radius**2
                h = h.detach().cpu().numpy()
                h = h**3
                h = matrix(h.tolist())

                sol = cvxopt.solvers.qp(P=P,q=f,G=G,h=h)
                u = np.array(sol["x"])
                # ipdb.set_trace()
                x[:, :, 2:4] = torch.tensor(traj_past).cuda().reshape(1,horizon,2) + torch.tensor(u).cuda().reshape(1,horizon,2) * torch.tensor(dtau).cuda()

            x = apply_conditioning(x, cond, self.action_dim)
            traj_past = x[:,:,2:4].flatten().detach().cpu().numpy()

            # time_past = time_cur

            # progress.update({'t': i})
            # u_past = u.squeeze()

            if return_diffusion: diffusion.append(x)

        # progress.close()

        if return_diffusion:
            return x, torch.stack(diffusion, dim=1)
        else:
            return x
        

    # @torch.no_grad()
    def weight_sample_loop(self, shape, cond, cx, cy, radius, s, verbose=True, return_diffusion=False):
        device = self.betas.device

        batch_size = shape[0]
        x = torch.randn(shape, device=device)
        x = apply_conditioning(x, cond, self.action_dim)

        if return_diffusion: diffusion = [x]


        # progress = utils.Progress(self.n_timesteps) if verbose else utils.Silent()
        for i in reversed(range(0, self.n_timesteps)):
            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)

            horizon = x.shape[1]

            weight = torch.zeros(horizon, 6).cuda()
            # for j in range(horizon):
            #     weight[j][0] = 2 * (x[0,j,2]-cx)/radius[0]
            #     weight[j][1] = 2 * (x[0,j,3]-cy)/radius[1]
            #     weight[j][2] = 2 * x[0,j,0]/radius[0] - 2 * (x[0,j,2]-cx)/radius[0]/radius[0]
            #     weight[j][3] = 2 * x[0,j,1]/radius[1] - 2 * (x[0,j,3]-cy)/radius[1]/radius[1]
            #     weight[j][4] = 2 * (x[0,j,2]-cx)/radius[0]
            #     weight[j][5] = 2 * (x[0,j,3]-cy)/radius[1]
            weight[:,0] = 2 * (x[0,:,2]-cx)/radius[0]/radius[0]
            weight[:,1] = 2 * (x[0,:,3]-cy)/radius[1]/radius[1]
            # weight[:,2] = 2 * x[0,:,4]/radius[0]/radius[0] - 2 * (x[0,:,2]-cx)/radius[0]/radius[0] # p=1
            # weight[:,3] = 2 * x[0,:,5]/radius[1]/radius[1] - 2 * (x[0,:,3]-cy)/radius[1]/radius[1] # p=1
            b = ((x[0,:,2]-cx)/radius[0])**2 + ((x[0,:,3]-cy)/radius[1])**2 - 1
            weight[:,2] = 2 * x[0,:,4]/radius[0]/radius[0] - 2 * (x[0,:,2]-cx)/radius[0]/radius[0] * 3 * (b**2)
            weight[:,3] = 2 * x[0,:,5]/radius[1]/radius[1] - 2 * (x[0,:,3]-cy)/radius[1]/radius[1] * 3 * (b**2)
            weight[:,4] = 2 * (x[0,:,2]-cx)/radius[0]/radius[0]
            weight[:,5] = 2 * (x[0,:,3]-cy)/radius[1]/radius[1]

            db = weight[:,:2]
            u = x[0,:,4:]
            b = ((x[0,:,2]-cx)/radius[0])**2 + ((x[0,:,3]-cy)/radius[1])**2 - 1
            V = torch.diag(torch.matmul(db, u.T)) - b**3
            mask = (V >= 0)
            weight[mask,:] = 0

            weight = weight.unsqueeze(0)

            x = self.p_sample_weight(x, cond, timesteps, weight, s)

            x = apply_conditioning(x, cond, self.action_dim)

            if return_diffusion: diffusion.append(x)

        # progress.close()

        ipdb.set_trace()

        if return_diffusion:
            return x, torch.stack(diffusion, dim=1)
        else:
            return x

    @torch.no_grad()
    def conditional_sample(self, cond, *args, horizon=None, **kwargs):
        '''
            conditions : [ (time, state), ... ]
        '''
        device = self.betas.device
        batch_size = len(cond[0])
        horizon = horizon or self.horizon
        shape = (batch_size, horizon, self.transition_dim)

        return self.p_sample_loop(shape, cond, *args, **kwargs)
    
    # @torch.no_grad()
    # def cbf_sample(self, cond, cx, cy, radius, dtau, *args, horizon=None, **kwargs):
    #     '''
    #         conditions : [ (time, state), ... ]
    #     '''
    #     device = self.betas.device
    #     batch_size = len(cond[0])
    #     horizon = horizon or self.horizon
    #     shape = (batch_size, horizon, self.transition_dim)

    #     return self.cbf_sample_loop(shape, cond, cx, cy, radius, dtau, *args, **kwargs)


    @torch.no_grad()
    def cbf_sample_vec(self, cond, cx, cy, radius, dtau, *args, horizon=None, **kwargs):
        '''
            conditions : [ (time, state), ... ]
        '''
        device = self.betas.device
        batch_size = len(cond[0])
        horizon = horizon or self.horizon
        shape = (batch_size, horizon, self.transition_dim)

        return self.cbf_sample_loop_vec(shape, cond, cx, cy, radius, dtau, *args, **kwargs)
    
    @torch.no_grad()
    def weight_sample(self, cond, cx, cy, radius, *args, horizon=None, **kwargs):
        '''
            conditions : [ (time, state), ... ]
        '''
        device = self.betas.device
        batch_size = len(cond[0])
        horizon = horizon or self.horizon
        shape = (batch_size, horizon, self.transition_dim)

        return self.weight_sample_loop(shape, cond, cx, cy, radius, *args, **kwargs)
    
    #------------------------------------------ training ------------------------------------------#

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sample = (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

        return sample

    def p_losses(self, x_start, cond, t):
        noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_noisy = apply_conditioning(x_noisy, cond, self.action_dim) # just replace start and goal point

        x_recon = self.model(x_noisy, cond, t)
        x_recon = apply_conditioning(x_recon, cond, self.action_dim)

        assert noise.shape == x_recon.shape

        if self.predict_epsilon:
            loss, info = self.loss_fn(x_recon, noise)
        else:
            loss, info = self.loss_fn(x_recon, x_start)

        return loss, info

    def loss(self, x, cond):
        batch_size = len(x)
        ipdb.set_trace()
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()
        return self.p_losses(x, cond, t)

    def forward(self, cond, *args, **kwargs):
        return self.conditional_sample(cond=cond, *args, **kwargs)

