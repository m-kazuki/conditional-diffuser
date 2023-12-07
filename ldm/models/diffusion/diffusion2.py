import numpy as np
import torch
from torch import nn
import ipdb
from cvxopt import matrix
import cvxopt

def cosine_beta_schedule(timesteps, s=0.008, dtype=torch.float32):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas_clipped = np.clip(betas, a_min=0, a_max=0.999)
    return torch.tensor(betas_clipped, dtype=dtype)

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

class WeightedLoss(nn.Module):

    def __init__(self, weights):
        super().__init__()
        self.register_buffer('weights', weights)

    def forward(self, pred, targ):
        '''
            pred, targ : tensor
                [ batch_size x horizon x transition_dim ]
        '''
        loss = self._loss(pred, targ)
        weighted_loss = (loss * self.weights).mean()
        return weighted_loss
    
class WeightedL1(WeightedLoss):

    def _loss(self, pred, targ):
        return torch.abs(pred - targ)
        

class GaussianDiffusion(nn.Module):
    def __init__(self, model, horizon, n_timesteps=1000,
        loss_type='l1', clip_denoised=False, predict_epsilon=False,
        action_weight=1.0, loss_discount=1.0, loss_weights=None,
    ):
        super().__init__()
        self.horizon = horizon
        self.transition_dim = 4
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
        loss_weights = self.get_loss_weights(loss_discount, loss_weights).transpose(0,1)
        self.loss_fn = WeightedL1(loss_weights)

    def get_loss_weights(self, discount, weights_dict):
        '''
            sets loss coefficients for trajectory

            action_weight   : float
                coefficient on first action loss
            discount   : float
                multiplies t^th timestep of trajectory loss by discount**t
            weights_dict    : dict
                { i: c } multiplies dimension i of observation loss by c
        '''

        dim_weights = torch.ones(4, dtype=torch.float32)

        ## set loss coefficients for dimensions of observation
        if weights_dict is None: weights_dict = {}
        for ind, w in weights_dict.items():
            dim_weights[self.action_dim + ind] *= w

        ## decay loss with trajectory timestep: discount**t
        discounts = discount ** torch.arange(self.horizon, dtype=torch.float)
        discounts = discounts / discounts.mean()
        loss_weights = torch.einsum('h,t->ht', discounts, dim_weights)

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
    def p_sample_weight(self, x, cond, t, dWcbf, s_cbf, dWclf, s_clf, apply_norm=False):
        b, *_, device = *x.shape, x.device
        model_mean, model_variance, model_log_variance = self.p_mean_variance(x=x, cond=cond, t=t)
        noise = torch.randn_like(x)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))

        g_cbf = dWcbf * model_variance
        g_clf = dWclf * model_variance

        g_cbf = g_cbf[:,2:,:]
        g_clf = g_clf[:,2:,:]

        g_cbf = torch.where(torch.abs(g_cbf)<1e-5, 0, g_cbf)
        nonzero_indices = torch.nonzero(g_cbf)
        g_cbf_nonzero = g_cbf[nonzero_indices[:, 0], nonzero_indices[:, 1]]

        g_clf = torch.where(torch.abs(g_clf)<1e-5, 0, g_clf)
        nonzero_indices = torch.nonzero(g_clf)
        g_clf_nonzero = g_clf[nonzero_indices[:, 0], nonzero_indices[:, 1]]

        # if g_cbf_nonzero.numel()==0:
        #     return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

        if apply_norm:
            if g_cbf_nonzero.numel()!=0:
                norm_cbf = model_mean[:,2:,:].norm(2)/g_cbf_nonzero.norm(2)
            else:
                norm_cbf = torch.tensor([0]).cuda()
            
            norm_cbf = torch.clamp(norm_cbf, min=0, max=1e7)


            norm_clf = model_mean[:,2:,:].norm(2)/g_clf_nonzero.norm(2)
            
            norm_clf = torch.clamp(norm_clf, min=0, max=1e7)

            print(norm_clf)

            pad = torch.zeros(1,2,160).cuda()
            g_cbf = torch.cat([pad, g_cbf], axis=1)
            g_clf = torch.cat([pad, g_clf], axis=1)


            return model_mean + s_cbf*norm_cbf*g_cbf + s_clf*norm_clf*g_clf + nonzero_mask * (0.5 * model_log_variance).exp() * noise
        else:
            return model_mean + s_cbf*g_cbf + s_clf*g_clf + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_weight2(self, x, cond, t, dWcbf, s_cbf, dWclf, s_clf, apply_norm=False):
        b, *_, device = *x.shape, x.device
        model_mean, model_variance, model_log_variance = self.p_mean_variance(x=x, cond=cond, t=t)
        noise = torch.randn_like(x)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))

        g_cbf = dWcbf * model_variance
        g_clf = dWclf * model_variance

        g_cbf = torch.where(torch.abs(g_cbf)<1e-5, 0, g_cbf)
        nonzero_indices = torch.nonzero(g_cbf)
        g_cbf_nonzero = g_cbf[nonzero_indices[:, 0], nonzero_indices[:, 1]]

        g_clf = torch.where(torch.abs(g_clf)<1e-5, 0, g_clf)
        nonzero_indices = torch.nonzero(g_clf)
        g_clf_nonzero = g_clf[nonzero_indices[:, 0], nonzero_indices[:, 1]]

        # if g_cbf_nonzero.numel()==0:
        #     return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

        if apply_norm:
            if g_cbf_nonzero.numel()!=0:
                norm_cbf = model_mean.norm(2)/g_cbf_nonzero.norm(2)
            else:
                norm_cbf = torch.tensor([0]).cuda()
            
            norm_cbf = torch.clamp(norm_cbf, min=0, max=1e7)


            norm_clf = model_mean.norm(2)/g_clf_nonzero.norm(2)
            
            norm_clf = torch.clamp(norm_clf, min=0, max=1e7)

            print(norm_clf)


            return model_mean + s_cbf*norm_cbf*g_cbf + s_clf*norm_clf*g_clf + nonzero_mask * (0.5 * model_log_variance).exp() * noise
        else:
            return model_mean + s_cbf*g_cbf + s_clf*g_clf + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, shape, cond, verbose=True, return_diffusion=False):
        device = self.betas.device

        batch_size = shape[0]
        x = torch.randn(shape, device=device)
        x[:,:2,0]=cond[:,:,0].clone()
        x[:,:2,-1]=cond[:,:,1].clone()

        if return_diffusion: diffusion = [x]

        # progress = utils.Progress(self.n_timesteps) if verbose else utils.Silent()
        for i in reversed(range(0, self.n_timesteps)):
            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x = self.p_sample(x, cond, timesteps)
            x[:,:2,0]=cond[:,:,0].clone()
            x[:,:2,-1]=cond[:,:,1].clone()

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
        x[:,:2,0]=cond[:,:,0].clone()
        x[:,:2,-1]=cond[:,:,1].clone()

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
                x[:, :, 2:4] = torch.tensor(traj_past).cuda().reshape(1,horizon,2) + torch.tensor(u).cuda().reshape(1,horizon,2) * torch.tensor(dtau).cuda()

            x[:,:2,0]=cond[:,:,0].clone()
            x[:,:2,-1]=cond[:,:,1].clone()
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
        
    @torch.no_grad()
    def compute_cbf_weight(self, x, neigh, r, p, mode='min', 
                           incl_vel=True, gradpos=False):
        n_neigh = neigh.shape[1]
        horizon = x.shape[-1]
        dx = x[:,0,:] - neigh[:,:,0,:]
        dy = x[:,1,:] - neigh[:,:,1,:]
        dvx = x[:,2,:] - neigh[:,:,2,:]
        dvy = x[:,3,:] - neigh[:,:,3,:]
        vx = x[:,2,:]
        vy = x[:,3,:]
        Vcbf = dx**2 + dy**2 - r**2
        if incl_vel:
            Wcbf = 2*dx*dvx + 2*dy*dvy + Vcbf**p
        else:
            Wcbf = 2*dx*vx + 2*dy*vy + Vcbf**p
        Wcbf[Wcbf.isnan()]=torch.inf
        dWcbf = torch.zeros(1,n_neigh,4,horizon).cuda()

        if gradpos:
            if p==1:
                if incl_vel:
                    dWcbf[:,:,0,:] = 2*dvx + 2*dx
                    dWcbf[:,:,1,:] = 2*dvy + 2*dy
                else:
                    dWcbf[:,:,0,:] = 2*vx + 2*dx
                    dWcbf[:,:,1,:] = 2*vy + 2*dy
            else:
                if incl_vel:
                    dWcbf[:,:,0,:] = p*Vcbf**(p-1)*2*dx + 2*dvx
                    dWcbf[:,:,1,:] = p*Vcbf**(p-1)*2*dy + 2*dvy
                else:
                    dWcbf[:,:,0,:] = p*Vcbf**(p-1)*2*dx + 2*vx
                    dWcbf[:,:,1,:] = p*Vcbf**(p-1)*2*dy + 2*vy
        dWcbf[:,:,2,:] = 2*dx
        dWcbf[:,:,3,:] = 2*dy
        mask = (Wcbf >= 0)
        dWcbf = dWcbf.transpose(2,3)
        dWcbf[mask,:]=0
        dWcbf[dWcbf.isnan()]=0
        dWcbf[dWcbf.isinf()]=0
        dWcbf = dWcbf.transpose(2,3)

        if mode=='min':
            min_idx = torch.argmin(Wcbf, axis=1).squeeze()
            dWcbf = dWcbf[:, min_idx, :, torch.arange(dWcbf.size(-1))]
            dWcbf = dWcbf.transpose(0,1)
            dWcbf = dWcbf.transpose(1,2)
        elif mode=='mean':
            dWcbf = dWcbf.mean(1)
        elif mode=='sum':
            dWcbf = dWcbf.sum(1)

        return Wcbf, dWcbf
    
    @torch.no_grad()
    def apply_cbf(self, x, neigh, r, p, Wcbf):
        cvxopt.solvers.options['show_progress'] = False

        horizon = x.shape[-1]
        dx = x[:,0,:] - neigh[:,:,0,:]
        dy = x[:,1,:] - neigh[:,:,1,:]
        dx = dx.squeeze(0)
        dy = dy.squeeze(0)
        P = matrix(np.eye(2*horizon))
        f = -1*x[:,2:,:].transpose(1,2).flatten()
        f = matrix(f.detach().cpu().numpy().astype(np.double))
        G = np.zeros((horizon, 2*horizon))
        h = np.zeros((horizon))
        min_idx = torch.argmin(Wcbf, axis=1).squeeze()
        for j in range(horizon):
            G[j][2*j] = -2 * dx[min_idx[j], j]
            G[j][2*j+1] = -2 * dy[min_idx[j], j]
            h[j] = (dx[min_idx[j], j]**2 + dy[min_idx[j], j]**2 - r**2)**p
        G = matrix(G)
        h = matrix(h)
        sol = cvxopt.solvers.qp(P=P,q=f,G=G,h=h)
        x[:,2:,:]=torch.tensor(np.array(sol["x"])).reshape((1,horizon,2)).transpose(1,2)

        return x

    @torch.no_grad()
    def compute_clf_weight(self, x, cond, p, gradpos=False):
        horizon = x.shape[-1]
        vx = x[:,2,:]
        vy = x[:,3,:]
        xg = x[:,0,:] - cond[:,0,1]
        yg = x[:,1,:] - cond[:,1,1]
        Vclf = xg**2 + yg**2
        Wclf = 2*xg*vx + 2*yg*vy + Vclf**p
        dWclf = torch.zeros(1,4,horizon).cuda()

        if gradpos:
            if p==1:
                dWclf[:,0,:] = 2*vx + 2*xg
                dWclf[:,1,:] = 2*vy + 2*yg
            else:
                dWclf[:,0,:] = p*Vclf**(p-1)*2*xg + 2*vx
                dWclf[:,1,:] = p*Vclf**(p-1)*2*yg + 2*vy
        dWclf[:,2,:] = 2*xg
        dWclf[:,3,:] = 2*yg

        mask = (Wclf <= 0)
        dWclf = dWclf.transpose(1,2)
        dWclf[mask,:]=0
        dWclf = dWclf.transpose(1,2)

        return dWclf

    @torch.no_grad()
    def weight_sample_loop(self, shape, cond, neigh, r, s_cbf=1, s_clf=1, p=1, mode='min', 
                           apply_norm=False, apply_cbf=False, apply_cbf_mid=False,
                           apply_clf=False, 
                           incl_vel=True, verbose=True, return_diffusion=False):
        # x: [bs, 4, time]
        # neigh: [bs, n_neigh, 2, time]

        device = self.betas.device
        batch_size = shape[0]
        x = torch.randn(shape, device=device)
        x[:,:2,0]=cond[:,:,0].clone()
        x[:,:2,-1]=cond[:,:,1].clone()

        if return_diffusion: diffusion = [x]

        for i in reversed(range(0, self.n_timesteps)):
            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)

            neigh[neigh.isnan()] = torch.inf
            Wcbf, dWcbf = self.compute_cbf_weight(x, neigh, r, p, mode, incl_vel)


            if apply_clf:
                dWclf = self.compute_clf_weight(x, cond, p)

                x = self.p_sample_weight(x, cond, timesteps, dWcbf, s_cbf, dWclf, s_clf, apply_norm=apply_norm)
            else:
                dWclf = torch.zeros_like(dWcbf).cuda()
                x = self.p_sample_weight(x, cond, timesteps, dWcbf, s_cbf, 0, s_clf, apply_norm=apply_norm)

            if (i%20) == 0 and apply_cbf_mid:
                min, _ = Wcbf.min(axis=1)
                min[min>0]=0
                min = min.sum()
                # print(min)
                if min < -1:
                    x = self.apply_cbf(x, neigh, r, p, Wcbf)


            x[:,:2,0]=cond[:,:,0].clone()
            x[:,:2,-1]=cond[:,:,1].clone()

        if apply_cbf:
            x = self.apply_cbf(x, neigh, r, p, Wcbf)

        if return_diffusion:
            return x, torch.stack(diffusion, dim=1)
        else:
            return x
        
    @torch.no_grad()
    def weight_sample_loop2(self, shape, cond, neigh, r, s_cbf=1, s_clf=1, p=1, mode='min', 
                           apply_norm=False, apply_cbf=False, apply_cbf_mid=False,
                           apply_clf=False, 
                           incl_vel=True, verbose=True, return_diffusion=False):
        # x: [bs, 4, time]
        # neigh: [bs, n_neigh, 2, time]

        device = self.betas.device
        batch_size = shape[0]
        x = torch.randn(shape, device=device)
        x[:,:2,0]=cond[:,:,0].clone()
        x[:,:2,-1]=cond[:,:,1].clone()

        if return_diffusion: diffusion = [x]

        for i in reversed(range(0, self.n_timesteps)):
            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)

            neigh[neigh.isnan()] = torch.inf
            Wcbf, dWcbf = self.compute_cbf_weight(x, neigh, r, p, mode, incl_vel, gradpos=True)


            if apply_clf:
                dWclf = self.compute_clf_weight(x, cond, p, gradpos=True)

                x = self.p_sample_weight2(x, cond, timesteps, dWcbf, s_cbf, dWclf, s_clf, apply_norm=apply_norm)
            else:
                dWclf = torch.zeros_like(dWcbf).cuda()
                x = self.p_sample_weight2(x, cond, timesteps, dWcbf, s_cbf, 0, s_clf, apply_norm=apply_norm)

            if (i%20) == 0 and apply_cbf_mid:
                min, _ = Wcbf.min(axis=1)
                min[min>0]=0
                min = min.sum()
                # print(min)
                if min < -1:
                    x = self.apply_cbf(x, neigh, r, p, Wcbf)


            x[:,:2,0]=cond[:,:,0].clone()
            x[:,:2,-1]=cond[:,:,1].clone()

        if apply_cbf:
            x = self.apply_cbf(x, neigh, r, p, Wcbf)

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
        batch_size = cond.shape[0]
        horizon = horizon or self.horizon
        shape = (batch_size, self.transition_dim, horizon)

        return self.p_sample_loop(shape, cond, *args, **kwargs)


    @torch.no_grad()
    def cbf_sample_vec(self, cond, cx, cy, radius, dtau, *args, horizon=None, **kwargs):
        '''
            conditions : [ (time, state), ... ]
        '''
        device = self.betas.device
        batch_size = len(cond[0])
        horizon = horizon or self.horizon
        shape = (batch_size, self.transition_dim, horizon)

        return self.cbf_sample_loop_vec(shape, cond, cx, cy, radius, dtau, *args, **kwargs)
    
    @torch.no_grad()
    def weight_sample(self, cond, neigh, radius, s_cbf=1, s_clf=1, p=1, mode='min', 
                      apply_norm=False, apply_cbf=False, apply_cbf_mid=False,
                      apply_clf=False,
                      incl_vel=True, *args, horizon=None, **kwargs):
        '''
            conditions : [ (time, state), ... ]
        '''
        device = self.betas.device
        batch_size = cond.shape[0]
        horizon = horizon or self.horizon
        shape = (batch_size, self.transition_dim, horizon)

        return self.weight_sample_loop(shape, cond, neigh, radius, s_cbf, s_clf, p, mode, 
                                        apply_norm, apply_cbf, apply_cbf_mid, 
                                        apply_clf,
                                        incl_vel, *args, **kwargs)
    
    @torch.no_grad()
    def energy_sample(self, cond, neigh, radius, s, p, mode='min', apply_cbf=False, *args, horizon=None, **kwargs):
        '''
            conditions : [ (time, state), ... ]
        '''
        device = self.betas.device
        batch_size = cond.shape[0]
        horizon = horizon or self.horizon
        shape = (batch_size, self.transition_dim, horizon)

        return self.energy_sample_loop(shape, cond, neigh, radius, s, p, mode, apply_cbf, *args, **kwargs)
    
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
        x_noisy[:,:2,0]=cond[:,:,0].clone()
        x_noisy[:,:2,-1]=cond[:,:,1].clone()
        x_recon = self.model(x_noisy, cond, t)
        x_recon[:,:2,0]=cond[:,:,0].clone()
        x_recon[:,:2,-1]=cond[:,:,1].clone()

        assert noise.shape == x_recon.shape

        if self.predict_epsilon:
            loss = self.loss_fn(x_recon, noise)
        else:
            loss = self.loss_fn(x_recon, x_start)

        return loss

    def loss(self, x, cond):
        batch_size = len(x)
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()
        return self.p_losses(x, cond, t)

    def forward(self, cond, *args, **kwargs):
        return self.conditional_sample(cond=cond, *args, **kwargs)

