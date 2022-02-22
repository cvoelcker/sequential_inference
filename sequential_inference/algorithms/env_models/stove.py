"""
TODO: Not ready yet! needs factored object-centric model

@author cvoelcker
@author jkossen
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from sequential_inference.nn_models.vae.image_vae import BroadcastVAE
from sequential_inference.nn_models.dynamics.structured_dynamics import Dynamics
from sequential_inference.abc.sequence_model import AbstractSequenceAlgorithm


class SequenceVAE(AbstractSequenceAlgorithm):
    """STOVE: Structured object-aware video prediction model.

    This class combines a visual encoder and the dynamics model.
    It is a simplification of the STOVE model, which does not use
    any structured latents and is therefore compatible with simple
    visual encoders and decoders. It is
    """

    def __init__(
        self,
        input_channels=3,  # VAE parameters
        latent_dim=32,  # VAE parameters
        enc_layers=[8, 16],  # VAE parameters
        dec_layers=[16, 16, 8],  # VAE parameters
        img_shape=(32, 32),  # VAE parameters
        dyn_layers=[32, 32],  # dynamics parameters
        dyn_core_layers=[32, 32],  # dynamics parameters
        reward_layers=[32, 8],  # dynamics parameters
        action_conditioned=True,  # sequence parameters
    ):
        """Set up model."""
        super().__init__()

        # instantiate with hydra for easy setup
        self.img_model = BroadcastVAE(
            input_channels, latent_dim, enc_layers, dec_layers, img_shape
        )
        self.dyn = Dynamics(latent_dim, dyn_core_layers, dyn_layers, reward_layers)

        self.latent_length = latent_dim

        # Latent prior for unstructured latent at t=0.
        latent_prior_mean = torch.Tensor([0])
        self.register_buffer("latent_prior_mean", latent_prior_mean)
        latent_prior_std = torch.Tensor([0.01])
        self.register_buffer("latent_prior_std", latent_prior_std)
        z_std_prior_mean = torch.Tensor([0.1])
        self.register_buffer("z_std_prior_mean", z_std_prior_mean)
        z_std_prior_std = torch.Tensor([0.01])
        self.register_buffer("z_std_prior_std", z_std_prior_std)
        transition_lik_std = torch.Tensor([1.0])
        self.register_buffer("transition_lik_std", transition_lik_std)

        self.action_conditioned = action_conditioned

    def forward(self, x, actions=None, rewards=None, last_z=None, mask=None):
        """Forward function.

        Can be used to train (action-conditioned) video prediction or
        SuPAIR only without any dynamics model.

        Args:
            x (torch.Tensor), (n, T, o, 3, w, h): Color images..
            step_counter (int): Current training progress.
            actions (torch.Tensor) (n ,T): Actions from env.
            pretrain (bool): Switch for SuPAIR-only training.

        Returns:
            Whatever the respective forwards return.

        """
        batch = x.shape[0]
        T = x.shape[1]
        image_shape = x.shape[2:]

        # 1. Obtain image encoder states.
        z_img, z_img_std = self.encode_img(x, last_z)

        # 2. Dynamics Loop.
        (
            z,
            log_z,
            z_dyn,
            z_dyn_std,
            z_mean,
            z_std,
            rewards_predicted,
        ) = self.infer_dynamics(x, actions, z_img, z_img_std, last_z)
        return (
            z,
            log_z,
            z_img,
            z_img_std,
            z_dyn,
            z_dyn_std,
            z_mean,
            z_std,
            rewards_predicted,
        )

    def encode_img(self, x, last_z=None):
        """Encodes the image sequence to latent reps using the image model
        and saves the gradients

        Args:
            x (torch.Tensor): image observations [batch, sq, obs]
            last_z (torch.Tensor, optional): In case of continuing sequences
            preprend the previous observation embedding for continuity. Defaults to None.

        Returns:
            [torch.Tensor, torch.Tensor]: image mean and std
        """
        T = x.shape[1]
        z_img, z_img_std, _, _ = self.img_model.encode_img(x.flatten(end_dim=1))
        nto_shape = (-1, T, self.latent_length)
        z_img = z_img.view(nto_shape)
        z_img_std = z_img_std.view(nto_shape)
        assert torch.all(z_img_std > 0)
        return z_img, z_img_std

    def infer_dynamics(self, x, actions, z_img_full, z_img_std_full, last_z=None):
        """Infers the dynamics update through the dynamics model and filtering of observation
        and hidden state

        Args:np.mean(log[key].detach().cpu().numpy())
            x ([torch.Tensor]): image sequence
            actions ([torch.Tensor]): action sequence for conditional model
            z_img_full ([torch.Tensor]): latent embedding of images mean
            z_img_std_full ([torch.Tensor]): latent embedding of images std
            last_z ([torch.Tensor], optional): last latent if updating sequence. Defaults to None.

        Returns:
            [type]: [description]
        """
        T = x.shape[1]
        if last_z is None:
            # if starting new sequence, sample latent from prior
            prior_shape = z_img_full[:, 0:1].shape
            latent_prior = Normal(self.latent_prior_mean, self.latent_prior_std)
            latent_prior_sample = latent_prior.rsample(prior_shape)
            init_z = latent_prior_sample.reshape(prior_shape)
        else:
            init_z = last_z

        z = torch.zeros_like(init_z).repeat(1, T, 1)
        z_dyn_s = torch.zeros_like(z)
        z_dyn_s_std = torch.zeros_like(z)
        z_std = torch.zeros_like(z)
        z_mean = torch.zeros_like(z)
        log_z = torch.zeros_like(z)
        rewards = torch.zeros_like(z[:, :, 0]).squeeze(-1)

        # setup actions. To keep indices straight, we preppend a dummy zero
        # action to the beginning of the sequence. This makes the indices slightly
        # less intuitive, but manageable
        if actions is not None:
            core_actions = actions
            core_actions = torch.cat((torch.zeros_like(actions[:, :1]), actions), 1)
        else:
            core_actions = (T + 1) * [None]

        # 2.2 Loop over sequence and do dynamics prediction.
        filter_z = init_z.squeeze()
        for t in range(T):
            # core ignores object sizes
            z_dyn, z_dyn_std, reward = self.dyn(filter_z, core_actions[:, t])
            z_dyn_std = 2 * torch.sigmoid(z_dyn_std) + 0.01

            assert not torch.any(z_dyn_std < 0)

            # obtain full state parameters combining dyn and supair
            # tensors are updated afterwards to prevent inplace assignment issues

            filter_z, filter_log_z, filter_z_mean, filter_z_std = self.full_state(
                z_dyn, z_dyn_std, z_img_full[:, t], z_img_std_full[:, t]
            )

            z[:, t] = filter_z
            log_z[:, t] = filter_log_z
            z_mean[:, t] = filter_z_mean
            z_std[:, t] = filter_z_std
            z_dyn_s[:, t] = z_dyn
            z_dyn_s_std[:, t] = z_dyn_std
            rewards[:, t] = reward.squeeze(-1)

            assert not torch.any(torch.isnan(log_z))

        return z, log_z, z_dyn_s, z_dyn_s_std, z_mean, z_std, rewards

    def dx_from_state(self, z_sup, last_z=None):
        """Get full state by combining supair states.

        The interaction network needs information from at least two previous
        states, since one state encoding does not contain dynamics information

        Args:
            z_sup (torch.Tensor), (n, T, o, img_model.graph_depth): Object
            state means from Image model.

        Returns:
            z_sup_full (torch.Tensor), (n, T, o, 2 * img_model.graph_depth):
            Object state means with pseudo velocities. All zeros at t=0, b/c
            no velocity available.

        """
        # get velocities as differences between positions
        v = z_sup[:, 1:] - z_sup[:, :-1]
        # add zeros to keep time index consistent
        if last_z is None:
            padding = torch.zeros_like(v[:, 0:1])
        else:
            padding = z_sup[:, :1] - last_z

        # keep scales and positions from T
        v_full = torch.cat([padding, v], 1)
        z_sup_full = torch.cat([z_sup, v_full], -1)

        return z_sup_full

    def dx_std_from_pos(self, z_sup_std):
        """Get std on v from std on positions.

        Args:
            z_sup_std (torch.Tensor), (n, T, o, 4): Object state std from SuPAIR.

        Returns:
            z_sup_std_full (torch.Tensor), (n, T, o, 4): Std with added velocity.

        """
        # Sigma of velocities = sqrt(sigma(x1)**2 + sigma(x2)**2)
        v_std = torch.sqrt(z_sup_std[:, 1:] ** 2 + z_sup_std[:, :-1] ** 2)
        # This CAN'T be zeros, this is a std
        zeros = torch.ones_like(z_sup_std[:, 0:1])
        v_std_full = torch.cat([zeros, v_std], 1)
        z_sup_std_full = torch.cat([z_sup_std, v_std_full], -1)

        assert torch.all(z_sup_std_full > 0)

        return z_sup_std_full

    def full_state(self, z_dyn, std_dyn, z_img, std_img):
        """Sample full state from dyn and supair predictions at time t.

        Args:
            z_dyn, std_dyn (torch.Tensor), 2 * (n, o, cl//2): Object state means
                and stds from dynamics core. (pos, velo, latent)
            z_img, std_sup (torch.Tensor), 2 * (n, o, 6): Object state means
                and stds from SuPAIR. (size, pos, velo)
        Returns:
            z_s, mean, std (torch.Tensor), 3 * (n, o, cl//2 + 2): Sample of
                full state, SuPAIR and dynamics information combined, means and
                stds of full state distribution.
            log_q (torch.Tensor), (n, o, cl//2 + 2): Log-likelihood of sampled
                state.

        """
        z_dyn_shared = z_dyn
        std_dyn_shared = std_dyn

        joined_z_mean = (
            torch.pow(std_img, 2) * z_dyn_shared + torch.pow(std_dyn_shared, 2) * z_img
        )
        joined_z_mean = joined_z_mean / (
            torch.pow(std_dyn_shared, 2) + torch.pow(std_img, 2)
        )
        joined_z_std = std_img * std_dyn_shared
        joined_z_std = joined_z_std / torch.sqrt(
            torch.pow(std_dyn_shared, 2) + torch.pow(std_img, 2)
        )

        dist = Normal(joined_z_mean, joined_z_std)

        z_s = dist.rsample()
        assert torch.all(torch.isfinite(z_s))
        log_q = dist.log_prob(z_s)
        assert torch.all(torch.isfinite(log_q))

        return z_s, log_q, joined_z_mean, joined_z_std

    def transition_lik(self, means, results):
        """Get generative likelihood of obtained transition.

        The generative dyn part predicts the mean of the distribution over the
        new state z_dyn = 'means'. At inference time, a final state prediction
        z = 'results' is obtained together with information from SuPAIR.
        The generative likelihood of that state is evaluated with distribution
        p(z_t| z_t-1) given by dyn. As in Becker-Ehms, while inference q and p
        of dynamics core share means they do not share stds.

        Args:
            means, results (torch.Tensor), (n, T, o, cl//2): Latent object
                states predicted by generative dynamics core and inferred from
                dynamics model and SuPAIR jointly. States contain
                (pos, velo, latent).

        Returns:
            log_lik (torch.Tensor), (n, T, o, 4): Log-likelihood of results
                under means, i.e. of inferred z under generative model.

        """
        # choose std s.t., if predictions are 'bad', punishment should be high
        assert torch.all(self.transition_lik_std > 0)
        dist = Normal(means, self.transition_lik_std)

        log_lik = dist.log_prob(results)

        return log_lik

    def sequence_loss(
        self,
        x,
        z_s,
        z_img,
        z_dyn,
        log_z,
        rewards_predicted,
        actions=None,
        x_color=None,
        last_z=None,
        mask=None,
        rewards=None,
        beta=1.0,
        img_recon_weight=0.1,
        dyn_recon_weight=0.1,
    ):
        """Forward pass of STOVE.

        n (batch_size), T (sequence length), c (number of channels), w (image
        width), h (image_height)

        Args:
            x (torch.Tensor), (n, T, c, w, h): Sequences of images.
            actions (torch.Tensor), (n, T): Actions for action-conditioned video
                prediction.

        Returns:
            average_elbo (torch.Tensor) (1,): Mean ELBO over sequence.
            self.prop_dict (dict): Dictionary containing performance metrics.
                Used for logging and plotting.

        """
        batch = x.shape[0]
        T = x.shape[1]

        # if working with RL replay buffer
        # mask represents resets from a batch, which masks out invalid transitions

        # 3. Assemble sequence ELBO.

        # predict images from all embeddings
        imgs_forward = self.img_model.decode_img(z_s.flatten(end_dim=1))
        imgs_forward_dyn = self.img_model.decode_img(z_dyn.flatten(end_dim=1))
        imgs_forward_model = self.img_model.decode_img(z_img.flatten(end_dim=1))

        flat_img = x.flatten(end_dim=1)

        img_lik_forward = -((imgs_forward - flat_img) ** 2)
        img_lik_forward_dyn = -((imgs_forward_dyn - flat_img) ** 2)
        img_lik_forward_model = -((imgs_forward_model - flat_img) ** 2)

        # 3.2. Get q(z|x), sample log-likelihoods of inferred z states (n(T-2)).
        log_z_f_masked = mask * log_z.mean((-1))

        # 3.3 Get p(z_t|z_t-1), generative dynamics distribution.
        trans_lik = self.transition_lik(means=z_dyn, results=z_s)

        trans_lik_masked = mask * trans_lik.sum((-1))
        img_lik_forward_masked = mask * img_lik_forward.sum((-3, -2, -1)).view(batch, T)
        img_lik_model_masked = mask * img_lik_forward_model.sum((-3, -2, -1)).view(
            batch, T
        )
        img_lik_forward_dyn_masked = mask * img_lik_forward_dyn.sum((-3, -2, -1)).view(
            batch, T
        )

        # 3.4 Finally assemble ELBO.
        elbo = (trans_lik_masked + img_lik_forward_masked) - beta * log_z_f_masked
        augmented_elbo = (
            torch.mean(elbo)
            + img_recon_weight * img_lik_model_masked
            + dyn_recon_weight * img_lik_forward_dyn_masked
        )
        if rewards is not None:
            augmented_elbo -= torch.mean(
                (rewards.view(batch, T) - rewards_predicted.view(batch, T)) ** 2
            )

        return -torch.sum(augmented_elbo) / batch, {
            "img_lik": img_lik_forward_masked,
            "img_lik_model": img_lik_model_masked,
        }

    def train(
        self,
        x,
        actions=None,
        rewards=None,
        last_z=None,
        mask=None,
        x_color=None,
        beta=1.0,
        img_recon_weight=0.1,
        dyn_recon_weight=0.1,
    ):

        if mask is None:
            mask = torch.ones_like(x[:, :, 0, 0, 0]).squeeze(-1)
            if x.is_cuda:
                mask = mask.cuda()
        else:
            mask = 1.0 - torch.cumsum(mask, -1).squeeze(-1)

        (
            z,
            log_z,
            z_img,
            _,
            z_dyn,
            _,
            _,
            _,
            rewards_predicted,
        ) = self.forward(x, actions, rewards, last_z, mask)
        loss, loss_d = self.sequence_loss(
            x,
            z,
            z_img,
            z_dyn,
            log_z,
            rewards_predicted,
            actions,
            x_color=None,
            mask=mask,
            rewards=rewards,
            beta=beta,
            img_recon_weight=img_recon_weight,
            dyn_recon_weight=dyn_recon_weight,
        )
        d = {
            "z": z,
            "z_img": z_img,
            "z_dyn": z_dyn,
            "log_z": log_z,
            "rewards_pred": rewards_predicted,
            "img_lik": loss_d["img_lik"],
        }
        return loss, d
