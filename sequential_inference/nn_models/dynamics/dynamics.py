import torch
import torch.distributions as dists
import torch.nn as nn

from sequential_inference.abc.sequence_model import AbstractLatentModel
from sequential_inference.nn_models.base.base_nets import (
    ConstantGaussian,
    Gaussian,
    OffsetGaussian,
    create_mlp,
)


class SimpleLatentNetwork(AbstractLatentModel):
    def __init__(
        self,
        action_shape,
        feature_dim=64,
        latent_dim=32,
        hidden_units=[64, 64],
        leaky_slope=0.2,
    ):
        super().__init__()

        self.latent_dim = latent_dim

        self.latent_init_prior = ConstantGaussian(latent_dim)
        self.latent_prior = OffsetGaussian(
            latent_dim + action_shape,
            latent_dim,
            hidden_units,
            leaky_slope=leaky_slope,
        )
        self.latent_init_posterior = Gaussian(
            feature_dim, latent_dim, hidden_units, leaky_slope=leaky_slope
        )
        self.latent_posterior = OffsetGaussian(
            feature_dim + latent_dim + action_shape,
            latent_dim,
            hidden_units,
            leaky_slope=leaky_slope,
        )

    def infer_prior(self, last_latent, action=None, global_belief=None):
        latent_dist = self.latent_prior([last_latent, action])
        latent_sample = latent_dist.rsample()
        return (
            latent_sample,
            latent_dist,
        )

    def infer_posterior(self, last_latent, state, action=None, global_belief=None):
        latent_dist = self.latent_posterior([state, last_latent, action])
        latent_sample = latent_dist.rsample()
        return (latent_sample, latent_dist)

    def obtain_initial(self, state, global_belief=None):
        prior_latent1_dist = self.latent_init_prior(state)
        prior_latent1_sample = prior_latent1_dist.rsample()

        post_latent1_dist = self.latent_init_posterior(state)
        post_latent1_sample = post_latent1_dist.rsample()

        return (prior_latent1_sample, prior_latent1_dist), (
            post_latent1_sample,
            post_latent1_dist,
        )


class SimpleStoveLatentNetwork(AbstractLatentModel):
    def __init__(
        self,
        action_shape,
        latent_dim=32,
        dynamics_latent_dim=32,
        hidden_units=[64, 64],
        leaky_slope=0.2,
    ):
        super().__init__()

        self.latent_dim = latent_dim

        self.latent_init_prior = ConstantGaussian(latent_dim + dynamics_latent_dim)
        self.latent_prior = OffsetGaussian(
            latent_dim + dynamics_latent_dim + action_shape,
            latent_dim + dynamics_latent_dim,
            hidden_units,
            leaky_slope=leaky_slope,
        )

    def infer_prior(self, last_latent, action=None, global_belief=None):
        latent_dist = self.latent_prior([last_latent, action])
        latent_sample = latent_dist.rsample()
        return (
            latent_sample,
            latent_dist,
        )

    def infer_posterior(self, last_latent, state, action=None, global_belief=None):
        latent_dist = self.latent_prior([last_latent, action])

        dyn_mean = latent_dist.mean[..., : self.latent_dim]
        dyn_std = latent_dist.std[..., : self.latent_dim]

        post_mean = state.mean
        post_std = state.std

        joined_z_mean = (
            torch.pow(post_std, 2) * dyn_mean + torch.pow(dyn_std, 2) * post_mean
        )
        joined_z_mean = joined_z_mean / (torch.pow(dyn_std, 2) + torch.pow(post_std, 2))
        joined_z_std = post_std * dyn_std
        joined_z_std = joined_z_std / torch.sqrt(
            torch.pow(dyn_std, 2) + torch.pow(post_std, 2)
        )
        joined_z_mean = torch.cat(
            (joined_z_mean, latent_dist.mean[..., self.latent_dim]), -1
        )
        joined_z_std = torch.cat(
            (joined_z_std, latent_dist.std[..., self.latent_dim]), -1
        )

        dist = dists.Normal(joined_z_mean, joined_z_std)

        return (dist.rsample(), dist)

    def obtain_initial(self, state, global_belief=None):
        prior_latent1_dist = self.latent_init_prior(state.mean)
        prior_latent1_sample = prior_latent1_dist.rsample()

        joined_z_mean = torch.cat(
            (state.mean, prior_latent1_dist.mean[..., self.latent_dim]), -1
        )
        joined_z_std = torch.cat(
            (state.std, prior_latent1_dist.std[..., self.latent_dim]), -1
        )

        post_latent1_dist = dists.Normal(joined_z_mean, joined_z_std)
        post_latent1_sample = post_latent1_dist.rsample()

        return (prior_latent1_sample, prior_latent1_dist), (
            post_latent1_sample,
            post_latent1_dist,
        )


class ConstantPriorLatentNetwork(AbstractLatentModel):
    def __init__(
        self,
        action_shape,
        feature_dim=32,
        latent_dim=32,
        hidden_units=[64, 64],
        leaky_slope=0.2,
    ):
        super().__init__()

        self.latent_dim = latent_dim

        self.latent_prior = ConstantGaussian(latent_dim)
        self.latent_init_posterior = Gaussian(
            feature_dim,
            latent_dim,
            hidden_units,
            leaky_slope=leaky_slope,
        )
        self.latent_posterior = OffsetGaussian(
            latent_dim + action_shape + feature_dim,
            latent_dim,
            hidden_units,
            leaky_slope=leaky_slope,
        )

    def infer_prior(self, last_latent, action=None, global_belief=None):
        latent_dist = self.latent_prior(last_latent)
        latent_sample = latent_dist.rsample()
        return (
            latent_sample,
            latent_dist,
        )

    def infer_posterior(self, last_latent, state, action=None, global_belief=None):
        latent_dist = self.latent_prior([last_latent, state, action])
        return (latent_dist.rsample(), latent_dist)

    def obtain_initial(self, state, global_belief=None):
        prior = self.infer_prior(state)
        posterior_dist = self.latent_init_posterior(state)
        posterior_sample = posterior_dist.rsample()
        return prior, (posterior_sample, posterior_dist)


class SLACLatentNetwork(AbstractLatentModel):
    def __init__(
        self,
        action_shape,
        feature_dim=64,
        latent1_dim=32,
        latent2_dim=32,
        belief_dim=0,
        hidden_units=[64, 64],
        leaky_slope=0.2,
    ):
        super().__init__()

        self.latent1_dim = latent1_dim
        self.latent2_dim = latent2_dim

        # p(z1(0)) = N(0, I)
        self.latent1_init_prior = ConstantGaussian(latent1_dim)
        # p(z2(0) | z1(0))
        self.latent2_init_prior = Gaussian(
            latent1_dim, latent2_dim, hidden_units, leaky_slope=leaky_slope
        )
        # p(z1(t+1) | z2(t), a(t))
        self.latent1_prior = Gaussian(
            latent2_dim + action_shape,
            latent1_dim,
            hidden_units,
            leaky_slope=leaky_slope,
        )
        # p(z2(t+1) | z1(t+1), z2(t), a(t))
        self.latent2_prior = Gaussian(
            latent1_dim + latent2_dim + action_shape,
            latent2_dim,
            hidden_units,
            leaky_slope=leaky_slope,
        )

        # q(z1(0) | feat(0))
        self.latent1_init_posterior = Gaussian(
            feature_dim, latent1_dim, hidden_units, leaky_slope=leaky_slope
        )
        # q(z2(0) | z1(0)) = p(z2(0) | z1(0))
        self.latent2_init_posterior = self.latent2_init_prior
        # q(z1(t+1) | feat(t+1), z2(t), a(t))
        self.latent1_posterior = Gaussian(
            feature_dim + latent2_dim + action_shape,
            latent1_dim,
            hidden_units,
            leaky_slope=leaky_slope,
        )
        # q(z2(t+1) | z1(t+1), z2(t), a(t)) = p(z2(t+1) | z1(t+1), z2(t), a(t))
        self.latent2_posterior = self.latent2_prior

        # p(r(t) | z1(t), z2(t), a(t), z1(t+1), z2(t+1))
        self.reward_predictor = Gaussian(
            latent1_dim + latent2_dim + action_shape,
            1,
            hidden_units,
            leaky_slope=leaky_slope,
        )

    def infer_prior(self, last_latent, action, global_belief=None):
        latent1_dist = self.latent1_prior(torch.cat([last_latent, action], -1))
        latent1_sample = latent1_dist.rsample()
        # p(z2(t) | z1(t), z2(t-1), a(t-1))
        latent2_dist = self.latent2_prior(
            torch.cat([latent1_sample, last_latent, action], -1)
        )
        latent2_sample = latent2_dist.rsample()
        return (latent1_sample, latent2_sample, latent1_dist, latent2_dist)

    def infer_posterior(self, last_latent, state, action=None, global_belief=None):
        latent1_dist = self.latent1_posterior(
            torch.cat([state, last_latent, action], -1)
        )
        latent1_sample = latent1_dist.rsample()
        # q(z2(t) | z1(t), z2(t-1), a(t-1))
        latent2_dist = self.latent2_posterior(
            torch.cat([latent1_sample, last_latent, action], -1)
        )
        latent2_sample = latent2_dist.rsample()
        return (latent1_sample, latent2_sample, latent1_dist, latent2_dist)

    def obtain_initial(self, state, global_belief=None):
        prior_latent1_dist = self.latent1_init_prior(state)
        prior_latent1_sample = prior_latent1_dist.rsample()
        prior_latent2_dist = self.latent2_init_prior(prior_latent1_sample)
        prior_latent2_sample = prior_latent2_dist.rsample()

        post_latent1_dist = self.latent1_init_posterior(state)
        post_latent1_sample = post_latent1_dist.rsample()
        post_latent2_dist = self.latent2_init_posterior(post_latent1_sample)
        post_latent2_sample = post_latent2_dist.rsample()

        return (
            prior_latent1_sample,
            prior_latent2_sample,
            prior_latent1_dist,
            prior_latent2_dist,
        ), (
            post_latent1_sample,
            post_latent2_sample,
            post_latent1_dist,
            post_latent2_dist,
        )


class PlaNetLatentNetwork(AbstractLatentModel):
    def __init__(
        self,
        action_dim,
        feature_dim=64,
        latent_dim=32,
        recurrent_hidden_dim=32,
        belief_dim=0,
        hidden_units=[64, 64],
        leaky_slope=0.2,
    ):
        super().__init__()

        self.feature_dim = feature_dim
        self.latent_dim = latent_dim

        assert latent_dim > recurrent_hidden_dim

        self.recurrent_hidden_dim = recurrent_hidden_dim
        self.stochastic_dim = self.latent_dim - self.recurrent_hidden_dim

        self.latent_init_prior = ConstantGaussian(self.stochastic_dim)
        self.rnn_init = ConstantGaussian(recurrent_hidden_dim, std=0.001)

        self.before_cell = create_mlp(
            self.stochastic_dim + action_dim[0], recurrent_hidden_dim, hidden_units
        )
        self.dynamics_cell = nn.GRUCell(recurrent_hidden_dim, recurrent_hidden_dim)
        self.prior = Gaussian(
            recurrent_hidden_dim, self.stochastic_dim, hidden_units=hidden_units
        )
        self.posterior = Gaussian(
            recurrent_hidden_dim + feature_dim,
            self.stochastic_dim,
            hidden_units=hidden_units,
        )

    def deter_encode(self, last_latent, action, global_belief=None):
        rnn_state = last_latent[..., self.stochastic_dim :]
        prev_state = last_latent[..., : self.stochastic_dim]
        if global_belief is not None:
            x = torch.cat([prev_state, global_belief, action], dim=-1)
        else:
            x = torch.cat([prev_state, action], dim=-1)
        x = self.before_cell(x)
        deter = self.dynamics_cell(
            x, rnn_state
        )  # b_tau, s_{t-1}, a_{t-1}, h^s_{t-1} --> h^s_t
        return deter

    def infer_prior(self, last_latent, action=None, global_belief=None):
        rnn_state = self.deter_encode(last_latent, action, global_belief=global_belief)
        latent_dist = self.prior(rnn_state)
        return (torch.cat((latent_dist.rsample(), rnn_state), -1), latent_dist)

    def infer_posterior(self, last_latent, state, action=None, global_belief=None):
        rnn_state = self.deter_encode(last_latent, action, global_belief=global_belief)
        enc = torch.cat((rnn_state, state), -1)
        latent_dist = self.posterior(enc)
        return (torch.cat((latent_dist.rsample(), rnn_state), -1), latent_dist)

    def obtain_initial(self, state, global_belief=None):
        prior_dist = self.latent_init_prior(state)
        last_rnn = self.rnn_init(state)

        rnn_state = last_rnn.rsample()
        prior_sample = prior_dist.rsample()
        prior_sample = torch.cat(((prior_sample, rnn_state)), -1)
        posterior_enc = torch.cat((rnn_state, state), -1)
        posterior_dist = self.posterior(posterior_enc)

        posterior_sample = torch.cat((posterior_dist.rsample(), rnn_state), -1)

        return (prior_sample, prior_dist), (posterior_sample, posterior_dist)
