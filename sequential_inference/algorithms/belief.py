from sequential_inference.models.base.network_util import calc_kl_divergence
from sequential_inference.abc.sequence_model import AbstractSequenceAlgorithm


class BeliefModelAlgorithm(AbstractSequenceAlgorithm):
    def __init__(self, vi_algorithm, belief_algorithm, expand_belief=True):
        super().__init__()
        self.vi_algorithm = vi_algorithm
        self.belief_algorithm = belief_algorithm
        self.expand_belief = expand_belief
        for k, v in self.vi_algorithm.buffer.items():
            self.register_module("vi_" + k, v)
        for k, v in self.belief_algorithm.buffer.items():
            self.register_module("belief_" + k, v)

    def infer_sequence(self, obs, actions=None, rewards=None):
        _, belief_posteriors = self.obtain_belief(obs, actions, rewards)
        belief_posterior_samples = belief_posteriors[0]
        priors, posteriors = self.vi_algorithm(
            obs, actions, rewards, belief_posterior_samples
        )
        priors = self.vi_algorithm.get_samples(priors)
        posteriors = self.vi_algorithm.get_samples(posteriors)
        return priors, posteriors

    def obtain_belief(self, obs, actions=None, rewards=None):
        belief_priors, belief_posteriors = self.belief_algorithm.infer_full_sequence(
            obs, actions, rewards=rewards
        )
        return belief_priors, belief_posteriors

    def infer_full_sequence(self, obs, actions=None, rewards=None):
        belief_priors, belief_posteriors = self.obtain_belief(obs, actions, rewards)
        belief_posterior_samples = belief_posteriors[0]

        if self.expand_belief:
            obs = self.expand(obs, "inp")
            actions = self.expand(actions, "inp")
            rewards = self.expand(rewards, "inp")
            belief_posteriors = self.expand(belief_posterior_samples, "belief")

        priors, posteriors = self.vi_algorithm(
            obs, actions, rewards, belief_posterior_samples
        )

        return (
            priors,
            posteriors,
            belief_priors,
            belief_posteriors,
            obs,
            actions,
            rewards,
        )

    def compute_loss(self, obs, actions=None, rewards=None):
        _, belief_posteriors = self.obtain_belief(obs, actions, rewards)
        belief_posterior_dist = self.belief_algorithm.get_dists(belief_posteriors)
        belief_posterior_samples = self.belief_algorithm.get_samples(belief_posteriors)

        if self.expand_belief:
            obs = self.expand(obs, "inp")
            actions = self.expand(actions, "inp")
            rewards = self.expand(rewards, "inp")
            belief_posteriors = self.expand(belief_posterior_samples, "belief")

        loss, stats = self.vi_algorithm.compute_loss(
            obs, actions, rewards, belief_posteriors
        )

        # KL divergence loss.
        kld_loss = calc_kl_divergence(
            belief_posterior_dist[:-1], belief_posterior_dist[1:]
        )
        loss -= kld_loss
        stats["belief_kld_loss"] = kld_loss.detach().cpu()

        return loss, stats

    def infer_single_step(
        self, last_latent, obs, action=None, rewards=None, global_belief=None
    ):
        belief = self.belief_algorithm.infer_single_step(
            global_belief, obs, action, rewards
        )
        posterior = self.vi_algorithm.infer_single_step(
            last_latent, obs, action, rewards, belief
        )
        return self.vi_algorithm.get_samples(posterior), belief

    def predict_sequence(
        self, initial_latent, actions=None, reward=None, global_belief=None
    ):
        return self.vi_algorithm.predict_sequence(
            initial_latent, actions, reward, global_belief
        )

    def expand(self, tensor, obs_type):
        batch_size = tensor.shape[0]
        horizon = tensor.shape[1]
        if obs_type == "inp":
            return (
                tensor.unsqueeze(1)
                .expand(-1, horizon, -1, -1)
                .reshape(batch_size * horizon, horizon, -1)
            )
        elif obs_type == "belief":
            return (
                tensor.unsqueeze(2)
                .expand(-1, -1, horizon, -1)
                .reshape(batch_size * horizon, horizon, -1)
            )
        else:
            raise KeyError(f"Tensor reshape operation invalid for {obs_type}")
