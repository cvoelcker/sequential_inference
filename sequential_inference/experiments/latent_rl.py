class LatentTrainingExperiment(RLTrainingExperiment, ModelTrainingExperiment):

    is_rl = True
    is_model = True

    def __init__(
        self,
        env: Env,
        pass_rl_gradients_to_model: bool,
        batch_size: int,
        rl_batch_size: int,
        epoch_steps: int,
        epochs: int,
        log_frequency: int,
    ):
        super().__init__(env, batch_size, epoch_steps, epochs, log_frequency)
        self.rl_batch_size = rl_batch_size
        self.pass_rl_gradients_to_model = pass_rl_gradients_to_model

    def train_step(self) -> Dict[str, torch.Tensor]:
        batch = self.data.get_batch(self.batch_size)
        unpacked_batch = self.unpack_batch(batch)
        stats = self.model_train_step(*unpacked_batch)

        rl_batch = self.data.get_batch(self.rl_batch_size)
        rl_batch = self.unpack_batch(rl_batch)
        obs, act, rew, done = rl_batch
        if self.pass_rl_gradients_to_model:
            _, latents = self.model_algorithm.infer_sequence(obs, act, rew)
            loss, rl_stats = self.rl_algorithm.compute_loss(latents, act, rew, done)
            self.step_rl_optimizer(loss)
            self.step_model_optimizer(loss)
        else:
            with torch.no_grad():
                _, latents = self.model_algorithm.infer_sequence(obs, act, rew)
            loss, rl_stats = self.rl_algorithm.compute_loss(latents, act, rew, done)
            self.step_rl_optimizer(loss)

        return {**stats, **rl_stats}

    def get_agent(self) -> InferencePolicyAgent:
        agent: InferencePolicyAgent = self.rl_algorithm.get_agent()
        agent.latent = True
        agent.observation = False
        return InferencePolicyAgent(agent, self.model_algorithm)

