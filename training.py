from van import van_based
from feature_extractor import VANFeatureExtractionWrapper
from feature_extractor import ResizeObservation
from stable_baselines3 import PPO
# from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
# from tqdm.auto import tqdm
import sys
import gym

if __name__ == "__main__":
    _, environment_name, timesteps = sys.argv
    img_size = 210
    env = ResizeObservation(gym.make(environment_name), img_size)
    policy_kwargs = dict(
        features_extractor_class=VANFeatureExtractionWrapper,
        features_extractor_kwargs=dict(features_dim=256, model_f=van_based, img_size=img_size),
    )
    experiment = PPO("CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=1, tensorboard_log="./van_ms_pacman/")
    experiment.learn(int(timesteps))
    env.reset()
    avg_reward = evaluate_policy(experiment, env, n_eval_episodes=10)
    print(avg_reward)
    # save model to folder
    experiment.save("/saved_models/" + "based_" + f"{environment_name}")


# class TrainingCallback(BaseCallback):
#     """
#     :param pbar: (tqdm.pbar) Progress bar object
#     """
#     def __init__(self, pbar):
#         super(TrainingCallback, self).__init__()
#         self._pbar = pbar

#     def _on_step(self):
#         # Update the progress bar:
#         self._pbar.n = self.num_timesteps
#         self._pbar.update(0)

#     # def _on_rollout_end(self) -> None:
#     #     gc.collect()

# # this callback uses the 'with' block, allowing for correct initialisation and destruction
# class ProgressBarManager(object):
#     def __init__(self, total_timesteps): # init object with total timesteps
#         self.pbar = None
#         self.total_timesteps = total_timesteps
        
#     def __enter__(self): # create the progress bar and callback, return the callback
#         self.pbar = tqdm(total=self.total_timesteps)
            
#         return TrainingCallback(self.pbar)

#     def __exit__(self, exc_type, exc_val, exc_tb): # close the callback
#         self.pbar.n = self.total_timesteps
#         self.pbar.update(0)
#         self.pbar.close()