from feature_extractor import VANFeatureExtractor, AttentionalCNNFeatureExtractor
from feature_extractor import ResizeObservation
from models.visual_attention_network import van_based
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import NatureCNN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.utils import set_random_seed
from tqdm.auto import tqdm
import argparse
import gym
# from procgen import ProcgenGym3Env
# from stable_baselines3.common.vec_env import VecMonitor

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visual attention training")
    parser.add_argument("--env", type=str)
    parser.add_argument("--timesteps", type=int, default=2e6)
    parser.add_argument("--attention", type=str, default="van")
    parser.add_argument("--img_size", type=int, default=210)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()


    feature_extractors = {
        "cnn": dict(
            features_extractor_class=NatureCNN,
            features_extractor_kwargs=dict(features_dim=256),
        ),
        "acnn": dict(
            features_extractor_class=AttentionalCNNFeatureExtractor,
            features_extractor_kwargs=dict(features_dim=256),
        ),
        "van": dict(
            features_extractor_class=VANFeatureExtractor,
            features_extractor_kwargs=dict(
                features_dim=256, model_f=van_based, img_size=args.img_size
            ),
        ),
    }

    env = ResizeObservation(gym.make(args.env), args.img_size)
    # env = VecMonitor(venv=ProcgenGym3Env(num_envs=4, env_name=args.env))
    policy_kwargs = feature_extractors[args.attention]

    experiment = PPO(
        "CnnPolicy",
        env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=f"./{args.env}/{args.attention}",
    )
    experiment.learn(int(args.timesteps))

    # gc.collect()
    # env.reset()

    # avg_reward = evaluate_policy(experiment, env, n_eval_episodes=10)
    # print(avg_reward)
    # save model to folder
    experiment.save(f"./saved_models/{args.env}/{args.attention}")
