import argparse
import numpy as np
import yaml
from sinergym.utils.common import (
    create_environment,
    deep_update,
    import_from_path,
    process_environment_parameters,
)
from sinergym.utils.wrappers import (
    BuildStateWrapper,
    LaggedObservationWrapper,
)

# ====================== 1) FAN FLOW BINS ====================== #
# These match your Discrete action space for cooling/heating fans:
FLOW_BINS = np.array([0.0, 2.0, 4.0, 6.0, 8.0, 12.0], dtype=np.float32)


# ====================== 2) Bang-Bang Controller ====================== #
def bangbang_controller(temp, last_state=[False]):
    """
    Outputs: np.array([cooling_fan_flow, heating_fan_flow])
    Uses hysteresis logic to avoid rapid switching.
    """

    # last_state[0] stores 'is cooling' flag
    cooling = last_state[0]

    # Start cooling
    if not cooling and temp > 29.0:
        cooling = True

    # Stop cooling
    elif cooling and temp < 27.5:
        cooling = False

    last_state[0] = cooling

    if cooling:
        return np.array([12.0, 0.0], dtype=np.float32)   # Max cooling
    elif temp < 18.0:
        return np.array([0.0, 12.0], dtype=np.float32)   # Max heating
    else:
        return np.array([0.0, 0.0], dtype=np.float32)    # System OFF


# ====================== 3) Collect Stats ====================== #
def collect_stats_bangbang(env, num_episodes=3):
    observations = []
    obs_vars = env.get_wrapper_attr("observation_variables")
    air_temp_index = obs_vars.index("indoor_temperature")

    for ep in range(num_episodes):
        obs, _ = env.reset()
        done = False
        last_state = [False]  # memory for hysteresis

        while not done:
            temp = obs[air_temp_index]
            if isinstance(temp, np.ndarray):
                temp = temp.mean()

            # Fan control via bang-bang
            action = bangbang_controller(temp, last_state)

            # Collect obs BEFORE next step
            observations.append(obs)

            obs, _, done, truncated, _ = env.step(action)
            if done or truncated:
                break

    observations = np.array(observations)
    mu = np.mean(observations, axis=0)
    std = np.std(observations, axis=0) + 1e-8
    return mu, std


# ====================== 4) MAIN EXECUTION ====================== #
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--configuration", "-conf", required=True,
        help="Path to experiment configuration (YAML file)"
    )
    args = parser.parse_args()

    # Load YAML config
    with open(args.configuration, "r") as f:
        conf = yaml.safe_load(f)

    # ------------- Create Environment (same as training) ------------- #
    env_params = {}
    if conf.get("env_yaml_config"):
        with open(conf["env_yaml_config"], "r") as f:
            env_params.update(yaml.load(f, Loader=yaml.FullLoader))

    if conf.get("env_params"):
        env_params = deep_update(
            env_params, process_environment_parameters(conf["env_params"])
        )

    wrappers = {}
    if conf.get("wrappers_yaml_config"):
        with open(conf["wrappers_yaml_config"], "r") as f:
            wrappers = yaml.safe_load(f)

    env = create_environment(
        env_id=conf["environment"],
        env_params=env_params,
        wrappers=wrappers
    )
    env = BuildStateWrapper(env)
    env = LaggedObservationWrapper(env,          # 2) add history
                               lagged_variables=['indoor_temperature', 'outdoor_temperature'],
                               n_lags=3)
    obs_vars = env.get_wrapper_attr("observation_variables")
    print(obs_vars)

    print("✔ Environment created — collecting stats using fan-based Bang-Bang controller")

    # ------------- Collect & Save Stats ------------- #
    mu, std = collect_stats_bangbang(env, num_episodes=1)

    stats_dict = {}
  
    np.savez("mean_std_newsate.npz", mu=mu, std=std)

  # Save as plain txt for NormalizeObservation
    np.savetxt("mean.txt", mu, fmt="%.6f")
    np.savetxt("var.txt", std**2, fmt="%.6f")  # variance = std²

    print("✔ Saved mean.txt and var.txt for NormalizeObservation wrapper")
