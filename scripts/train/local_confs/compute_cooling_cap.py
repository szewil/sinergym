"""Implementation of a bang-bang controller with energy tracking.

This script runs a bang-bang controller within a Sinergym environment. It has been
modified to specifically track and report the maximum energy consumed during any
single cooling event, which is a more direct measure of the energy needed to
bring the indoor temperature back into the comfort range.
"""

import argparse
import sys
import traceback
from datetime import datetime
import pdb
import time
import gymnasium as gym
import numpy as np
import wandb
import yaml
from stable_baselines3 import __version__ as sb3_version
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.logger import HumanOutputFormat
from stable_baselines3.common.logger import Logger as SB3Logger
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.utils import get_linear_fn, get_schedule_fn
from sinergym.utils.wrappers import MultiObsWrapper
from sinergym.utils.wrappers import DelayRewardWrapper
from sinergym.utils.wrappers import BuildStateWrapper, MinMaxNormWrapper
from torch import nn


import sinergym
import sinergym.utils.gcloud as gcloud
from sinergym.utils.callbacks import *
from sinergym.utils.common import (
    create_environment,
    deep_update,
    import_from_path,
    is_wrapped,
    process_algorithm_parameters,
    process_environment_parameters,
)
from sinergym.utils.logger import WandBOutputFormat
from sinergym.utils.wrappers import WandBLogger
from sinergym.utils.common import import_from_path

# ---------------------------------------------------------------------------- #
#                          PLR Action Space Definition                         #
# ---------------------------------------------------------------------------- #

PLR_BINS = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float32)

BINS = {
    "heat_set": np.array([18.5]),
    "cool_set": np.array([40.0]),
    "plr_ac": PLR_BINS,
    "plr_heat": PLR_BINS
}

DISCRETE_SPACE = gym.spaces.MultiDiscrete([
    len(BINS["heat_set"]),
    len(BINS["cool_set"]),
    len(BINS["plr_ac"]),
    len(BINS["plr_ac"]),
    len(BINS["plr_ac"]),
    len(BINS["plr_heat"]),
    len(BINS["plr_heat"]),
])

def map_discrete_to_continuous_action(a):
    """
    Maps a list of discrete action indices to a continuous action array.
    This function is used to convert our simple bang-bang action to the format
    expected by the environment.
    """
    heat_sp = BINS["heat_set"][a[0]]
    cool_sp = BINS["cool_set"][a[1]]
    ac1, ac2, ac3 = BINS["plr_ac"][a[2:5]]
    h1, h2 = BINS["plr_heat"][a[5:7]]

    return np.array(
        [heat_sp, cool_sp, ac1, ac2, ac3, h1, h2],
        dtype=np.float32,
    )

# ---------------------------------------------------------------------------- #
#                           Bang-Bang Controller                             #
# ---------------------------------------------------------------------------- #

def bangbang_controller(indoor_temp, outdoor_temp, last_state=[False, False]):
    """
    Returns a discrete action based on the air temperature and the 18-24°C
    comfort band, using a hysteresis logic.
    The action controls the PLR of the cooling and heating units.
    
    MODIFIED LOGIC:
    - No heating is ever performed.
    - Cooling is activated continuously once the outdoor temperature reaches 40°C.
    - Cooling remains active until the indoor temperature is at or below 23°C.
    """
    cooling_on = last_state[0]
    heating_on = last_state[1]

    # New logic based on user's request
    # Hysteresis logic for cooling
    if not cooling_on and outdoor_temp >= 40.0: # Turn on cooling if it gets too warm
        cooling_on = True
        
    elif cooling_on and indoor_temp <= 23.0: # Turn off cooling once it hits comfort range
        cooling_on = False

    last_state[0] = cooling_on
    last_state[1] = False # Heating is always off

    if cooling_on:
        # Action to cool: set ALL AC PLR to max (1.0)
        # The indices for the PLR bins are 0 to 4. Max is index 4.
        # Fixed setpoints from BINS are at index 0.
        return [0, 0, 4, 4, 4, 0, 0]
    else:
        # Action to be idle: all PLR values are 0.0 (index 0)
        return [0, 0, 0, 0, 0, 0, 0]

def init_running():
    return {"cooling_minutes": 0,
            "heating_minutes": 0,
            "off_minutes": 0}

# ------------------------ Load configuration file ------------------------ #
parser = argparse.ArgumentParser()
parser.add_argument(
    '--configuration',
    '-conf',
    required=True,
    type=str,
    help='Path to experiment configuration (YAML file)'
)
args = parser.parse_args()

with open(args.configuration, 'r') as f:
    conf = yaml.safe_load(f)

# ------------------------ Setup experiment name ------------------------ #
experiment_date = datetime.today().strftime('%Y-%m-%d_%H-%M')
experiment_name = f"{conf['experiment_name']}_bangbang_plr_{experiment_date}"

# ------------------------ Prepare environment params ------------------------ #
env_params = {}
if conf.get('env_yaml_config'):
    with open(conf['env_yaml_config'], 'r') as f:
        env_params.update(yaml.load(f, Loader=yaml.FullLoader))

if conf.get('env_params'):
    env_params = deep_update(env_params, process_environment_parameters(conf['env_params']))

env_params.update({'env_name': experiment_name})

# ------------------------ Wrappers ------------------------ #
wrappers = {}
if conf.get('wrappers'):
    for wrapper in conf['wrappers']:
        for wrapper_name, wrapper_arguments in wrapper.items():
            for name, value in wrapper_arguments.items():
                if isinstance(value, str) and ':' in value:
                    wrapper_arguments[name] = import_from_path(value)
            wrappers = deep_update(wrappers, {wrapper_name: wrapper_arguments})

# ------------------------ Create environment ------------------------ #
# Add the discretization wrapper with the new action mapping
wrappers['sinergym.utils.wrappers:DiscretizeEnv'] = {
    'discrete_space': DISCRETE_SPACE,
    'action_mapping': map_discrete_to_continuous_action
}

env = create_environment(
    env_id=conf['environment'],
    env_params=env_params,
    wrappers=wrappers
)


# After you create the env but before training/running
print("Actuator order seen by Sinergym:")
for i, act in enumerate(env.get_wrapper_attr("actuators")):
    print(f"action[{i}] → {act}")

# ------------------------ Run Bang-Bang Control ------------------------ #
print("Running Bang-Bang control...")
# List to store the total energy consumed for each cooling event
cooling_event_energies = []
# Accumulator for the current cooling event's energy
current_cooling_energy = 0.0
is_cooling_active = False # Flag to track if a cooling event is in progress

try:
    total_episodes = conf['episodes']
    timesteps_per_episode = env.get_wrapper_attr('timestep_per_episode')
    
    # We will need the timestep duration to calculate energy (Power * Time)
    timestep_duration_s = env.get_wrapper_attr('timestep')
    timestep_duration_h = timestep_duration_s / 3600.0 # Convert to hours for kWh

    for episode in range(total_episodes):
        obs, _ = env.reset()
        running = init_running() # counters for this episode
        done = False
        obs_vars = env.get_wrapper_attr('observation_variables')
        
        sim_time = 0.0 # hours
        next_print = 1.0
        
        # We need to track the last state for the hysteresis controller
        last_state = [False, False]
        
        for _ in range(timesteps_per_episode):
            # Find indices for the temperatures
            indoor_temp_index = obs_vars.index('indoor_temperature')
            outdoor_temp_index = obs_vars.index('outdoor_temperature')
            
            # Extract the temperature values
            indoor_temp = obs[indoor_temp_index]
            outdoor_temp = obs[outdoor_temp_index]
            
            # Use the new bang-bang controller to get a discrete action
            discrete_action = bangbang_controller(indoor_temp, outdoor_temp, last_state)
            
            # Step the environment with the discrete action
            obs, reward, done, truncated, info = env.step(discrete_action)
            
            sim_time = info["time_elapsed(hours)"]

            # ---------------------------------------------------------------
            # --- MODIFIED LOGIC: Track energy per cooling event ---
            # ---------------------------------------------------------------
            # The bangbang_controller's last_state[0] flag is true when cooling
            # is active, which is a better trigger than outdoor temperature.
            if last_state[0]:
                # If cooling is on, add the current power to the accumulator
                # Ensure 'total_power_demand' is in the info dict
                if 'total_power_demand' in info:
                    current_cooling_energy += info['total_power_demand'] * timestep_duration_h
                
                # Set the flag to indicate a cooling event is in progress
                is_cooling_active = True
            
            # Check if cooling has just ended
            elif is_cooling_active and not last_state[0]:
                # A cooling event has ended, so save the total accumulated energy
                cooling_event_energies.append(current_cooling_energy)
                # Reset the accumulator and the flag for the next event
                current_cooling_energy = 0.0
                is_cooling_active = False

            if done or truncated:
                # If the episode ends while cooling is active, save the current accumulated energy
                if is_cooling_active:
                    cooling_event_energies.append(current_cooling_energy)
                break
        
        print("-" * 40, flush=True)
        print(f"Episode {episode + 1} completed.")

finally:
    if env.get_wrapper_attr('is_running'):
        env.close()

    # ---------------------------------------------------------------
    # Calculate and print the maximum cooling event energy
    # ---------------------------------------------------------------
    if cooling_event_energies:
        max_cooling_energy = max(cooling_event_energies)
        print(f"\n--- Maximum Cooling Event Energy Found ---")
        print(f"Maximum energy (kWh) to cool down the building in one event: {max_cooling_energy:.2f} kWh")
    else:
        print("\n--- No Cooling Data ---")
        print("No cooling events were recorded during the simulation.")
