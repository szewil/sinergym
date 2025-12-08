import copy
import gymnasium as gym
from gymnasium.envs.registration import register

BASE_ID = 'Eplus-5zone-hot-continuous-stochastic-v1'
spec = gym.spec(BASE_ID)
kwargs = copy.deepcopy(spec.kwargs)
print(kwargs)
# # Detect where actuators live (Sinergym versions differ)
# if 'actuators' in kwargs:
#     # Demo-style dict: {name: (type, control_type, component_name)}
#     acts = kwargs['actuators']
#     for k in ['Heating_Setpoint_RL', 'Cooling_Setpoint_RL', 'htg_setpoint', 'clg_setpoint']:
#         acts.pop(k, None)
# elif 'action_definition' in kwargs and 'actuators' in kwargs['action_definition']:
#     # Newer style list of dicts
#     filt = []
#     for a in kwargs['action_definition']['actuators']:
#         n = a.get('name','').lower()
#         ct = a.get('control_type','').lower()
#         if ('setpoint' in n) or ('setpoint' in ct):
#             continue
#         filt.append(a)
#     kwargs['action_definition']['actuators'] = filt
# else:
#     raise RuntimeError('Could not find actuators in env kwargs.')

# NEW_ID = BASE_ID.replace('stochastic', 'noSP-stochastic')
# # Ensure uniqueness if needed:
# if NEW_ID in [e.id for e in gym.envs.registration.registry.values()]:
#     NEW_ID += '-v1'

# register(id=NEW_ID, entry_point=spec.entry_point, kwargs=kwargs)

# env = gym.make(NEW_ID)
# print('New action space:', env.action_space)
