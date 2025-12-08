"""Implementation of reward functions."""

from datetime import datetime
from math import exp
from typing import Any, Dict, List, Optional, Tuple, Union

from sinergym.utils.constants import LOG_REWARD_LEVEL, YEAR
from sinergym.utils.logger import TerminalLogger
import numpy as np


class BaseReward(object):

    logger = TerminalLogger().getLogger(name='REWARD', level=LOG_REWARD_LEVEL)

    def __init__(self):
        """
        Base reward class.

        All reward functions should inherit from this class.

        Args:
            env (Env): Gym environment.
        """

    def __call__(self, obs_dict: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """Method for calculating the reward function."""
        raise NotImplementedError("Reward class must have a `__call__` method.")


class LinearReward(BaseReward):

    def __init__(
        self,
        temperature_variables: List[str],
        energy_variables: List[str],
        range_comfort_winter: Tuple[float, float],
        range_comfort_summer: Tuple[float, float],
        summer_start: Tuple[int, int] = (6, 1),
        summer_final: Tuple[int, int] = (9, 30),
        energy_weight: float = 0.5,
        lambda_energy: float = 1.0,
        lambda_temperature: float = 1.0,
    ):
        """
        Linear reward function.

        It considers the energy consumption and the absolute difference to temperature comfort.

        .. math::
            R = - W * lambda_E * power - (1 - W) * lambda_T * (max(T - T_{low}, 0) + max(T_{up} - T, 0))

        Args:
            temperature_variables (List[str]): Name(s) of the temperature variable(s).
            energy_variables (List[str]): Name(s) of the energy/power variable(s).
            range_comfort_winter (Tuple[float,float]): Temperature comfort range for cold season. Depends on environment you are using.
            range_comfort_summer (Tuple[float,float]): Temperature comfort range for hot season. Depends on environment you are using.
            summer_start (Tuple[int,int]): Summer session tuple with month and day start. Defaults to (6,1).
            summer_final (Tuple[int,int]): Summer session tuple with month and day end. defaults to (9,30).
            energy_weight (float, optional): Weight given to the energy term. Defaults to 0.5.
            lambda_energy (float, optional): Constant for removing dimensions from power(1/W). Defaults to 1e-4.
            lambda_temperature (float, optional): Constant for removing dimensions from temperature(1/C). Defaults to 1.0.
        """

        super().__init__()

        # Basic validations
        if not (0 <= energy_weight <= 1):
            self.logger.error(
                f'energy_weight must be between 0 and 1. Received: {energy_weight}'
            )
            raise ValueError
        if not all(
            isinstance(v, str) for v in temperature_variables + energy_variables
        ):
            self.logger.error('All variable names must be strings.')
            raise TypeError

        # Name of the variables
        self.temp_names = temperature_variables
        self.energy_names = energy_variables

        # Reward parameters
        self.range_comfort_winter = range_comfort_winter
        self.range_comfort_summer = range_comfort_summer
        self.W_energy = energy_weight
        self.lambda_energy = lambda_energy
        self.lambda_temp = lambda_temperature

        # Summer period
        self.summer_start = summer_start  # (month, day)
        self.summer_final = summer_final  # (month, day)

        self.logger.info('Reward function initialized.')

    def __call__(self, obs_dict: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """Calculate the reward function value based on observation data.

        Args:
            obs_dict (Dict[str, Any]): Dict with observation variable name (key) and observation variable value (value)

        Returns:
            Tuple[float, Dict[str, Any]]: Reward value and dictionary with their individual components.
        """

        # Energy calculation
        energy_values = self._get_energy_consumed(obs_dict)
        self.total_energy = sum(energy_values)
        self.energy_penalty = -self.total_energy

        # Comfort violation calculation
        temp_violations = self._get_temperature_violation(obs_dict)
        self.total_temp_violation = sum(temp_violations)
        self.comfort_penalty = -self.total_temp_violation

        # Weighted sum of both terms
        reward, energy_term, comfort_term = self._get_reward(obs_dict)

        reward_terms = {
            'energy_term': energy_term,
            'comfort_term': comfort_term,
            'energy_penalty': self.energy_penalty,
            'comfort_penalty': self.comfort_penalty,
            'total_power_demand': self.total_energy,
            'total_temperature_violation': self.total_temp_violation,
            'reward_weight': self.W_energy,
        }

        return reward, reward_terms

    def _get_energy_consumed(self, obs_dict: Dict[str, Any]) -> List[float]:
        """Calculate the energy consumed in the current observation.

        Args:
            obs_dict (Dict[str, Any]): Environment observation.

        Returns:
            List[float]: List with energy consumed in each energy variable.
        """
        return [obs_dict[v] for v in self.energy_names]

    def _get_temperature_violation(self, obs_dict: Dict[str, Any]) -> List[float]:
        """Calculate the temperature violation (ºC) in each observation's temperature variable.

        Returns:
            List[float]: List with temperature violation in each zone.
        """

        # Current datetime and summer period
        current_dt = datetime(
            YEAR, int(obs_dict['month']), int(obs_dict['day_of_month'])
        )
        summer_start_date = datetime(YEAR, *self.summer_start)
        summer_final_date = datetime(YEAR, *self.summer_final)

        temp_range = (
            self.range_comfort_summer
            if summer_start_date <= current_dt <= summer_final_date
            else self.range_comfort_winter
        )

        temp_values = [obs_dict[v] for v in self.temp_names]

        return [max(temp_range[0] - T, 0, T - temp_range[1]) for T in temp_values]

    def _get_reward(self, obs_dict) -> Tuple[float, ...]:
        """Compute the final reward value.

        Args:
            energy_penalty (float): Negative absolute energy penalty value.
            comfort_penalty (float): Negative absolute comfort penalty value.

        Returns:
            Tuple[float, ...]: Total reward calculated and reward terms.
        """
        energy_term = self.lambda_energy * self.W_energy * self.energy_penalty
        comfort_term = self.lambda_temp * (1 - self.W_energy) * self.comfort_penalty
        reward = energy_term + comfort_term
        reward = obs_dict['indoor_temperature']
        return reward, energy_term, comfort_term


class EnergyCostLinearReward(LinearReward):

    def __init__(
        self,
        temperature_variables: List[str],
        energy_variables: List[str],
        range_comfort_winter: Tuple[float, float],
        range_comfort_summer: Tuple[float, float],
        energy_cost_variables: List[str],
        summer_start: Tuple[int, int] = (6, 1),
        summer_final: Tuple[int, int] = (9, 30),
        energy_weight: float = 0.4,
        temperature_weight: float = 0.4,
        lambda_energy: float = 1.0,
        lambda_temperature: float = 1.0,
        lambda_energy_cost: float = 1.0,
    ):
        """
        Linear reward function with the addition of the energy cost term.

        Considers energy consumption, absolute difference to thermal comfort and energy cost.

        .. math::
            R = - W_E * lambda_E * power - W_T * lambda_T * (max(T - T_{low}, 0) + max(T_{up} - T, 0)) - (1 - W_P - W_T) * lambda_EC * power_cost

        Args:
            temperature_variables (List[str]): Name(s) of the temperature variable(s).
            energy_variables (List[str]): Name(s) of the energy/power variable(s).
            range_comfort_winter (Tuple[float,float]): Temperature comfort range for cold season. Depends on environment you are using.
            range_comfort_summer (Tuple[float,float]): Temperature comfort range for hot season. Depends on environment you are using.
            summer_start (Tuple[int,int]): Summer session tuple with month and day start. Defaults to (6,1).
            summer_final (Tuple[int,int]): Summer session tuple with month and day end. defaults to (9,30).
            energy_weight (float, optional): Weight given to the energy term. Defaults to 0.4.
            temperature_weight (float, optional): Weight given to the temperature term. Defaults to 0.4.
            lambda_energy (float, optional): Constant for removing dimensions from power(1/W). Defaults to 1.0.
            lambda_temperature (float, optional): Constant for removing dimensions from temperature(1/C). Defaults to 1.0.
            lambda_energy_cost (float, optional): Constant for removing dimensions from temperature(1/E). Defaults to 1.0.
        """

        super().__init__(
            temperature_variables,
            energy_variables,
            range_comfort_winter,
            range_comfort_summer,
            summer_start,
            summer_final,
            energy_weight,
            lambda_energy,
            lambda_temperature,
        )

        self.energy_cost_names = energy_cost_variables
        self.W_temperature = temperature_weight
        self.lambda_energy_cost = lambda_energy_cost

        self.logger.info('Reward function initialized.')

    def __call__(self, obs_dict: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """Calculate the reward function.

        Args:
            obs_dict (Dict[str, Any]): Dict with observation variable name (key) and observation variable value (value)

        Returns:
            Tuple[float, Dict[str, Any]]: Reward value and dictionary with their individual components.
        """
        # Energy calculation
        energy_values = self._get_energy_consumed(obs_dict)
        self.total_energy = sum(energy_values)
        self.energy_penalty = -self.total_energy

        # Comfort violation calculation
        temp_violations = self._get_temperature_violation(obs_dict)
        self.total_temp_violation = sum(temp_violations)
        self.comfort_penalty = -self.total_temp_violation

        # Energy cost calculation
        energy_cost_values = self._get_money_spent(obs_dict)
        self.total_energy_cost = sum(energy_cost_values)
        self.energy_cost_penalty = -self.total_energy_cost

        # Weighted sum of terms
        reward, energy_term, comfort_term, energy_cost_term = self._get_reward()

        reward_terms = {
            'energy_term': energy_term,
            'comfort_term': comfort_term,
            'energy_cost_term': energy_cost_term,
            'reward_energy_weight': self.W_energy,
            'reward_temperature_weight': self.W_temperature,
            'energy_penalty': self.energy_penalty,
            'comfort_penalty': self.comfort_penalty,
            'energy_cost_penalty': self.energy_cost_penalty,
            'total_power_demand': self.total_energy,
            'total_temperature_violation': self.total_temp_violation,
            'money_spent': self.total_energy_cost,
        }

        return reward, reward_terms

    def _get_money_spent(self, obs_dict: Dict[str, Any]) -> List[float]:
        """Calculate the total money spent in the current observation.

        Args:
            obs_dict (Dict[str, Any]): Environment observation.

        Returns:
            List[float]: List with money spent in each energy cost variable.
        """
        return [v for k, v in obs_dict.items() if k in self.energy_cost_names]

    def _get_reward(self) -> Tuple[float, ...]:
        """It calculates reward value using the negative absolute comfort, energy penalty and energy cost penalty calculates previously.

        Returns:
            Tuple[float, ...]: Total reward calculated, reward term for energy, reward term for comfort and reward term for energy cost.
        """
        energy_term = self.lambda_energy * self.W_energy * self.energy_penalty
        comfort_term = self.lambda_temp * self.W_temperature * self.comfort_penalty
        energy_cost_term = (
            self.lambda_energy_cost
            * (1 - self.W_energy - self.W_temperature)
            * self.energy_cost_penalty
        )

        reward = energy_term + comfort_term + energy_cost_term
        return reward, energy_term, comfort_term, energy_cost_term


class ExpReward(LinearReward):

    def __init__(
        self,
        temperature_variables: List[str],
        energy_variables: List[str],
        range_comfort_winter: Tuple[float, float],
        range_comfort_summer: Tuple[float, float],
        summer_start: Tuple[int, int] = (6, 1),
        summer_final: Tuple[int, int] = (9, 30),
        energy_weight: float = 0.5,
        lambda_energy: float = 1.0,
        lambda_temperature: float = 1.0,
    ):
        """
        Reward considering exponential absolute difference to temperature comfort.

        .. math::
            R = - W * lambda_E * power - (1 - W) * lambda_T * exp( (max(T - T_{low}, 0) + max(T_{up} - T, 0)) )

        Args:
            temperature_variables (List[str]): Name(s) of the temperature variable(s).
            energy_variables (List[str]): Name(s) of the energy/power variable(s).
            range_comfort_winter (Tuple[float,float]): Temperature comfort range for cold season. Depends on environment you are using.
            range_comfort_summer (Tuple[float,float]): Temperature comfort range for hot season. Depends on environment you are using.
            summer_start (Tuple[int,int]): Summer session tuple with month and day start. Defaults to (6,1).
            summer_final (Tuple[int,int]): Summer session tuple with month and day end. defaults to (9,30).
            energy_weight (float, optional): Weight given to the energy term. Defaults to 0.5.
            lambda_energy (float, optional): Constant for removing dimensions from power(1/W). Defaults to 1e-4.
            lambda_temperature (float, optional): Constant for removing dimensions from temperature(1/C). Defaults to 1.0.
        """

        super().__init__(
            temperature_variables,
            energy_variables,
            range_comfort_winter,
            range_comfort_summer,
            summer_start,
            summer_final,
            energy_weight,
            lambda_energy,
            lambda_temperature,
        )

    def __call__(self, obs_dict: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """Calculate the reward function value based on observation data.

        Args:
            obs_dict (Dict[str, Any]): Dict with observation variable name (key) and observation variable value (value)

        Returns:
            Tuple[float, Dict[str, Any]]: Reward value and dictionary with their individual components.
        """

        # Energy calculation
        energy_values = self._get_energy_consumed(obs_dict)
        self.total_energy = sum(energy_values)
        self.energy_penalty = -self.total_energy

        # Comfort violation calculation
        temp_violations = self._get_temperature_violation(obs_dict)
        self.total_temp_violation = sum(temp_violations)
        # Exponential Penalty
        self.comfort_penalty = -sum(
            exp(violation) for violation in temp_violations if violation > 0
        )

        # Weighted sum of both terms
        reward, energy_term, comfort_term = self._get_reward()

        reward_terms = {
            'energy_term': energy_term,
            'comfort_term': comfort_term,
            'energy_penalty': self.energy_penalty,
            'comfort_penalty': self.comfort_penalty,
            'total_power_demand': self.total_energy,
            'total_temperature_violation': self.total_temp_violation,
            'reward_weight': self.W_energy,
        }

        return reward, reward_terms

class SegExpReward(LinearReward):

    def __init__(
        self,
        temperature_variables: List[str],
        energy_variables: List[str],
        range_comfort_winter: Tuple[int, int],
        range_comfort_summer: Tuple[int, int],
        summer_start: Tuple[int, int] = (6, 1),
        summer_final: Tuple[int, int] = (9, 30),
        energy_weight: float = 0.5,
        lambda_energy: float = 1.0,
        lambda_temperature: float = 1.0
    ):
        """
        Reward considering exponential absolute difference to temperature comfort.

        .. math::
            R = - W * lambda_E * power - (1 - W) * lambda_T * exp( (max(T - T_{low}, 0) + max(T_{up} - T, 0)) )

        Args:
            temperature_variables (List[str]): Name(s) of the temperature variable(s).
            energy_variables (List[str]): Name(s) of the energy/power variable(s).
            range_comfort_winter (Tuple[int,int]): Temperature comfort range for cold season. Depends on environment you are using.
            range_comfort_summer (Tuple[int,int]): Temperature comfort range for hot season. Depends on environment you are using.
            summer_start (Tuple[int,int]): Summer session tuple with month and day start. Defaults to (6,1).
            summer_final (Tuple[int,int]): Summer session tuple with month and day end. defaults to (9,30).
            energy_weight (float, optional): Weight given to the energy term. Defaults to 0.5.
            lambda_energy (float, optional): Constant for removing dimensions from power(1/W). Defaults to 1e-4.
            lambda_temperature (float, optional): Constant for removing dimensions from temperature(1/C). Defaults to 1.0.
        """

        super().__init__(
            temperature_variables,
            energy_variables,
            range_comfort_winter,
            range_comfort_summer,
            summer_start,
            summer_final,
            energy_weight,
            lambda_energy,
            lambda_temperature
        )

    def __call__(self, obs_dict: Dict[str, Any]
                 ) -> Tuple[float, Dict[str, Any]]:
        """Calculate the reward function value based on observation data.

        Args:
            obs_dict (Dict[str, Any]): Dict with observation variable name (key) and observation variable value (value)

        Returns:
            Tuple[float, Dict[str, Any]]: Reward value and dictionary with their individual components.
        """

        # Energy calculation
        energy_values = self._get_energy_consumed(obs_dict)
        self.total_energy = sum(energy_values)
        self.energy_penalty = -self.total_energy

        # Comfort violation calculation
        temp_violations = self._get_temperature_violation(obs_dict)
        self.total_temp_violation = sum(temp_violations)
        # Exponential Penalty
        self.comfort_penalty = -sum(2.0 / (1.0 + exp(-violation)) - 1.0
                               for violation in temp_violations if violation > 0)

        # Weighted sum of both terms
        reward, energy_term, comfort_term = self._get_reward()

        reward_terms = {
            'energy_term': energy_term,
            'comfort_term': comfort_term,
            'energy_penalty': self.energy_penalty,
            'comfort_penalty': self.comfort_penalty,
            'total_power_demand': self.total_energy,
            'total_temperature_violation': self.total_temp_violation,
            'reward_weight': self.W_energy
        }

        return reward, reward_terms


class HourlyLinearReward(LinearReward):

    def __init__(
        self,
        temperature_variables: List[str],
        energy_variables: List[str],
        range_comfort_winter: Tuple[float, float],
        range_comfort_summer: Tuple[float, float],
        summer_start: Tuple[int, int] = (6, 1),
        summer_final: Tuple[int, int] = (9, 30),
        default_energy_weight: float = 0.5,
        lambda_energy: float = 1.0,
        lambda_temperature: float = 1.0,
        range_comfort_hours: tuple = (9, 19),
    ):
        """
        Linear reward function with a time-dependent weight for consumption and energy terms.

        Args:
            temperature_variables (List[str]]): Name(s) of the temperature variable(s).
            energy_variables (List[str]): Name(s) of the energy/power variable(s).
            range_comfort_winter (Tuple[float,float]): Temperature comfort range for cold season. Depends on environment you are using.
            range_comfort_summer (Tuple[float,float]): Temperature comfort range for hot season. Depends on environment you are using.
            summer_start (Tuple[int,int]): Summer session tuple with month and day start. Defaults to (6,1).
            summer_final (Tuple[int,int]): Summer session tuple with month and day end. defaults to (9,30).
            default_energy_weight (float, optional): Default weight given to the energy term when thermal comfort is considered. Defaults to 0.5.
            lambda_energy (float, optional): Constant for removing dimensions from power(1/W). Defaults to 1e-4.
            lambda_temperature (float, optional): Constant for removing dimensions from temperature(1/C). Defaults to 1.0.
            range_comfort_hours (tuple, optional): Hours where thermal comfort is considered. Defaults to (9, 19).
        """

        super(HourlyLinearReward, self).__init__(
            temperature_variables,
            energy_variables,
            range_comfort_winter,
            range_comfort_summer,
            summer_start,
            summer_final,
            default_energy_weight,
            lambda_energy,
            lambda_temperature,
        )

        # Reward parameters
        self.range_comfort_hours = range_comfort_hours
        self.default_energy_weight = default_energy_weight

    def __call__(self, obs_dict: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """Calculate the reward function.

        Args:
            obs_dict (Dict[str, Any]): Dict with observation variable name (key) and observation variable value (value)

        Returns:
            Tuple[float, Dict[str, Any]]: Reward value and dictionary with their individual components.
        """
        # Energy calculation
        energy_values = self._get_energy_consumed(obs_dict)
        self.total_energy = sum(energy_values)
        self.energy_penalty = -self.total_energy

        # Comfort violation calculation
        temp_violations = self._get_temperature_violation(obs_dict)
        self.total_temp_violation = sum(temp_violations)
        self.comfort_penalty = -self.total_temp_violation

        # Determine reward weight depending on the hour
        self.W_energy = (
            self.default_energy_weight
            if self.range_comfort_hours[0]
            <= obs_dict['hour']
            <= self.range_comfort_hours[1]
            else 1.0
        )

        # Weighted sum of both terms
        reward, energy_term, comfort_term = self._get_reward()

        reward_terms = {
            'energy_term': energy_term,
            'comfort_term': comfort_term,
            'energy_penalty': self.energy_penalty,
            'comfort_penalty': self.comfort_penalty,
            'total_power_demand': self.total_energy,
            'total_temperature_violation': self.total_temp_violation,
            'reward_weight': self.W_energy,
        }

        return reward, reward_terms


class NormalizedLinearReward(LinearReward):

    def __init__(
        self,
        temperature_variables: List[str],
        energy_variables: List[str],
        range_comfort_winter: Tuple[float, float],
        range_comfort_summer: Tuple[float, float],
        summer_start: Tuple[int, int] = (6, 1),
        summer_final: Tuple[int, int] = (9, 30),
        energy_weight: float = 0.5,
        max_energy_penalty: float = 8,
        max_comfort_penalty: float = 12,
        lambda_energy: float = 1.0,           # <— add this
        lambda_temperature: float = 1.0,      # <— and this
    ):
        """
        Linear reward function with a time-dependent weight for consumption and energy terms.

        Args:
            temperature_variables (List[str]]): Name(s) of the temperature variable(s).
            energy_variables (List[str]): Name(s) of the energy/power variable(s).
            range_comfort_winter (Tuple[float,float]): Temperature comfort range for cold season. Depends on environment you are using.
            range_comfort_summer (Tuple[float,float]): Temperature comfort range for hot season. Depends on environment you are using.
            summer_start (Tuple[int,int]): Summer session tuple with month and day start. Defaults to (6,1).
            summer_final (Tuple[int,int]): Summer session tuple with month and day end. defaults to (9,30).
            energy_weight (float, optional): Default weight given to the energy term when thermal comfort is considered. Defaults to 0.5.
            max_energy_penalty (float, optional): Maximum energy penalty value. Defaults to 8.
            max_comfort_penalty (float, optional): Maximum comfort penalty value. Defaults to 12.
        """

        super().__init__(
            temperature_variables,
            energy_variables,
            range_comfort_winter,
            range_comfort_summer,
            summer_start,
            summer_final,
            energy_weight,
        )

        # Reward parameters
        self.max_energy_penalty = max_energy_penalty
        self.max_comfort_penalty = max_comfort_penalty

    def _get_reward(self) -> Tuple[float, ...]:
        """It calculates reward value using energy consumption and grades of temperature out of comfort range. Applying normalization

        Returns:
            Tuple[float, ...]: total reward calculated, reward term for energy and reward term for comfort.
        """
        # Update max energy and comfort
        self.max_energy_penalty = max(self.max_energy_penalty, self.energy_penalty)
        self.max_comfort_penalty = max(self.max_comfort_penalty, self.comfort_penalty)

        # Calculate normalization
        energy_norm = (
            self.energy_penalty / self.max_energy_penalty
            if self.max_energy_penalty
            else 0
        )
        comfort_norm = (
            self.comfort_penalty / self.max_comfort_penalty
            if self.max_comfort_penalty
            else 0
        )

        # Calculate reward terms with norm values
        energy_term = self.W_energy * energy_norm
        comfort_term = (1 - self.W_energy) * comfort_norm
        reward = energy_term + comfort_term
        return reward, energy_term, comfort_term


class NewNormalizedLinearReward(BaseReward):

    def __init__(
        self,
        temperature_variables: List[str],
        energy_variables: List[str],
        range_comfort_winter: Tuple[int, int],
        range_comfort_summer: Tuple[int, int],
        summer_start: Tuple[int, int] = (6, 1),
        summer_final: Tuple[int, int] = (9, 30),
        energy_weight: float = 0.5,
        max_energy_penalty: float = 8,
        max_comfort_penalty: float = 12,
        lambda_energy: float = 0.1,
        temperature_weight: float = 0.2,
        lambda_temperature: float = 0.1
    ):
        """
        Linear reward function with normalized energy and comfort terms.

        Args:
            temperature_variables (List[str]): Name(s) of the temperature variable(s).
            energy_variables (List[str]): Name(s) of the energy/power variable(s).
            range_comfort_winter (Tuple[int,int]): Temperature comfort range for cold season.
            range_comfort_summer (Tuple[int,int]): Temperature comfort range for hot season.
            summer_start (Tuple[int,int]): Summer session tuple with month and day start. Defaults to (6,1).
            summer_final (Tuple[int,int]): Summer session tuple with month and day end. defaults to (9,30).
            energy_weight (float, optional): Default weight given to the energy term. Defaults to 0.5.
            max_energy_penalty (float, optional): Maximum possible energy penalty. Defaults to 8.
            max_comfort_penalty (float, optional): Maximum possible comfort penalty. Defaults to 12.
        """
        
        # We don't need to pass lambda values to the parent class since we are
        # not using them. The `super()` call is now simplified.
        super().__init__()

        # Basic validations
        if not (0 <= energy_weight <= 1):
            self.logger.error(f'energy_weight must be between 0 and 1. Received: {energy_weight}')
            raise ValueError
        if not all(isinstance(v, str) for v in temperature_variables + energy_variables):
            self.logger.error('All variable names must be strings.')
            raise TypeError

        # Name of the variables
        self.temp_names = temperature_variables
        self.energy_names = energy_variables

        # Reward parameters
        self.range_comfort_winter = range_comfort_winter
        self.range_comfort_summer = range_comfort_summer
        self.W_energy = energy_weight
        
        # These are static, pre-defined maximums for normalization
        self.max_energy_penalty = max_energy_penalty
        self.max_comfort_penalty = max_comfort_penalty

        # Summer period
        self.summer_start = summer_start
        self.summer_final = summer_final
        
        # Initialize instance variables to prevent UnboundLocalError
        self.total_energy = 0
        self.total_temp_violation = 0
        self.energy_penalty = 0
        self.comfort_penalty = 0

        self.logger.info('Reward function initialized.')

    def __call__(self, obs_dict: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate the reward function value based on observation data.

        Args:
            obs_dict (Dict[str, Any]): Dict with observation variable name (key) and observation variable value (value)

        Returns:
            Tuple[float, Dict[str, Any]]: Reward value and dictionary with their individual components.
        """

        # Energy calculation
        energy_values = self._get_energy_consumed(obs_dict)
        self.total_energy = sum(energy_values)
        self.energy_penalty = -self.total_energy

        # Comfort violation calculation
        temp_violations = self._get_temperature_violation(obs_dict)
        self.total_temp_violation = sum(temp_violations)
        self.comfort_penalty = -self.total_temp_violation

        # Weighted sum of both terms
        reward, energy_term, comfort_term = self._get_reward()

        reward_terms = {
            'energy_term': energy_term,
            'comfort_term': comfort_term,
            'energy_penalty': self.energy_penalty,
            'comfort_penalty': self.comfort_penalty,
            'total_power_demand': self.total_energy,
            'total_temperature_violation': self.total_temp_violation,
            'reward_weight': self.W_energy
        }

        return reward, reward_terms

    def _get_energy_consumed(self, obs_dict: Dict[str, Any]) -> List[float]:
        """
        Calculate the energy consumed in the current observation.
        
        Args:
            obs_dict (Dict[str, Any]): Environment observation.
            
        Returns:
            List[float]: List with energy consumed in each energy variable.
        """
        return [obs_dict[v] for v in self.energy_names]

    def _get_temperature_violation(self, obs_dict: Dict[str, Any]) -> List[float]:
        """
        Calculate the temperature violation (ºC) in each observation's temperature variable.

        Args:
            obs_dict (Dict[str, Any]): Environment observation.
            
        Returns:
            List[float]: List with temperature violation in each zone.
        """
        current_dt = datetime(
            YEAR, int(obs_dict['month']), int(obs_dict['day_of_month']))
        summer_start_date = datetime(YEAR, *self.summer_start)
        summer_final_date = datetime(YEAR, *self.summer_final)

        temp_range = self.range_comfort_summer if \
            summer_start_date <= current_dt <= summer_final_date else \
            self.range_comfort_winter

        temp_values = [obs_dict[v] for v in self.temp_names]

        # max(T - T_high, 0) + max(T_low - T, 0)
        return [max(T - temp_range[1], 0) + max(temp_range[0] - T, 0) for T in temp_values]

    def _get_reward(self) -> Tuple[float, ...]:
        """
        Compute the final normalized reward value.

        Returns:
            Tuple[float, ...]: Total reward calculated, reward term for energy, and reward term for comfort.
        """
        # The key is to NOT update the max penalties here.
        # They should be static values defined in __init__.
        
        # Handle the case where a max penalty is zero to avoid division by zero
        energy_norm = self.energy_penalty / self.max_energy_penalty if self.max_energy_penalty != 0 else 0
        comfort_norm = self.comfort_penalty / self.max_comfort_penalty if self.max_comfort_penalty != 0 else 0

        # Calculate reward terms with norm values.
        # Note: the `lambda` terms are no longer needed.
        energy_term = self.W_energy * energy_norm
        comfort_term = (1 - self.W_energy) * comfort_norm
        reward = energy_term + comfort_term
        
        return reward, energy_term, comfort_term


class MultiZoneReward(BaseReward):

    def __init__(
        self,
        energy_variables: List[str],
        temperature_and_setpoints_conf: Dict[str, str],
        comfort_threshold: float = 0.5,
        energy_weight: float = 0.5,
        lambda_energy: float = 1.0,
        lambda_temperature: float = 1.0,
    ):
        """
        A linear reward function for environments with different comfort ranges in each zone. Instead of having
        a fixed and common comfort range for the entire building, each zone has its own comfort range, which is
        directly obtained from the setpoints established in the building. This function is designed for buildings
        where temperature setpoints are not controlled directly but rather used as targets to be achieved, while
        other actuators are controlled to reach these setpoints. A setpoint observation variable can be assigned
        per zone if it is available in the specific building. It is also possible to assign the same setpoint
        variable to multiple air temperature zones.

        Args:
            energy_variables (List[str]): Name(s) of the energy/power variable(s).
            temperature_and_setpoints_conf (Dict[str, str]): Dictionary with the temperature variable name (key) and the setpoint variable name (value) of the observation space.
            comfort_threshold (float, optional): Comfort threshold for temperature range (+/-). Defaults to 0.5.
            energy_weight (float, optional): Weight given to the energy term. Defaults to 0.5.
            lambda_energy (float, optional): Constant for removing dimensions from power(1/W). Defaults to 1e-4.
            lambda_temperature (float, optional): Constant for removing dimensions from temperature(1/C). Defaults to 1.0.
        """

        super().__init__()

        # Name of the variables
        self.energy_names = energy_variables
        self.comfort_configuration = temperature_and_setpoints_conf
        self.comfort_threshold = comfort_threshold

        # Reward parameters
        self.W_energy = energy_weight
        self.lambda_energy = lambda_energy
        self.lambda_temp = lambda_temperature
        self.comfort_ranges = {}

        self.logger.info('Reward function initialized.')

    def __call__(self, obs_dict: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """Calculate the reward function value based on observation data.

        Args:
            obs_dict (Dict[str, Any]): Dict with observation variable name (key) and observation variable value (value)

        Returns:
            Tuple[float, Dict[str, Any]]: Reward value and dictionary with their individual components.
        """

        # Energy calculation
        energy_values = self._get_energy_consumed(obs_dict)
        self.total_energy = sum(energy_values)
        self.energy_penalty = -self.total_energy

        # Comfort violation calculation
        temp_violations = self._get_temperature_violation(obs_dict)
        self.total_temp_violation = sum(temp_violations)
        self.comfort_penalty = -self.total_temp_violation

        # Weighted sum of both terms
        reward, energy_term, comfort_term = self._get_reward()

        reward_terms = {
            'energy_term': energy_term,
            'comfort_term': comfort_term,
            'energy_penalty': self.energy_penalty,
            'comfort_penalty': self.comfort_penalty,
            'total_power_demand': self.total_energy,
            'total_temperature_violation': self.total_temp_violation,
            'reward_weight': self.W_energy,
            'comfort_threshold': self.comfort_threshold,
        }

        return reward, reward_terms

    def _get_energy_consumed(self, obs_dict: Dict[str, Any]) -> List[float]:
        """Calculate the energy consumed in the current observation.

        Args:
            obs_dict (Dict[str, Any]): Environment observation.

        Returns:
            List[float]: List with energy consumed in each energy variable.
        """
        return [obs_dict[v] for v in self.energy_names]

    def _get_temperature_violation(self, obs_dict: Dict[str, Any]) -> List[float]:
        """Calculate the total temperature violation (ºC) in the current observation.

        Returns:
           List[float]: List with temperature violation (ºC) in each zone.
        """
        # Calculate current comfort range for each zone
        self._get_comfort_ranges(obs_dict)

        temp_violations = [
            (
                max(0.0, min(abs(T - comfort_range[0]), abs(T - comfort_range[1])))
                if T < comfort_range[0] or T > comfort_range[1]
                else 0.0
            )
            for temp_var, comfort_range in self.comfort_ranges.items()
            if (T := obs_dict[temp_var])
        ]

        return temp_violations

    def _get_comfort_ranges(self, obs_dict: Dict[str, Any]):
        """Calculate the comfort range for each zone in the current observation.

        Returns:
            Dict[str, Tuple[float, float]]: Comfort range for each zone.
        """
        # Calculate current comfort range for each zone
        self.comfort_ranges = {
            temp_var: (
                setpoint - self.comfort_threshold,
                setpoint + self.comfort_threshold,
            )
            for temp_var, setpoint_var in self.comfort_configuration.items()
            if (setpoint := obs_dict[setpoint_var]) is not None
        }

    def _get_reward(self) -> Tuple[float, ...]:
        """Compute the final reward value.

        Args:
            energy_penalty (float): Negative absolute energy penalty value.
            comfort_penalty (float): Negative absolute comfort penalty value.

        Returns:
            Tuple[float, ...]: Total reward calculated and reward terms.
        """
        energy_term = self.lambda_energy * self.W_energy * self.energy_penalty
        comfort_term = self.lambda_temp * (1 - self.W_energy) * self.comfort_penalty
        reward = energy_term + comfort_term
        return reward, energy_term, comfort_term

class FixedNormalizedLinearReward(NormalizedLinearReward):
    def _get_reward(self) -> Tuple[float, float, float]:
        # update raw penalties (these are negative if you currently set them that way)
        raw_energy_pen  = self.energy_penalty
        raw_comfort_pen = self.comfort_penalty

        # work with positive magnitudes
        pos_energy_pen  = - raw_energy_pen
        pos_comfort_pen = - raw_comfort_pen

        # update running maxima on the positive values
        self.max_energy_penalty  = max(self.max_energy_penalty,  pos_energy_pen)
        self.max_comfort_penalty = max(self.max_comfort_penalty, pos_comfort_pen)

        # now normalize into [0,1]
        energy_norm  = pos_energy_pen  / self.max_energy_penalty  if self.max_energy_penalty  else 0.0
        comfort_norm = pos_comfort_pen / self.max_comfort_penalty if self.max_comfort_penalty else 0.0

        # combine into a raw score in [0,1]
        energy_term  = self.W_energy * energy_norm
        comfort_term = (1 - self.W_energy) * comfort_norm
        r_raw        = energy_term + comfort_term

        reward_0to1 = (2 * r_raw) - 1
        return reward_0to1, energy_term, comfort_term


# pip install river
from river import stats
EPS = 1e-8  # guard against divide‑by‑zero


class P2NormalizedLinearReward(LinearReward):
    """
    LinearReward ➜ percentile‑normalised to [-1, 1] using Jain‑&‑Chlamtac P².

    Parameters
    ----------
    alpha : float, default 0.05
        Lower/upper tail fraction used as clipping bounds (5 % / 95 % if 0.05).
    warmup : int, default 5
        Minimum number of samples before the bounds are trusted.
        During warm‑up, raw rewards are passed through unchanged.
    """

    def __init__(
        self,
        temperature_variables: List[str],
        energy_variables: List[str],
        range_comfort_winter: Tuple[int, int],
        range_comfort_summer: Tuple[int, int],
        summer_start: Tuple[int, int] = (6, 1),
        summer_final: Tuple[int, int] = (9, 30),
        energy_weight: float = 0.5,
        lambda_energy: float = 1.0,
        lambda_temperature: float = 1.0,
        alpha: float = 0.05,
        warmup: int = 0,
    ):
        super().__init__(
            temperature_variables,
            energy_variables,
            range_comfort_winter,
            range_comfort_summer,
            summer_start,
            summer_final,
            energy_weight,
            lambda_energy,
            lambda_temperature,
        )

        self.alpha = alpha
        self.warmup = warmup
        self._init_quantiles()

        # running bounds (start permissive)
        self.L, self.H = -1.0, +1.0
        self._n_seen = 0

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _init_quantiles(self):
        """Create fresh P² sketches."""
        self.q_low = stats.Quantile(self.alpha)           # e.g. 5‑th pct
        self.q_high = stats.Quantile(1.0 - self.alpha)    # e.g. 95‑th pct

    def reset(self):
        """Call this inside env.reset()."""
        self._init_quantiles()
        self.L, self.H = -1.0, +1.0
        self._n_seen = 0

    # ------------------------------------------------------------------
    # main callable
    # ------------------------------------------------------------------
    def __call__(self, obs_dict: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        # 1) underlying linear reward
        raw_r, terms = super().__call__(obs_dict)

        # 2) stream raw reward into P² sketches
        self.q_low.update(raw_r)
        self.q_high.update(raw_r)
        self._n_seen += 1

        # 3) update clipping window after warm‑up
        if self._n_seen >= self.warmup:
            Lcand = self.q_low.get()
            Hcand = self.q_high.get()
            if Hcand is not None and Lcand is not None and Hcand > Lcand + EPS:
                self.L, self.H = Lcand, Hcand

        # 4) linear normalisation ➜ [0, 1]
        r0 = (raw_r - self.L) / (self.H - self.L + EPS)

        # 5) squash/clip ➜ [-1, 1]
        norm_r = 2.0 * np.clip(r0, 0.0, 1.0) - 1.0

        # 6) diagnostics for tensorboard or CSV logs
        terms.update(
            {
                "raw_reward": raw_r,
                "pct_low": self.L,
                "pct_high": self.H,
                "normalized_reward": norm_r,
            }
        )
        return float(norm_r), terms

"""Implementation of a comfort-gated reward function for HVAC control.

This module defines a reward function based on a comfort-gated paradigm.
The total reward is a combination of a piecewise-linear temperature reward
and a normalized energy reward. The energy reward is only active when
the temperature is within a comfortable range, ensuring comfort is the
highest priority.
"""

from typing import Any, Dict, List, Tuple
from datetime import date

import numpy as np


class BaseReward(object):
    """
    Base reward class for inheritance.
    
    All reward functions should inherit from this class to ensure a consistent
    interface. It raises a NotImplementedError if the `__call__` method is not
    implemented.
    """
    def __init__(self):
        pass

    def __call__(self, obs_dict: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """Method for calculating the reward function."""
        raise NotImplementedError("Reward class must have a `__call__` method.")


"""Implementation of a comfort-gated reward function for HVAC control.

This module defines a reward function based on a comfort-gated paradigm.
The total reward is a combination of a piecewise-linear temperature reward
and a normalized energy reward. The energy reward is only active when
the temperature is within a comfortable range, ensuring comfort is the
highest priority.
"""

from typing import Any, Dict, List, Tuple
from datetime import date

import numpy as np


class BaseReward(object):
    """
    Base reward class for inheritance.
    
    All reward functions should inherit from this class to ensure a consistent
    interface. It raises a NotImplementedError if the `__call__` method is not
    implemented.
    """
    def __init__(self):
        pass

    def __call__(self, obs_dict: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """Method for calculating the reward function."""
        raise NotImplementedError("Reward class must have a `__call__` method.")


from datetime import date
from typing import Any, Dict, List, Tuple
import numpy as np
import math

class ComfortGatedReward(BaseReward):
    """
    Comfort-gated reward (Formulation B with λ_E):

      R_t = r_temp  +  λ_E * w_comfort * r_energy

    - r_temp(τ)  : two-stage, saturating piecewise reward in [-M, M], then normalized to [-1,1]
    - r_energy   : adaptive per-difficulty-bin (EMA μ/σ, ±kσ band) → [-1,1], scaled by mildness
    - w_comfort  : (1 + r_temp)/2 ∈ [0,1] — energy only counts when comfort is good

    Parameters (key ones you asked for):
      M, eta, delta(Δ), K     : temperature reward shape (see doc)
      lambda_energy (λ_E)     : energy weight once comfort is good
      c, p                    : mildness weight params

\\    Other parameters:
      alpha                   : EMA smoothing for energy stats (default 0.02)
      k                       : tolerance half-width in σ for energy normalization (default 2.0)
      difficulty_bins         : difficulty d (°C away from [τ_min, τ_max]) bin edges (default (0,2,5,10))
      epsilon                 : small number to avoid division by zero
    """

    def __init__(
        self,
        temperature_variables: List[str],
        energy_variables: List[str],
        range_comfort_winter: Tuple[float, float],
        range_comfort_summer: Tuple[float, float],
        summer_start: Tuple[int, int] = (6, 1),
        summer_final: Tuple[int, int] = (9, 30),
        
        # ---- Temperature reward params ----
        M: float = 1.0,
        eta: float = 0.2,
        delta: float = 2.0,   # Δ in the doc
        K: float = 5.0,
        
        # ---- Combination (doc Formulation B) ----
        lambda_energy: float = 0.5,  # λ_E
        
        # ---- Mildness weight (doc Eq. 7) ----
        c: float = 10.0,
        p: float = 2.0,
        
        # ---- Adaptive energy normalization (doc Sec. 2.2–2.5) ----
        alpha: float = 0.02,
        k: float = 2.0,
        difficulty_bins: Tuple[float, ...] = (0.0, 2.0, 5.0, 10.0),
        epsilon: float = 1e-9,
        gate_theta: float = 0.0,
        gate_k: float = 4.0,
        
        # ---- Summary penalty params ----
        window_steps: int = 96,           # daily summary window
        lambda_dev: float = 0.05,        # penalty for comfort band deviation window sum
    ):
        super().__init__()

        # Store variables/comfort bands/season
        self.temp_names = list(temperature_variables)
        self.energy_names = list(energy_variables)
        self.range_comfort_winter = tuple(range_comfort_winter)
        self.range_comfort_summer = tuple(range_comfort_summer)
        self.summer_start = date(1, int(summer_start[0]), int(summer_start[1]))
        self.summer_final = date(1, int(summer_final[0]), int(summer_final[1]))

        # Validate & store temperature reward params
        if not (M > 0 and delta > 0 and K > 0):
            raise ValueError("M, delta and K must be > 0")
        if not (0.0 < eta < 1.0):
            raise ValueError("eta must be in (0,1)")

        self.M = float(M)
        self.eta = float(eta)
        self.delta = float(delta)
        self.K = float(K)

        # Precompute slopes and inner comfort sub-bands (summer/winter)
        self.s_L = (self.eta * self.M) / self.delta 
        self.s_R = (self.eta * self.M) / self.delta 
        self.S   = (self.M * (2.0 - self.eta)) / self.K

        self.a_summer = self.range_comfort_summer[0] + self.delta
        self.b_summer = self.range_comfort_summer[1] - self.delta
        self.a_winter = self.range_comfort_winter[0] + self.delta
        self.b_winter = self.range_comfort_winter[1] - self.delta

        # Combination weight
        self.lambda_energy = float(lambda_energy)

        # Mildness
        if c <= 0:
            raise ValueError("c must be > 0")
        if p <= 0:
            raise ValueError("p must be > 0")
        self.c = float(c)
        self.p = float(p)

        # Adaptive energy stats
        if alpha <= 0 or alpha >= 1:
            raise ValueError("alpha must be in (0,1)")
        if k <= 0:
            raise ValueError("k must be > 0")
        self.alpha = float(alpha)
        self.k = float(k)
        self.eps = float(epsilon)

        # Difficulty bins (ensure sorted and starts at 0)
        edges = sorted(difficulty_bins)
        if edges[0] != 0.0:
            edges = [0.0] + edges
        self.bin_edges = tuple(edges)
        self.num_bins = len(self.bin_edges)

        # Per-bin EMA: μ, variance (EMA of squared deviation), σ
        self.mu_bins    = [0.0] * self.num_bins
        self.var_bins   = [0.0] * self.num_bins
        self.sigma_bins = [0.0] * self.num_bins
        self._bin_init  = [False] * self.num_bins  # lazy init with first sample
        self.gate_theta = gate_theta
        self.gate_k =  gate_k

        #  summary penality extention
        self.window_steps = int(window_steps)
        self.lambda_dev = float(lambda_dev)

        self.dev_accum = 0.0
        self.step_counter = 0
    # ---------------- Helpers: season & comfort band ----------------

    def _is_summer(self, obs: Dict[str, Any]) -> bool:
        d = date(1, int(obs['month']), int(obs['day_of_month']))
        return self.summer_start <= d <= self.summer_final

    def _comfort_band(self, is_summer: bool) -> Tuple[float, float, float, float]:
        if is_summer:
            return (self.range_comfort_summer[0], self.range_comfort_summer[1],
                    self.a_summer, self.b_summer)
        else:
            return (self.range_comfort_winter[0], self.range_comfort_winter[1],
                    self.a_winter, self.b_winter)

    # ---------------- Temperature reward (piecewise, saturating) ----------------

    def _r_temp_single(self, tau: float, tau_min: float, tau_max: float, a: float, b: float) -> float:
        M, S, sL, sR, K = self.M, self.S, self.s_L, self.s_R, self.K

        if tau <= tau_min - K:
            return -M 
        elif tau_min - K < tau < tau_min:
            return M * (1.0 - self.eta) - S * (tau_min - tau)
        elif tau_min <= tau <= a:
            return M - sL * (a - tau)
        elif a < tau < b:
            return M
        elif b <= tau <= tau_max:
            return M - sR * (tau - b)
        elif tau_max < tau < tau_max + K:
            return M * (1.0 - self.eta) - S * (tau - tau_max)
        else:
            return -M

    # ---------------- Difficulty and bins ----------------

    @staticmethod
    def _difficulty(Tout: float, tau_min: float, tau_max: float) -> float:
        if Tout < tau_min:
            return tau_min - Tout
        if Tout > tau_max:
            return Tout - tau_max
        return 0.0

    def _bin_index(self, d: float) -> int:
        # Find i s.t. bin_edges[i] <= d < bin_edges[i+1], last bin is [last, ∞)
        for i in range(self.num_bins - 1):
            if d < self.bin_edges[i + 1]:
                return i
        return self.num_bins - 1

    # ---------------- EMA updates per bin ----------------

    def _update_bin_stats(self, idx: int, E_t: float) -> Tuple[float, float]:
        """Update EMA μ and variance for bin idx; return (mu, sigma)."""
        if not self._bin_init[idx]:
            self.mu_bins[idx] = E_t
            self.var_bins[idx] = 0.0
            self.sigma_bins[idx] = 0.0
            self._bin_init[idx] = True
            return self.mu_bins[idx], self.sigma_bins[idx]

        mu_old = self.mu_bins[idx]
        v_old  = self.var_bins[idx]

        mu = (1.0 - self.alpha) * mu_old + self.alpha * E_t
        v  = (1.0 - self.alpha) * v_old + self.alpha * (E_t - mu) ** 2
        sigma = math.sqrt(v + self.eps)

        self.mu_bins[idx] = mu
        self.var_bins[idx] = v
        self.sigma_bins[idx] = sigma
        return mu, sigma

    # ---------------- Energy reward (adaptive, bounded) ----------------

    def _r_energy(self, E_t: float, d: float, bin_idx: int) -> float:
        # Update bin stats and compute normalized score via ±kσ window
        mu_b, sigma_b = self._update_bin_stats(bin_idx, E_t)

        if sigma_b <= self.eps:
            u_t = 1.0 if E_t <= mu_b else 0.0
        else:
            u_t = (mu_b + self.k * sigma_b - E_t) / (2.0 * self.k * sigma_b + self.eps)
            u_t = float(np.clip(u_t, 0.0, 1.0))

        r_raw = 2.0 * u_t - 1.0  # in [-1, 1]

        # Mildness weight (doc Eq. 7)
        w_mild = 1.0 / (1.0 + (d / self.c) ** self.p)
        return w_mild * r_raw

    # ---------------- Main API ----------------

    def __call__(self, obs_dict: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        # Determine season and comfort band
        summer = self._is_summer(obs_dict)
        tau_min, tau_max, a, b = self._comfort_band(summer)

        # r_temp: average over provided zone temperatures, then normalize to [-1,1]
        temps = [float(obs_dict[name]) for name in self.temp_names]
        r_temps = [self._r_temp_single(t, tau_min, tau_max, a, b) for t in temps]
        r_temp_raw = float(np.mean(r_temps)) if r_temps else 0.0
        r_temp = float(np.clip(r_temp_raw / self.M, -1.0, 1.0))  # normalize by M

        
        # Outdoor difficulty and bin
        Tout = float(obs_dict.get('outdoor_temperature',
                       obs_dict.get('outdoor_temp',
                       obs_dict.get('Tout', (tau_min + tau_max) / 2.0))))
        d = self._difficulty(Tout, tau_min, tau_max)
        bin_idx = self._bin_index(d)

        # Aggregate energy/power across listed variables
        E_t = float(np.sum([float(obs_dict[n]) for n in self.energy_names])) if self.energy_names else 0.0

        # r_energy (adaptive, then mildness)
        r_energy = self._r_energy(E_t, d, bin_idx)

        # Comfort gate and final doc formulation
        # w_comfort = (1.0 + r_temp) / 2.0              # ∈ [0,1]
        x = self.gate_k * (r_temp - self.gate_theta)
        w_comfort = float(1.0 / (1.0 + np.exp(-x))) 
        # R = r_temp + self.lambda_energy * w_comfort * r_energy
        R = r_temp
        R = float(np.clip(R, -1.0, 1.0))              # safety clip
        
        # ---------------- SUMMARY PENALTY BLOCK ----------------
        indoor_temp = float(obs_dict['indoor_temperature'])

        # compute overcool and overheat degrees
        overcool = max(0.0, tau_min - indoor_temp)
        overheat = max(0.0, indoor_temp - tau_max)

        deviation = overcool + overheat
        self.dev_accum += deviation
        self.step_counter += 1

        # # # apply penalties every window
        # if self.step_counter >= self.window_steps:
        #     R += -self.lambda_dev *  self.dev_accum

        #     # reset accumulators
        #     self.dev_accum = 0.0
        #     self.step_counter = 0

        w_mild = 1.0 / (1.0 + (d / self.c) ** self.p)
        # Helpful terms for logging/analysis
        terms = {
            'reward': R,
            'r_temp': r_temp,
            'r_temp_raw': r_temp_raw,
            'r_energy': r_energy,
            'lambda_energy': self.lambda_energy,
            'w_comfort': w_comfort,
            'difficulty_d': d,
            'difficulty_bin': bin_idx,
            'temp_avg': float(np.mean(temps)) if temps else None,
            'tau_min': tau_min,
            'tau_max': tau_max,
            'a': a,
            'b': b,
            'E_t_total': E_t,
        }
        #### DEBUG ####
        # DEBUG: collect full trace
        debug_trace = {
            "step": obs_dict.get("timestep", "N/A"),
            "indoor_temp": float(obs_dict["indoor_temperature"]),
            "outdoor_temp": Tout,
            "E_t": E_t,
            "r_temp_raw": r_temp_raw,
            "r_temp": r_temp,
            "difficulty_d": d,
            "bin_idx": bin_idx,
            "mu_bin": self.mu_bins[bin_idx],
            "sigma_bin": self.sigma_bins[bin_idx],
            "r_raw": r_energy / w_mild if w_mild > 1e-6 else None,
            "w_mild": w_mild ,          # <-- ADD THIS
            "w_comfort": w_comfort,
            "r_energy_final": r_energy,
            "lambda_energy": self.lambda_energy,
            "final_reward": R,
        }
        # SAVE to CSV after each step
        with open("trace_debug_lamOne.csv", "a") as f:
            f.write(str(debug_trace) + "\n")
        return R, terms

class L1Reward(BaseReward):
    """
    Simple L1 temperature reward:
        r_temp = -|T - target| / M      → clipped to [-1, 0]

    - Only uses indoor temperature.
    - No energy dependency.
    - No normalization.
    - No seasonal logic.
    - No gating / comfort weights.

    Purpose:
    --------
    *Baseline PPO check*: verify that PPO can track a single scalar target.
    """

    def __init__(self,
                 temperature_variables: List[str],
                 target: float = 24.0,
                 M: float = 1.0):
        super().__init__()
        if M <= 0:
            raise ValueError("M must be > 0")

        self.temp_names = list(temperature_variables)
        self.target = float(target)
        self.M = float(M)

    def __call__(self, obs_dict: Dict[str, Any]):
        # Extract indoor temperatures
        temps = [float(obs_dict[name]) for name in self.temp_names]
        temp_avg = float(np.mean(temps)) if temps else 0.0

        # Compute raw L1 penalty: -(distance_to_target)
        r_temp_raw = -(abs(temp_avg - self.target))

        # Normalize just by M (scale), clip to [-1, 0]
        r_temp = float(np.clip(r_temp_raw / self.M, -1.0, 0.0))

        # Final reward = r_temp only
        R = r_temp

        # Logging info
        terms = {
            "reward": R,
            "r_temp": r_temp,
            "r_temp_raw": r_temp_raw,
            "temp_avg": temp_avg,
            "target": self.target
        }

        #### OPTIONAL CSV DEBUG ####
        debug_trace = {
            "step": obs_dict.get("timestep", "N/A"),
            "indoor_temp": temp_avg,
            "r_temp_raw": r_temp_raw,
            "r_temp": r_temp,
            "final_reward": R
        }
        with open("trace_l1reward.csv", "a") as f:
            f.write(str(debug_trace) + "\n")

        return R, terms

class ConvexMixtureReward(BaseReward):
    """
    Convex Mixture (Formulation A):
        R_t = w * r_temp + (1 - w) * r_energy

    r_temp  : two-stage, saturating piecewise reward in [-M, M], then normalized to [-1,1]
    r_energy: adaptive per-difficulty-bin (EMA μ/σ, ±kσ) → [-1,1], scaled by mildness
    """

    def __init__(
        self,
        temperature_variables: List[str],
        energy_variables: List[str],
        range_comfort_winter: Tuple[float, float],
        range_comfort_summer: Tuple[float, float],
        summer_start: Tuple[int, int] = (6, 1),
        summer_final: Tuple[int, int] = (9, 30),
        # --- Temperature reward params ---
        M: float = 1.0,
        eta: float = 0.2,
        delta: float = 2.0,   # inner comfort margin Δ
        K: float = 5.0,       # outside saturation half-width
        # --- Mixture weight ---
        w: float = 0.7,       # 70% comfort, 30% energy (example)
        # --- Mildness (for energy) ---
        c: float = 10.0,
        p: float = 2.0,
        # --- Adaptive energy normalization ---
        alpha: float = 0.02,          # EMA smoothing
        k: float = 2.0,               # ±kσ tolerance
        difficulty_bins: Tuple[float, ...] = (0.0, 2.0, 5.0, 10.0),
        epsilon: float = 1e-9,
        energy_weight = 1.0,
        lambda_temperature = 0.1,
        lambda_energy = 0.1,
    ):
        super().__init__()

        # Store variables/comfort bands/season
        self.temp_names = list(temperature_variables)
        self.energy_names = list(energy_variables)
        self.range_comfort_winter = tuple(range_comfort_winter)
        self.range_comfort_summer = tuple(range_comfort_summer)
        self.summer_start = date(1, int(summer_start[0]), int(summer_start[1])
                                 )
        self.summer_final = date(1, int(summer_final[0]), int(summer_final[1])
                                 )

        # Validate mixture weight
        if not (0.0 <= w <= 1.0):
            raise ValueError("w must be in [0,1]")
        self.w = float(w)

        # Validate & store temp reward params
        if not (M > 0 and delta > 0 and K > 0):
            raise ValueError("M, delta and K must be > 0")
        if not (0.0 < eta < 1.0):
            raise ValueError("eta must be in (0,1)")

        self.M = float(M)
        self.eta = float(eta)
        self.delta = float(delta)
        self.K = float(K)

        # Precompute slopes and inner bands
        self.s_L = (self.eta * self.M) / self.delta
        self.s_R = (self.eta * self.M) / self.delta
        self.S   = (self.M * (2.0 - self.eta)) / self.K

        self.a_summer = self.range_comfort_summer[0] + self.delta
        self.b_summer = self.range_comfort_summer[1] - self.delta
        self.a_winter = self.range_comfort_winter[0] + self.delta
        self.b_winter = self.range_comfort_winter[1] - self.delta

        # Mildness (for energy)
        if c <= 0 or p <= 0:
            raise ValueError("c and p must be > 0")
        self.c = float(c)
        self.p = float(p)

        # Adaptive energy params
        if alpha <= 0 or alpha >= 1:
            raise ValueError("alpha must be in (0,1)")
        if k <= 0:
            raise ValueError("k must be > 0")
        self.alpha = float(alpha)
        self.k = float(k)
        self.eps = float(epsilon)

        # Difficulty bins (ensure sorted and starts at 0)
        edges = sorted(difficulty_bins)
        if edges[0] != 0.0:
            edges = [0.0] + edges
        self.bin_edges = tuple(edges)
        self.num_bins = len(self.bin_edges)

        # Per-bin EMA: μ, variance (EMA), σ
        self.mu_bins    = [0.0] * self.num_bins
        self.var_bins   = [0.0] * self.num_bins
        self.sigma_bins = [0.0] * self.num_bins
        self._bin_init  = [False] * self.num_bins  # lazy init

    # ---------- helpers: season & band ----------
    def _is_summer(self, obs: Dict[str, Any]) -> bool:
        d = date(1, int(obs['month']), int(obs['day_of_month']))
        return self.summer_start <= d <= self.summer_final

    def _comfort_band(self, is_summer: bool) -> Tuple[float, float, float, float]:
        if is_summer:
            return (self.range_comfort_summer[0], self.range_comfort_summer[1],
                    self.a_summer, self.b_summer)
        else:
            return (self.range_comfort_winter[0], self.range_comfort_winter[1],
                    self.a_winter, self.b_winter)

    # ---------- r_temp: two-stage, saturating ----------
    def _r_temp_single(self, tau: float, tau_min: float, tau_max: float, a: float, b: float) -> float:
        M, S, sL, sR, K = self.M, self.S, self.s_L, self.s_R, self.K

        if tau <= tau_min - K:
            return -M
        elif tau_min - K < tau < tau_min:
            return M * (1.0 - self.eta) - S * (tau_min - tau)
        elif tau_min <= tau <= a:
            return M - sL * (a - tau)
        elif a < tau < b:
            return M
        elif b <= tau <= tau_max:
            return M - sR * (tau - b)
        elif tau_max < tau < tau_max + K:
            return M * (1.0 - self.eta) - S * (tau - tau_max)
        else:
            return -M

    # ---------- difficulty & bins ----------
    @staticmethod
    def _difficulty(Tout: float, tau_min: float, tau_max: float) -> float:
        if Tout < tau_min:
            return tau_min - Tout
        if Tout > tau_max:
            return Tout - tau_max
        return 0.0

    def _bin_index(self, d: float) -> int:
        for i in range(self.num_bins - 1):
            if d < self.bin_edges[i + 1]:
                return i
        return self.num_bins - 1

    # ---------- EMA updates ----------
    def _update_bin_stats(self, idx: int, E_t: float) -> Tuple[float, float]:
        if not self._bin_init[idx]:
            self.mu_bins[idx] = E_t
            self.var_bins[idx] = 0.0
            self.sigma_bins[idx] = 0.0
            self._bin_init[idx] = True
            return self.mu_bins[idx], self.sigma_bins[idx]

        mu_old = self.mu_bins[idx]
        v_old  = self.var_bins[idx]

        mu = (1.0 - self.alpha) * mu_old + self.alpha * E_t
        v  = (1.0 - self.alpha) * v_old + self.alpha * (E_t - mu) ** 2
        sigma = math.sqrt(v + self.eps)

        self.mu_bins[idx] = mu
        self.var_bins[idx] = v
        self.sigma_bins[idx] = sigma
        return mu, sigma

    # ---------- r_energy: adaptive, bounded ----------
    def _r_energy(self, E_t: float, d: float, bin_idx: int) -> float:
        mu_b, sigma_b = self._update_bin_stats(bin_idx, E_t)
        if sigma_b <= self.eps:
            u_t = 1.0 if E_t <= mu_b else 0.0
        else:
            u_t = (mu_b + self.k * sigma_b - E_t) / (2.0 * self.k * sigma_b + self.eps)
            u_t = float(np.clip(u_t, 0.0, 1.0))
        r_raw = 2.0 * u_t - 1.0  # [-1,1]
        w_mild = 1.0 / (1.0 + (d / self.c) ** self.p)
        return w_mild * r_raw

    # ---------- main ----------
    def __call__(self, obs_dict: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        # Season & comfort band
        summer = self._is_summer(obs_dict)
        tau_min, tau_max, a, b = self._comfort_band(summer)

        # r_temp over listed zones, then normalize to [-1,1]
        temps = [float(obs_dict[name]) for name in self.temp_names]
        r_temps = [self._r_temp_single(t, tau_min, tau_max, a, b) for t in temps]
        r_temp_raw = float(np.mean(r_temps)) if r_temps else 0.0
        r_temp = float(np.clip(r_temp_raw / self.M, -1.0, 1.0))

        # Outdoor difficulty & bin
        Tout = float(obs_dict.get('outdoor_temperature',
                       obs_dict.get('outdoor_temp',
                       obs_dict.get('Tout', (tau_min + tau_max) / 2.0))))
        d = self._difficulty(Tout, tau_min, tau_max)
        bin_idx = self._bin_index(d)

        # Aggregate energy/power across provided vars
        E_t = float(np.sum([float(obs_dict[n]) for n in self.energy_names])) if self.energy_names else 0.0

        # r_energy
        r_energy = self._r_energy(E_t, d, bin_idx)

        # Convex mixture
        R = self.w * r_temp + (1.0 - self.w) * r_energy
        R = float(np.clip(R, -1.0, 1.0))

        info = {
            'reward': R,
            'r_temp': r_temp,
            'r_temp_raw': r_temp_raw,
            'r_energy': r_energy,
            'w_mixture': self.w,
            'difficulty_d': d,
            'difficulty_bin': bin_idx,
            'temp_avg': float(np.mean(temps)) if temps else None,
            'tau_min': tau_min,
            'tau_max': tau_max,
            'a': a,
            'b': b,
            'E_t_total': E_t,
        }
        return R, info

from typing import Any, Dict, List, Tuple
from datetime import datetime
import numpy as np

from sinergym.utils.constants import LOG_REWARD_LEVEL, YEAR
from sinergym.utils.logger import TerminalLogger



# rewards_myopic.py
from typing import Dict, Any, List, Tuple, Optional
import numpy as np

class MyopicNearestSetpointCompat:
    """
    Dual-setpoint, myopic temperature reward (interface-compatible).

    Reward (temperature-only):
        r = - lambda_temperature * mean_i | T_zone[i] - T_set*(nearest) |

    Notes
    -----
    - Constructor keeps the SAME parameters as your other reward classes:
      temperature_variables, energy_variables, range_comfort_winter/summer,
      summer_start/final, energy_weight, lambda_energy, lambda_temperature.
      (Most are kept for compatibility and logging; only lambda_temperature
      affects the scalar reward here.)
    - Energy is summed and returned in info (E_t), but NOT used in the reward.
    - Assumes global setpoint names exist in obs_dict: 'htg_setpoint', 'clg_setpoint'.
      You can change these two attribute names after construction if needed.

    Returns
    -------
    (reward, info_dict)
      where info_dict includes comfort/energy terms for consistency with your pipeline.
    """

    def __init__(
        self,
        temperature_variables: List[str],
        energy_variables: List[str],
        range_comfort_winter: Tuple[int, int],
        range_comfort_summer: Tuple[int, int],
        summer_start: Tuple[int, int] = (6, 1),
        summer_final: Tuple[int, int] = (9, 30),
        energy_weight: float = 0.5,
        lambda_energy: float = 1.0,
        lambda_temperature: float = 1.0,
        # ---- optional extras for convenience (do not break signature) ----
        setpoint_heat_name: str = "htg_setpoint",
        setpoint_cool_name: str = "clg_setpoint",
        aggregate: str = "mean",  # or "sum"
    ):
        # Keep exact API
        self.temp_names = list(temperature_variables)
        self.energy_names = list(energy_variables)
        self.range_comfort_winter = tuple(range_comfort_winter)
        self.range_comfort_summer = tuple(range_comfort_summer)
        self.summer_start = tuple(summer_start)
        self.summer_final = tuple(summer_final)
        self.W_energy = float(energy_weight)
        self.lambda_energy = float(lambda_energy)
        self.lambda_temp = float(lambda_temperature)

        # Extras (not part of the original API, but harmless defaults)
        self.htg_name = setpoint_heat_name
        self.clg_name = setpoint_cool_name
        self.aggregate = aggregate.lower()
        self.logger = None  # your framework may inject

    def __call__(self, obs_dict: Dict[str, Any]):
        # --- read setpoints (global dual-SP) ---
        try:
            Th = float(obs_dict[self.htg_name])
            Tc = float(obs_dict[self.clg_name])
        except KeyError as e:
            raise KeyError(
                f"Missing setpoint in obs_dict: {e}. "
                f"Expected '{self.htg_name}' and '{self.clg_name}'."
            )

        # --- myopic temperature loss across selected zones ---
        errs = []
        for name in self.temp_names:
            Tz = float(obs_dict[name])
            Tset = Tc if abs(Tz - Tc) <= abs(Tz - Th) else Th
            errs.append(abs(Tz - Tset))

        if self.aggregate == "sum":
            temp_err = float(np.sum(errs))
        else:
            temp_err = float(np.mean(errs))  # default

        # Comfort penalty (negative); reward is negative loss
        comfort_penalty = -temp_err
        comfort_term = self.lambda_temp * comfort_penalty  # this IS the reward (myopic)

        # --- energy diagnostics (not used in reward) ---
        E_t = float(np.sum([float(obs_dict[n]) for n in self.energy_names])) if self.energy_names else 0.0
        energy_penalty = -E_t
        energy_term = 0.0  # myopic: energy does not influence reward

        reward = float(comfort_term)  # temperature-only

        # --- info terms for compatibility with your logging pipeline ---
        info = {
            "energy_term": energy_term,
            "comfort_term": comfort_term,
            "energy_penalty": energy_penalty,
            "comfort_penalty": comfort_penalty,
            "total_power_demand": E_t,
            "mean_abs_temp_error": temp_err,
            "reward_weight": self.W_energy,          # kept for compatibility
            "setpoint_heat": Th,
            "setpoint_cool": Tc,
            "reward": reward,
        }
        return reward, info

