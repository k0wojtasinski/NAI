# Fuzzy Logic Washing Machine Time to do the laundry
# Author: Kacper Wojtasiński
# Inspiration: https://pythonhosted.org/scikit-fuzzy/auto_examples/plot_tipping_problem_newapi.html

import json
import logging
from typing import Dict, List

import click
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from skfuzzy.control.rule import Rule

logging.basicConfig(format="%(asctime)s | %(levelname)s | %(message)s", level="INFO")

logger = logging.getLogger(__name__)


class WashingMachineFuzzyLogicSimulation:
    MIN_TEMPERATURE_VALUE = 20
    MAX_TEMPERATURE_VALUE = 60
    MIN_WEIGHT_VALUE = 1000
    MAX_WEIGHT_VALUE = 5000
    MIN_DIRTINESS_VALUE = 1
    MAX_DIRTINESS_VALUE = 100

    @classmethod
    def import_from_file(cls, path: str) -> "WashingMachineFuzzyLogicSimulation":
        """class method to create simulation from given JSON file
        
        Args:
            path (str): path to JSON file
        Returns:
            [type]: [description]
        """
        logger.info(f'Importing WashingMachineFuzzyLogicSimulation from path "{path}"')
        with open(path) as _file:
            data = json.load(_file)
            return cls(**data)

    @staticmethod
    def prepare_arguments(
        temperature: int, weight: int, dirtiness: int
    ) -> Dict[str, int]:
        """static method to prepare and validate given arguments (inputs)

        Args:
            temperature (int): temperature of laundry
            weight (int): weight of laundry
            dirtiness (int): dirtiness of laundry

        Raises:
            ValueError: when any of the arguments does not satisfy limits defined in static fields (MIN, MAX)

        Returns:
            Dict[str, int]: dictionary with arguments to be used
        """
        logger.info(
            f"Getting arguments (temperature={temperature}, weight={weight}, dirtiness={dirtiness}) and validation"
        )
        if (
            temperature < WashingMachineFuzzyLogicSimulation.MIN_TEMPERATURE_VALUE
            or temperature > WashingMachineFuzzyLogicSimulation.MAX_WEIGHT_VALUE
        ):
            raise ValueError(f"{temperature} is incorrect temperature")
        if (
            weight < WashingMachineFuzzyLogicSimulation.MIN_WEIGHT_VALUE
            or weight > WashingMachineFuzzyLogicSimulation.MAX_WEIGHT_VALUE
        ):
            raise ValueError("weight is incorrect")
        if (
            dirtiness < WashingMachineFuzzyLogicSimulation.MIN_DIRTINESS_VALUE
            or dirtiness > WashingMachineFuzzyLogicSimulation.MAX_DIRTINESS_VALUE
        ):
            raise ValueError("Temperature is incorrect")

        return {
            "temperature": temperature,
            "weight": weight,
            "dirtiness": dirtiness,
        }

    def prepare_antecedents(self) -> Dict[str, ctrl.Antecedent]:
        """method to prepare antecedents based on MIN and MAX values of them (static fields)

        Returns:
            Dict[str, ctrl.Antecedent]: dictionary with antecedents to be used
        """
        logger.info("Preparing antecedents")

        antecedents = {
            "temperature": ctrl.Antecedent(
                np.arange(
                    self.MIN_TEMPERATURE_VALUE, self.MAX_TEMPERATURE_VALUE + 1, 1
                ),
                "temperature",
            ),
            "weight": ctrl.Antecedent(
                np.arange(self.MIN_WEIGHT_VALUE, self.MAX_WEIGHT_VALUE + 100, 100),
                "weight",
            ),
            "dirtiness": ctrl.Antecedent(
                np.arange(self.MIN_DIRTINESS_VALUE, self.MAX_DIRTINESS_VALUE + 1, 1),
                "dirtiness",
            ),
        }

        for antecedent in antecedents.values():
            antecedent.automf(3)

        return antecedents

    @staticmethod
    def prepare_consequent() -> ctrl.Consequent:
        """static method to prepare consequent

        Returns:
            ctrl.Consequent: consequent to be used
        """
        logger.info("Preparing consequent")

        consequent = ctrl.Consequent(np.arange(1, 6, 1), "time")
        consequent["short"] = fuzz.trimf(consequent.universe, [0, 2, 3])
        consequent["medium"] = fuzz.trimf(consequent.universe, [0, 3, 5])
        consequent["long"] = fuzz.trimf(consequent.universe, [3, 5, 5])

        return consequent

    def prepare_rules(self) -> List[ctrl.Rule]:
        """method to prepare rules based on prepared antecedents

        Returns:
            List[ctrl.Rule]: rules to be applied
        """
        logger.info("Preparing rules")

        temperature = self.antecedents.get("temperature")
        weight = self.antecedents.get("weight")
        dirtiness = self.antecedents.get("dirtiness")
        time = self.consequent

        return (
            ctrl.Rule(dirtiness["good"], time["long"]),
            ctrl.Rule(weight["poor"] & dirtiness["poor"], time["short"]),
            ctrl.Rule(
                temperature["poor"] & weight["poor"] & dirtiness["poor"],
                time["short"],
            ),
            ctrl.Rule(
                temperature["average"] & weight["average"] & dirtiness["average"],
                time["medium"],
            ),
            ctrl.Rule(
                temperature["good"] & weight["average"] & dirtiness["poor"],
                time["short"],
            ),
            ctrl.Rule(
                temperature["poor"] & weight["average"] & dirtiness["good"],
                time["long"],
            ),
        )

    def __init__(self, temperature: int, weight: int, dirtiness: int) -> None:
        self.arguments = self.prepare_arguments(temperature, weight, dirtiness)
        self.antecedents = self.prepare_antecedents()
        self.consequent = self.prepare_consequent()
        self.rules = self.prepare_rules()

        self.washing_machine = ctrl.ControlSystemSimulation(
            ctrl.ControlSystem(self.rules)
        )

    def run(self) -> float:
        """method to compute simulation, it should return estimated hours of laundry

        Returns:
            float: hours of laundry
        """
        logger.info(f"Running {self}")

        self.washing_machine.input["temperature"] = self.arguments.get("temperature")
        self.washing_machine.input["weight"] = self.arguments.get("weight")
        self.washing_machine.input["dirtiness"] = self.arguments.get("dirtiness")

        self.washing_machine.compute()

        result = self.washing_machine.output["time"]
        logger.info(f"Result of {self} : {result}")

        return result

    def export_to_file(self, path: str) -> None:
        """method to export simulation to JSON file

        Args:
            path (str): path to the JSON file
        """
        logger.info(f'Exporting WashingMachineFuzzyLogicSimulation to file "{path}"')
        with open(path, "w") as _file:
            json.dump(self.arguments, _file)

    def __str__(self) -> str:
        """ overriden str method """
        return str(f"{self.__class__.__name__} {self.arguments}")


def prepare_input_text(option_name:str, min:int, max:int):
    """helper function to provide input text

    Args:
        option_name (str): name of the option
        min (int): min value of the option
        max (int): max value of the option

    Returns:
        str: text for input
    """
    return f"{option_name} to provide (min={min}, max={max})"


@click.group()
def cli():
    """ click group to create CLI powered by click """
    ...


@cli.command()
@click.option(
    "--temperature",
    prompt=prepare_input_text(
        "Temperature",
        WashingMachineFuzzyLogicSimulation.MIN_TEMPERATURE_VALUE,
        WashingMachineFuzzyLogicSimulation.MAX_TEMPERATURE_VALUE,
    ),
    required=True,
    type=click.IntRange(
        min=WashingMachineFuzzyLogicSimulation.MIN_TEMPERATURE_VALUE,
        max=WashingMachineFuzzyLogicSimulation.MAX_TEMPERATURE_VALUE,
    ),
)
@click.option(
    "--weight",
    prompt=prepare_input_text(
        "Weight",
        WashingMachineFuzzyLogicSimulation.MIN_WEIGHT_VALUE,
        WashingMachineFuzzyLogicSimulation.MAX_WEIGHT_VALUE,
    ),
    required=True,
    type=click.IntRange(
        min=WashingMachineFuzzyLogicSimulation.MIN_WEIGHT_VALUE,
        max=WashingMachineFuzzyLogicSimulation.MAX_WEIGHT_VALUE,
    ),
)
@click.option(
    "--dirtiness",
    prompt=prepare_input_text(
        "Dirtiness",
        WashingMachineFuzzyLogicSimulation.MIN_DIRTINESS_VALUE,
        WashingMachineFuzzyLogicSimulation.MAX_DIRTINESS_VALUE,
    ),
    required=True,
    type=click.IntRange(
        min=WashingMachineFuzzyLogicSimulation.MIN_DIRTINESS_VALUE,
        max=WashingMachineFuzzyLogicSimulation.MAX_DIRTINESS_VALUE,
    ),
)
@click.option("--export_path", type=str)
def run(temperature: int, weight: int, dirtiness: int, export_path: str):
    """function to provide running simulation via CLI with data provided by input

    Args:
        temperature (int): [description]
        weight (int): [description]
        dirtiness (int): [description]
        export_path (str): [description]
    """
    machine = WashingMachineFuzzyLogicSimulation(temperature, weight, dirtiness)
    print(machine)
    machine.run()

    if export_path:
        machine.export_to_file(export_path)


@cli.command()
@click.option("--path", prompt="Path to JSON file", required=True)
def from_file(path: str):
    """function to provide import from CLI

    Args:
        path (str): path to the file
    """
    machine = WashingMachineFuzzyLogicSimulation.import_from_file(path)
    machine.run()


if __name__ == "__main__":
    cli()
