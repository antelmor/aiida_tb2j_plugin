import numpy as np
from aiida.orm import ArrayData, StructureData
from .exchange import MagSite
from scipy.optimize import curve_fit

class CurieData(ArrayData):

    @property
    def units(self):

        return self.get_attribute('units', None)

    @units.setter
    def units(self, value):

        the_units = dict(value)

        self.set_attribute('units', the_units)

    def set_temperature_values(self, values):

        try:
            the_values = np.array(values, dtype=np.float64)
        except ValueError:
            raise ValueError("The temperature values must be numerical data.")
        if the_values.ndim != 1:
            raise ValueError("The temperature values must be an array of dimension 1.")

        self.set_array('temperature', the_values)

    def get_temperature_values(self):

        try:
            return self.get_array('temperature')
        except KeyError:
            raise AttributeError("No stored 'temperature' values have been found.")

    def set_magnetization_values(self, values):

        try:
            temperature = self.get_array('temperature')
        except KeyError:
            raise AttributeError("The temperature values must be set first, then the magnetization.")
        try:
            the_values = np.array(values, dtype=np.float64)
        except ValueError:
            raise ValueError("The magnetization values must be numerical data.")
        if the_values.ndim > 2:
            raise ValueError("The magnetization values must be an array of dimension no greater than 2.")

        self.set_array('magnetization', the_values)

    def get_magnetization_values(self):

        try:
            return self.get_array('magnetization')
        except KeyError:
            raise AttributeError("No stored 'magnetization' values have been found.")

    def _magnetization_fit(self, exponent=0.5):

        try:
            magnetization_values = self.get_array('magnetization').T
        except KeyError:
            raise AttributeError("The 'magnetization' values must be set before calculating the critical temperature")
        temperature_values = self.get_array('temperature')

        def magnetization(T, Tc):
            return np.where(
                T > Tc,
                0.0,
                (1 - T/Tc)**exponent)

        Tc_vals = []
        bounds = (0.0, np.inf)
        for row in magnetization_values:
            init_guess = np.mean(temperature_values) / (1 - (np.mean(row)**(1/exponent)))
            pars, cov = curve_fit(
                f=magnetization, xdata=temperature_values, ydata=row, p0=init_guess, bounds=bounds
            )
            Tc_vals.append(pars[0])
        Tc = np.mean(Tc_vals)

        return lambda T : magnetization(T, Tc), Tc

    def get_critical_temperature(self, exponent=0.5):

        _, Tc = self._magnetization_fit(exponent=exponent)

        return Tc
