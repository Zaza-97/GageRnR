"""Module containing the algorithm for GageRnR."""
import numpy as np
import math
import scipy.stats as stats
from tabulate import tabulate
from .statistics import Statistics, Result, Component, MyComponent, \
    ComponentNoInter, ComponentNamesNoInter, ComponentNames, MyComponentNames

ResultNames = {
    Result.DF: 'DF',
    Result.SS: 'SS',
    Result.MS: 'MS',
    # Result.Var: 'Var',
    # Result.Std: 'Std',
    Result.F: 'F-value',
    Result.P: 'P-value'
}

MyResultNames = {
    Result.Variance: 'Var',
    Result.Std_results: 'Standar deviations',
    Result.Std_percentage: '% of Total Std',
}

MyResultNamesNoInteraction = {
    Result.Variance_no_inter: 'Var',
    Result.Std_results_without_interaction: 'Standar deviations',
    Result.Std_percentage_without_interaction: '% of Total Std'
}


class GageRnR(Statistics):
    """Main class for calculating GageRnR."""

    GRR = 'GageRnR'
    GRR_WITHOUT_INTERACTION = 'GageRnR without operator-part interaction'
    title = "Gauge R&R"

    def __init__(self, data):
        """Initialize GageRnR algorithm.

        :param numpy.array data:
            The data tha we want to analyse using GageRnR.
            The input should be structured in a 3d array
            n[i,j,k] where i = operator, j = part, k = measurement
        """
        super().__init__(data)

    def summary(self, tableFormat="fancy_grid", precision='.3f'):
        """Convert result to tabular."""
        if not hasattr(self, 'result'):
            raise Exception(
                'GageRnR.calculate() should be run before calling summary()')

        headers = ['Sources of Variance']

        for key in ResultNames:
            headers.append(ResultNames[key])

        table = []
        for comp in Component:
            innerTable = [ComponentNames[comp]]
            for key in ResultNames:
                if key not in (Result.Var, Result.Std):
                    if comp in self.result[key]:
                        innerTable.append(
                            format(self.result[key][comp], precision))
                    else:
                        innerTable.append('')

            table.append(innerTable)
        return tabulate(
            table,
            headers=headers,
            tablefmt=tableFormat)

    # Mio
    def summary_2(self, tableFormat="fancy_grid", precision='.3f'):
        """Convert result to tabular."""
        if not hasattr(self, 'result'):
            raise Exception(
                'GageRnR.calculate() should be run before calling summary()')

        headers = ['Std. Deviation']

        for key in MyResultNames:
            headers.append(MyResultNames[key])

        table = []
        for comp in MyComponent:
            innerTable = [MyComponentNames[comp]]
            for key in MyResultNames:
                if comp in self.result[key]:
                    innerTable.append(
                        format(self.result[key][comp], precision))
                else:
                    innerTable.append('')

            table.append(innerTable)

        return tabulate(
            table,
            headers=headers,
            tablefmt=tableFormat)

    # Mio

    def summary_3(self, tableFormat="fancy_grid", precision='.3f'):
        """Convert result to tabular."""
        if not hasattr(self, 'result'):
            raise Exception(
                'GageRnR.calculate() should be run before calling summary()')

        headers = ['Std. Deviation']

        for key in MyResultNamesNoInteraction:
            headers.append(MyResultNamesNoInteraction[key])

        table = []
        for comp in ComponentNoInter:
            innerTable = [ComponentNamesNoInter[comp]]
            for key in MyResultNamesNoInteraction:
                if comp in self.result[key]:
                    innerTable.append(
                        format(self.result[key][comp], precision))
                else:
                    innerTable.append('')

            table.append(innerTable)

        return tabulate(
            table,
            headers=headers,
            tablefmt=tableFormat)

# Mio per analisi dati strumenti di misur
    def summary_instruments(self, precision='.10f'):
        """Convert result to tabular."""
        if not hasattr(self, 'result'):
            raise Exception(
                'GageRnR.calculate() should be run before calling summary()')

        headers = ['Std. Deviation']

        for key in MyResultNames:
            headers.append(MyResultNames[key])

        table = []
        for comp in MyComponent:
            innerTable = [MyComponentNames[comp]]
            for key in MyResultNames:
                if comp in self.result[key]:
                    innerTable.append(
                        format(self.result[key][comp], precision))
                else:
                    innerTable.append('')

            table.append(innerTable)

        return table

    def calculate(self):
        """Calculate GageRnR."""
        self.result = dict()
        self.result[Result.DF] = self.calculateDoF()
        self.result[Result.Mean] = self.calculateMean()
        self.result[Result.SS] = self.calculateSS()

        self.result[Result.MS] = self.calculateMS(
            self.result[Result.DF],
            self.result[Result.SS])

        self.result[Result.Var],                                \
            self.result[Result.Std_results],                        \
            self.result[Result.Std_percentage],                     \
            self.result[Result.Std_results_without_interaction],    \
            self.result[Result.Std_percentage_without_interaction], \
            self.result[Result.Variance],                           \
            self.result[Result.Variance_no_inter],                  \
            = self.calculateVar(self.result[Result.MS])

        self.result[Result.Std] = self.calculateStd(self.result[Result.Var])

        self.result[Result.F] = self.calculateF(self.result[Result.MS])

        self.result[Result.P] = self.calculateP(
            self.result[Result.DF],
            self.result[Result.F])

        return self.result

    def calculateDoF(self):
        """Calculate Degrees of freedom."""
        oDoF = self.operators - 1
        pDoF = self.parts - 1
        opDoF = (self.parts - 1) * (self.operators - 1)
        eDof = self.parts * self.operators * (self.measurements - 1)
        totDof = self.parts * self.operators * self.measurements - 1
        return {
            Component.OPERATOR: oDoF,
            Component.PART: pDoF,
            Component.OPERATOR_BY_PART: opDoF,
            Component.MEASUREMENT: eDof,
            Component.MEASUREMENT_WITHOUT_INTERACTION: eDof + opDoF,
            Component.TOTAL: totDof}

    def calculateSquares(self):
        """Calculate Squares."""
        mean = self.calculateMean()
        tS = (self.data - mean[Component.TOTAL])**2
        oS = (mean[Component.OPERATOR] - mean[Component.TOTAL])**2
        pS = (mean[Component.PART] - mean[Component.TOTAL])**2

        dataE = self.data.reshape(
            self.operators * self.parts,
            self.measurements)
        meanMeas = np.repeat(mean[Component.MEASUREMENT], self.measurements)
        meanMeas = meanMeas.reshape(
            self.operators * self.parts,
            self.measurements)

        mS = (dataE - meanMeas)**2
        return {
            Component.TOTAL: tS,
            Component.OPERATOR: oS,
            Component.PART: pS,
            Component.MEASUREMENT: mS}

    def calculateSumOfDeviations(self):
        """Calculate Sum of Deviations."""
        squares = self.calculateSquares()
        SD = dict()
        for key in squares:
            SD[key] = np.sum(squares[key])
        return SD

    def calculateSS(self):
        """Calculate Sum of Squares."""
        SS = self.calculateSumOfDeviations()

        SS[Component.OPERATOR] = \
            self.parts * self.measurements * \
            SS[Component.OPERATOR]

        SS[Component.PART] = \
            self.operators * self.measurements * \
            SS[Component.PART]

        SS[Component.OPERATOR_BY_PART] = \
            SS[Component.TOTAL] - (
                SS[Component.OPERATOR] +
                SS[Component.PART] +
                SS[Component.MEASUREMENT])

        SS[Component.MEASUREMENT_WITHOUT_INTERACTION] = SS[Component.MEASUREMENT] + \
            SS[Component.OPERATOR_BY_PART]

        return SS

    def calculateMS(self, dof, SS):
        """Calculate Mean of Squares."""
        MS = dict()

        for key in SS:

            if key != Component.TOTAL:
                MS[key] = SS[key] / dof[key]

        return MS

    def calculateVar(self, MS):
        """Calculate GageRnR Variances."""

        # Variability considering intereaction between parts and operator
        def check_pos(Var):
            for key in Var:
                if Var[key] < 0:
                    Var[key] = 0
        '''
        Var = dict()
    
        Var[Component.MEASUREMENT] = MS[Component.MEASUREMENT]
       
        Var[Component.OPERATOR_BY_PART] = ((
            MS[Component.OPERATOR_BY_PART] - MS[Component.MEASUREMENT]) /
            self.parts)
       
        Var[Component.OPERATOR] = ((
            MS[Component.OPERATOR] - MS[Component.OPERATOR_BY_PART]) /
            (self.parts * self.measurements))

        Var[Component.PART] = ((
            MS[Component.PART] - MS[Component.OPERATOR_BY_PART]) /
            (self.operators * self.measurements))

        check_pos(Var)


        Var[Component.TOTAL] = \
            Var[Component.OPERATOR] + \
            Var[Component.PART] + \
            Var[Component.OPERATOR_BY_PART] + \
            Var[Component.MEASUREMENT]

        Var[GageRnR.GRR] = \
            Var[Component.MEASUREMENT] + \
            Var[Component.OPERATOR] + \
            Var[Component.OPERATOR_BY_PART]
        '''

        # Considering interaction
        MyVar = dict()

        MyVar[MyComponent.EV] = MS[Component.MEASUREMENT]

        MyVar[MyComponent.OPERATOR_BY_PART] = ((
            MS[Component.OPERATOR_BY_PART] - MS[Component.MEASUREMENT]) /
            self.parts)

        MyVar[MyComponent.AV] = ((
            MS[Component.OPERATOR] - MS[Component.OPERATOR_BY_PART]) /
            (self.parts * self.measurements))

        MyVar[MyComponent.PV] = ((
            MS[Component.PART] - MS[Component.OPERATOR_BY_PART]) /
            (self.operators * self.measurements))

        check_pos(MyVar)

        MyVar[MyComponent.TOTAL_VAR] = \
            MyVar[MyComponent.AV] + \
            MyVar[MyComponent.PV] + \
            MyVar[MyComponent.OPERATOR_BY_PART] + \
            MyVar[MyComponent.EV]

        MyVar[MyComponent.GRR] = \
            MyVar[MyComponent.EV] + \
            MyVar[MyComponent.AV] + \
            MyVar[MyComponent.OPERATOR_BY_PART]

        MyStd = dict()
        MyStd[MyComponent.GRR] = math.sqrt(MyVar[MyComponent.GRR])
        MyStd[MyComponent.EV] = math.sqrt(MyVar[MyComponent.EV])
        MyStd[MyComponent.AV] = math.sqrt(MyVar[MyComponent.AV])
        MyStd[MyComponent.OPERATOR_BY_PART] = math.sqrt(
            MyVar[MyComponent.OPERATOR_BY_PART])
        MyStd[MyComponent.PV] = math.sqrt(MyVar[MyComponent.PV])
        MyStd[MyComponent.TOTAL_VAR] = math.sqrt(MyVar[MyComponent.TOTAL_VAR])

        MyStd_percent = dict()
        MyStd_percent[MyComponent.GRR] = 100 * \
            math.sqrt(MyVar[MyComponent.GRR] / MyVar[MyComponent.TOTAL_VAR])
        MyStd_percent[MyComponent.EV] = 100 * \
            math.sqrt(MyVar[MyComponent.EV] / MyVar[MyComponent.TOTAL_VAR])
        MyStd_percent[MyComponent.AV] = 100 * \
            math.sqrt(MyVar[MyComponent.AV] / MyVar[MyComponent.TOTAL_VAR])
        MyStd_percent[MyComponent.OPERATOR_BY_PART] = 100 * \
            math.sqrt(MyVar[MyComponent.OPERATOR_BY_PART] /
                      MyVar[MyComponent.TOTAL_VAR])
        MyStd_percent[MyComponent.PV] = 100 * \
            math.sqrt(MyVar[MyComponent.PV] / MyVar[MyComponent.TOTAL_VAR])
        MyStd_percent[MyComponent.TOTAL_VAR] = 100 * \
            math.sqrt(MyVar[MyComponent.TOTAL_VAR] /
                      MyVar[MyComponent.TOTAL_VAR])

        # Without considering interaction
        MyVar_no_inter = dict()

        MyVar_no_inter[ComponentNoInter.EV_WITHOUT_INTERACTION] = MS[Component.MEASUREMENT_WITHOUT_INTERACTION]

        MyVar_no_inter[ComponentNoInter.OPERATOR_BY_PART] = ((
            MS[Component.OPERATOR_BY_PART] - MS[Component.MEASUREMENT_WITHOUT_INTERACTION]) /
            self.parts)

        MyVar_no_inter[ComponentNoInter.AV] = ((
            MS[Component.OPERATOR] - MS[Component.OPERATOR_BY_PART]) /
            (self.parts * self.measurements))

        MyVar_no_inter[ComponentNoInter.PV] = ((
            MS[Component.PART] - MS[Component.OPERATOR_BY_PART]) /
            (self.operators * self.measurements))

        check_pos(MyVar_no_inter)

        MyVar_no_inter[ComponentNoInter.TOTAL_VAR] = \
            MyVar_no_inter[ComponentNoInter.AV] + \
            MyVar_no_inter[ComponentNoInter.PV] + \
            MyVar_no_inter[ComponentNoInter.OPERATOR_BY_PART] + \
            MyVar_no_inter[ComponentNoInter.EV_WITHOUT_INTERACTION]

        MyVar_no_inter[ComponentNoInter.GRR_WITHOUT_INTERACTION] = \
            MyVar_no_inter[ComponentNoInter.EV_WITHOUT_INTERACTION] + \
            MyVar_no_inter[ComponentNoInter.AV] + \
            MyVar_no_inter[ComponentNoInter.OPERATOR_BY_PART]

        MyStd_no_interaction = dict()
        MyStd_no_interaction[ComponentNoInter.EV_WITHOUT_INTERACTION] = math.sqrt(
            MyVar_no_inter[ComponentNoInter.EV_WITHOUT_INTERACTION])
        MyStd_no_interaction[ComponentNoInter.AV] = math.sqrt(
            MyVar_no_inter[ComponentNoInter.AV])
        MyStd_no_interaction[ComponentNoInter.OPERATOR_BY_PART] = math.sqrt(
            MyVar_no_inter[ComponentNoInter.OPERATOR_BY_PART])
        MyStd_no_interaction[ComponentNoInter.PV] = math.sqrt(
            MyVar_no_inter[ComponentNoInter.PV])
        MyStd_no_interaction[ComponentNoInter.GRR_WITHOUT_INTERACTION] = math.sqrt(
            MyVar_no_inter[ComponentNoInter.GRR_WITHOUT_INTERACTION])
        MyStd_no_interaction[ComponentNoInter.TOTAL_VAR] = math.sqrt(
            MyVar_no_inter[ComponentNoInter.TOTAL_VAR])

        MyStd_percent_no_interaction = dict()

        MyStd_percent_no_interaction[ComponentNoInter.GRR_WITHOUT_INTERACTION] = 100 * \
            math.sqrt(MyVar_no_inter[ComponentNoInter.GRR_WITHOUT_INTERACTION] /
                      MyVar_no_inter[ComponentNoInter.TOTAL_VAR])

        MyStd_percent_no_interaction[ComponentNoInter.EV_WITHOUT_INTERACTION] = 100 * \
            math.sqrt(MyVar_no_inter[ComponentNoInter.EV_WITHOUT_INTERACTION] /
                      MyVar_no_inter[ComponentNoInter.TOTAL_VAR])

        MyStd_percent_no_interaction[ComponentNoInter.AV] = 100 * \
            math.sqrt(MyVar_no_inter[ComponentNoInter.AV] /
                      MyVar_no_inter[ComponentNoInter.TOTAL_VAR])

        MyStd_percent_no_interaction[ComponentNoInter.OPERATOR_BY_PART] = 100 * \
            math.sqrt(MyVar_no_inter[ComponentNoInter.OPERATOR_BY_PART] /
                      MyVar_no_inter[ComponentNoInter.TOTAL_VAR])

        MyStd_percent_no_interaction[ComponentNoInter.PV] = 100 * \
            math.sqrt(MyVar_no_inter[ComponentNoInter.PV] /
                      MyVar_no_inter[ComponentNoInter.TOTAL_VAR])

        MyStd_percent_no_interaction[ComponentNoInter.TOTAL_VAR] = 100

        return MyVar, MyStd, MyStd_percent, MyStd_no_interaction, \
            MyStd_percent_no_interaction, MyVar, MyVar_no_inter

    def calculateStd(self, Var):
        """Calculate GageRnR Standard Deviations."""
        Std = dict()
        for key in Var:
            Std[key] = math.sqrt(Var[key])

        return Std

    def calculateF(self, MS):
        """Calculate F-Values."""
        F = dict()

        F[Component.OPERATOR] = (
            MS[Component.OPERATOR] /
            MS[Component.OPERATOR_BY_PART])

        F[Component.PART] = (
            MS[Component.PART] /
            MS[Component.OPERATOR_BY_PART])

        F[Component.OPERATOR_BY_PART] = (
            MS[Component.OPERATOR_BY_PART] /
            MS[Component.MEASUREMENT])

        return F

    def calculateP(self, dof, F):
        """Calculate P-Values."""
        P = dict()

        P[Component.OPERATOR] = \
            stats.f.sf(
            F[Component.OPERATOR],
            dof[Component.OPERATOR],
            dof[Component.OPERATOR_BY_PART])

        P[Component.PART] = \
            stats.f.sf(
            F[Component.PART],
            dof[Component.PART],
            dof[Component.OPERATOR_BY_PART])

        P[Component.OPERATOR_BY_PART] = \
            stats.f.sf(
            F[Component.OPERATOR_BY_PART],
            dof[Component.OPERATOR_BY_PART],
            dof[Component.MEASUREMENT])
        return P
