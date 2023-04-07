from enum import Enum
import numpy as np
from tabulate import tabulate
import plotly.graph_objects as go


class Component(Enum):
    """Enum containing the different Variance parts of GageRnR."""

    OPERATOR = 0
    PART = 1
    OPERATOR_BY_PART = 2
    MEASUREMENT = 3
    MEASUREMENT_WITHOUT_INTERACTION = 4
    TOTAL = 5


class MyComponent(Enum):
    GRR = 0
    EV = 1
    AV = 2
    OPERATOR_BY_PART = 3
    PV = 4
    TOTAL_VAR = 5


class ComponentNoInter(Enum):
    GRR_WITHOUT_INTERACTION = 0
    EV_WITHOUT_INTERACTION = 1
    AV = 2
    OPERATOR_BY_PART = 3
    PV = 4
    TOTAL_VAR = 5


ComponentNames = {
    Component.OPERATOR: 'Operator',
    Component.PART: 'Part',
    Component.OPERATOR_BY_PART: 'Operator by Part',
    Component.MEASUREMENT: 'Measurement (with interaction)',
    Component.TOTAL: 'Total',
    Component.MEASUREMENT_WITHOUT_INTERACTION: 'Measurement (without interaction)'}

MyComponentNames = {
    MyComponent.GRR: 'GRR',
    MyComponent.EV: 'Equipment (Repetability)',
    MyComponent.AV: 'Operators',
    MyComponent.OPERATOR_BY_PART: 'Interaction',
    MyComponent.PV: 'Parts',
    MyComponent.TOTAL_VAR: 'Total'}

ComponentNamesNoInter = {
    ComponentNoInter.GRR_WITHOUT_INTERACTION: 'GRR without interaction',
    ComponentNoInter.EV_WITHOUT_INTERACTION: 'Equipment (Repetability) without interaction',
    ComponentNoInter.AV: 'Operators',
    ComponentNoInter.OPERATOR_BY_PART: 'Interaction',
    ComponentNoInter.PV: 'Parts',
    ComponentNoInter.TOTAL_VAR: 'Total'}


class Result(Enum):
    """Enum containing the measurements calculated by GageRnR."""

    DF = 0
    Mean = 1
    SS = 3
    MS = 4
    Var = 5
    Std = 6
    F = 7
    P = 8
    W = 9
    K = 10
    Bias = 11

    Std_results = 12
    Std_percentage = 13
    Std_results_without_interaction = 14
    Std_percentage_without_interaction = 15
    Variance = 16
    Variance_no_inter = 17


ResultNames = {
    Result.Mean: 'Mean',
    Result.Std: 'Std'}


class Statistics(object):

    title = "Statistics"

    def __init__(self, data, labels=None):
        self.data = data
        self.parts = data.shape[1]
        self.operators = data.shape[0]
        self.measurements = data.shape[2]

        if labels is None:
            self.labels = {}
        else:
            self.labels = labels

        if "Operator" not in self.labels:
            self.labels["Operator"] = [("Operator %d" % x)
                                       for x in range(self.operators)]

        if "Part" not in self.labels:
            self.labels["Part"] = [("Part %d" % x) for x in range(self.parts)]

    def __str__(self):
        """Enum containing the measurements calculated by Statistics."""
        if not hasattr(self, 'result'):
            return 'Shape: ' + \
                str([self.operators, self.parts, self.measurements])
        return self.summary()

    def summary(self, tableFormat="fancy_grid", precision='.3f'):
        """Convert result to tabular."""
        if not hasattr(self, 'result'):
            raise Exception(
                'Statistics.calculate() should be run before calling summary()')

        headers = ['Sources of Variance',
                   ResultNames[Result.Mean],
                   ResultNames[Result.Std]]

        table = []
        results = [Result.Mean, Result.Std]
        self.addToTable(results, Component.TOTAL, table, precision)
        self.addToTable(results, Component.OPERATOR, table, precision)
        self.addToTable(results, Component.PART, table, precision)

        return tabulate(
            table,
            headers=headers,
            tablefmt=tableFormat)

    def summary_mio(self, precision='.3f'):
        """Convert result to tabular."""
        if not hasattr(self, 'result'):
            raise Exception(
                'Statistics.calculate() should be run before calling summary()')

        table = []
        results = [Result.Mean, Result.Std]
        self.addToTable(results, Component.TOTAL, table, precision)
        self.addToTable(results, Component.OPERATOR, table, precision)
        self.addToTable(results, Component.PART, table, precision)

        return table

    def createOperatorsBoxData(self):
        data = []
        for i in range(0, self.operators):
            data.append(go.Box(
                y=self.data[i, :, :].flatten(),
                boxpoints='all',
                name=self.labels["Operator"][i],
                notched=True,
                boxmean='sd'))
        return data

    def createOperatorsBoxPlot(self):
        data = self.createOperatorsBoxData()
        fig = go.Figure(data=data)
        return fig

    def createPartsBoxData(self):
        data = []
        for i in range(0, self.parts):
            data.append(go.Box(
                y=self.data[:, i, :].flatten(),
                boxpoints='all',
                name=self.labels["Part"][i],
                notched=True,
                boxmean='sd'))
        return data

    def createPartsBoxPlot(self):
        data = self.createPartsBoxData()
        fig = go.Figure(data=data)
        return fig

    def addToTable(self, results, component, table, precision='.3f'):
        size = self.result[results[0]][component].size
        for i in range(0, self.result[results[0]][component].size):
            name = ComponentNames[component]
            if (size > 1):
                name += ' ' + str(i)
            row = [name]
            for result in results:
                row.append(
                    format(self.result[result][component][i], precision))
            table.append(row)

    def calculateMean(self):
        """Calculate Mean."""
        mu = np.array([np.mean(self.data)])

        omu = np.mean(self.data, axis=1)
        omu = np.mean(omu, axis=1)

        pmu = np.mean(self.data, axis=0)
        pmu = np.mean(pmu, axis=1)

        emu = np.mean(self.data, axis=2)
        emu = emu.reshape(self.parts * self.operators)

        return {
            Component.TOTAL: mu,
            Component.OPERATOR: omu,
            Component.PART: pmu,
            Component.MEASUREMENT: emu}

    def calculateStd(self):
        std = np.array([np.std(self.data, ddof=1)])
        stdo = np.std(
            self.dataToOperators(),
            axis=1,
            ddof=1)
        stdp = np.std(
            self.dataToParts(),
            axis=1,
            ddof=1)
        return {
            Component.TOTAL: std,
            Component.OPERATOR: stdo,
            Component.PART: stdp
        }

    def calculate(self):
        self.result = dict()
        self.result[Result.Mean] = self.calculateMean()
        self.result[Result.Std] = self.calculateStd()

    def dataToParts(self):
        data = np.transpose(self.data, axes=(1, 0, 2))
        return data.reshape(
            self.parts,
            self.measurements*self.operators)

    def dataToOperators(self):
        return self.data.reshape(
            self.operators,
            self.measurements*self.parts)
