
from shapely.geometry import *
import matplotlib.pyplot as plt


def plot_polygons_and_linestrings(structure_to_plot, color_for_plot='#00000'):

    if isinstance(structure_to_plot, MultiLineString):
        for bit_to_plot in structure_to_plot:
            x, y = bit_to_plot.xy
            plt.plot(x, y, color=color_for_plot)
    elif isinstance(structure_to_plot, MultiPolygon):
        for bit_to_plot in structure_to_plot:
            x, y = bit_to_plot.boundary.xy
            plt.plot(x, y, color=color_for_plot)
    elif isinstance(structure_to_plot, Polygon):
        x, y = structure_to_plot.boundary.xy
        plt.plot(x, y, color=color_for_plot)
    elif isinstance(structure_to_plot, LineString):
        x, y = structure_to_plot.xy
        plt.plot(x, y, color=color_for_plot)
    else:
        print('Unable to plot structure type: ', type(structure_to_plot))


def main():
    # Put in just in case. Not used at the moment
    print('This file is not very standalone')
    return


if __name__ == '__main__':
    main()
