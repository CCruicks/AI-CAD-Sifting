import glob, os
import re
import matplotlib as mpl
from matplotlib.axes._axes import _log as matplotlib_axes_logger
import matplotlib.pyplot as plt
import numpy as np
import math

def plot(line_nums, labels, layerdicts, edges, filename):
    """ Plot a tree diagram using STEP data

            param:line_nums - list of STEP line id numbers
            param:labels - list of STEP line categories (LINE, PLANE etc.)
            param:layerdicts - dictionary where key = line id number, value = tree diagram layer
            param:edges - list of id number pairs representing connections in the tree
            param:filename - name of input STEP file
    """

    matplotlib_axes_logger.setLevel('ERROR')
    layers = []

    # produce a list of tree diagram layers for each node
    for num in line_nums:
        for ldict in layerdicts:
            if ldict["id_num"] == num:
                layers.append(ldict["layer"])

    l = layers[0]
    Xn = []
    x = 0
    width = 0

    # produces x-data for tree diagram nodes
    # values are arbitrary and chosen to fill graph area evenly
    for layer in layers:
        if layer != l:
            if width == 0:
                width = x
            for i in range(x):
                Xn.append(((i+0.5) / x) * width - width/2)
            x = 0
            l = layer
        x += 1
    Xn.append(0)

    Xe = []
    Ye = []

    # produces data to plot connections in tree
    for edge in edges:
        loc1 = line_nums.index(edge[0])
        loc2 = line_nums.index(edge[1])
        Xe.append([Xn[loc1], Xn[loc2]])
        Ye.append([layers[loc1], layers[loc2]])
    
    unique_labels = []

    # produces list of unique labels for diagram key
    for label in labels:
        if label not in unique_labels:
            unique_labels.append(label)

    clabels = []
    slabels = []
    shapes = ["o", "v", "s", "D"]

    # assigns values representing colour and shape to each node
    for label in labels:
        label_index = unique_labels.index(label)
        clabel = label_index
        clabels.append(clabel)
        slabel = shapes[label_index%4]
        slabels.append(slabel)

    N = len(unique_labels)

    # defines colourmap
    cmap = plt.cm.nipy_spectral
    cmaplist = [cmap(i) for i in range(cmap.N)]
    cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)
    bounds = np.linspace(0,N,N+1)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    colours = []

    # assigns colour values to each node
    for clabel in clabels:
        colour = cmap(norm(clabel))
        colours.append(colour)

    # create figure
    fig_width = 10 * math.floor(width/10)
    fig,ax = plt.subplots(1,1,figsize=(fig_width,20))
    ax.set_frame_on(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.axes.get_xaxis().set_visible(False)

    # plot tree nodes
    for i in range(len(Xn)):
        scat = ax.scatter(Xn[i],
                        layers[i],
                        c=colours[i],
                        s=180, 
                        marker=slabels[i])

    # annotate nodes
    for i, num in enumerate(line_nums):
        ax.annotate(num, (Xn[i]+0.1, layers[i]+0.1), size=8)

    # produce key 
    for i, label in enumerate(unique_labels):
        key = ax.scatter(width/2 + 2,
                        11 - (i/3),
                        c=cmap(norm(i)),
                        s=180,
                        marker=shapes[i%4])
        ax.annotate(label, (width/2 + 2.5, 11 - (i/3) - 0.04), size=10)

    # plot tree connections
    for xe, ye in zip(Xe, Ye):
        plt.plot(xe,ye,color='gray',linewidth=0.5)

    # save figure using the same name as the input STEP file
    file_out = filename.replace(".STEP", ".png")
    plt.savefig(file_out, bbox_inches='tight', pad_inches=0, dpi=200)


###################################################################################

input_files = []

# produce list of every STEP file in source directory
for file in glob.glob("*.STEP"):
    input_files.append(file)

for file_in in input_files:
    print("Plotting from {}".format(file_in))

    # read data from file
    STEP_FILE = open(file_in, "r")
    STEP_DATA = STEP_FILE.readlines()
    STEP_FILE.close()

    num_points = 0
    labels = []
    STEP_LINES = []
    concat_line = ""

    # convert raw input to useful data
    # store only lines starting with line id
    # remove line breaks and white space from data representing one line id
    for line in STEP_DATA:
        if concat_line == "":
            if line[0] == "#":
                concat_line += line.replace(" ","")
        else:
            concat_line += line.replace(" ","")
        if ";" in concat_line:
            STEP_LINES.append(concat_line.replace("\n",""))
            concat_line = ""

    sorted_lines = []
    sorted_nums = []

    # first add lowest-level features to sorted lines
    for line in STEP_LINES:
        line_num = re.split("=", line)[0][1:]
        if "#" not in line[1:]:
            sorted_lines.append(line)
            sorted_nums.append(line_num)

    # add features which call other features to sorted lines
    while len(sorted_lines) < len(STEP_LINES):
        for line in STEP_LINES:
            if line not in sorted_lines:
                refs = []
                split_line = re.split("\(|,|\)", line)[1:-1]
                for part in split_line:
                    if "#" in part:
                        refs.append(part[1:])
                can_write = 1
                for ref in refs:
                    if ref not in sorted_nums:
                        can_write = 0
                if can_write:
                    sorted_lines.append(line)
                    line_num = re.split("#|=", line)[1]
                    sorted_nums.append(line_num)

    reverse_sorted_lines = sorted_lines[::-1]
    filtered_nums = []
    filtered_lines = []
    line_nums = []
    labels = []
    layers = []
    layerdicts = []
    edges = []
    layer = 0

    # add highest-level shape representation to data structures
    filtered_lines.append(reverse_sorted_lines[0])
    line_nums.append(reverse_sorted_lines[0].split("=")[0][1:])
    labels.append(re.split("=|\(", reverse_sorted_lines[0])[1].strip())
    layers.append(0)
    layerdicts.append({"id_num": line_nums[0], "layer": 0})

    # produce data structures containing only features necessary for highest-level 
    # shape representation
    while(True):
        count = 0
        for line in filtered_lines:
            parent_num = line.split("=")[0][1:]
            refs = []
            split_line = re.split("\(|,|\)", line)[1:-1]
            for part in split_line:
                if "#" in part:
                    refs.append(part[1:])

            # check for direct connection between two features
            for line_comp in reverse_sorted_lines:
                line_num = line_comp.split("=")[0][1:]
                if line_num in refs:
                    edges.append((line_num, parent_num))
                    if line_comp not in filtered_lines:
                        filtered_lines.append(line_comp)
                        line_nums.append(line_num)
                        label = re.split("=|\(", line_comp)[1].strip()

                        # some lines do not have a singular label
                        # specific examples hard-coded to have relevant labels assigned
                        if not label:
                            if "CONTEXT" in line_comp:
                                label = "CONTEXT"
                            elif "UNIT" in line_comp:
                                label = "UNIT"
                        labels.append(label)
                        count += 1
                        for ldict in layerdicts:
                            if ldict["id_num"] == parent_num:
                                layer = ldict["layer"] + 1
                                layerdicts.append({"id_num": line_num, "layer": layer})
        if count == 0:
            break

    # reverse the order of all lists so lowest-level features come first
    filtered_lines = filtered_lines[::-1]
    labels = labels[::-1]
    line_nums = line_nums[::-1]
    edges = edges[::-1]

    plot(line_nums, labels, layerdicts, edges, file_in)

print("Finished")
