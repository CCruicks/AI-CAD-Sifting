import glob, os
import re
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

def plot(line_nums, labels, layerdicts, edges, filename):

    layers = []
    for num in line_nums:
        for ldict in layerdicts:
            if ldict["id_num"] == num:
                layers.append(ldict["layer"])

    l = layers[0]
    Xn = []
    x = 0
    width = 0
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
    for edge in edges:
        loc1 = line_nums.index(edge[0])
        loc2 = line_nums.index(edge[1])
        Xe.append([Xn[loc1], Xn[loc2]])
        Ye.append([layers[loc1], layers[loc2]])
    
    unique_labels = []
    for label in labels:
        if label not in unique_labels:
            unique_labels.append(label)
    clabels = []
    for label in labels:
        clabel = unique_labels.index(label)
        clabels.append(clabel)
    N = len(unique_labels)

    cmap = plt.cm.nipy_spectral
    cmaplist = [cmap(i) for i in range(cmap.N)]
    cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)
    bounds = np.linspace(0,N,N+1)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    fig,ax = plt.subplots(1,1,figsize=(30,8))
    ax.set_frame_on(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.axes.get_xaxis().set_visible(False)
    scat = ax.scatter(Xn,layers,c=clabels,cmap=cmap,norm=norm,s=180)
    cb = plt.colorbar(scat, spacing='proportional',ticks=bounds)
    cb.ax.tick_params(labelsize=10)
    cb.ax.set_yticklabels(unique_labels)

    for i, num in enumerate(line_nums):
        ax.annotate(num, (Xn[i]+0.1, layers[i]+0.1), size=8)

    for xe, ye in zip(Xe, Ye):
        plt.plot(xe,ye,color='gray',linewidth=0.5)

    file_out = filename.replace(".STEP", ".png")
    plt.savefig(file_out, bbox_inches='tight', pad_inches=0, dpi=200)

selected = False
while(not selected):
    print("\nAvailable files to analyse:")

    for file in glob.glob("*.STEP"):
        print(file)

    file_in = input("\nEnter the filename to open. ")

    try:
        STEP_FILE = open(file_in, "r")
        selected = True
    except (IOError, OSError):
        print("\nNot a valid filename.")

STEP_DATA = STEP_FILE.readlines()
STEP_FILE.close()

num_points = 0
labels = []
STEP_LINES = []
concat_line = ""
for line in STEP_DATA:
    if concat_line == "":
        if line[0] == "#":
            concat_line += line.replace(" ","")
    else:
        concat_line += line.replace(" ","")
    if ";" in concat_line:
        STEP_LINES.append(concat_line.replace("\n",""))
        label = re.split("=|\(", concat_line)[1].strip()
        if label not in labels:
            labels.append(label)
        concat_line = ""

labels = sorted(labels)
print("Labels: ")

for label in labels:
    print(label)

exit = False
while(not exit):
    choice = input("Type a label or line number for more information, p to reprint labels, s to sort and filter lines or exit to close. ")
    if choice == "exit":
        exit = True
    elif choice == "p":
        print("")
        for label in labels:
            print(label)
    elif choice == "s":
        print("\n")
        sorted_lines = []
        sorted_nums = []
        for line in STEP_LINES:
            line_num = re.split("=", line)[0][1:]
            if "#" not in line[1:]:
                sorted_lines.append(line)
                sorted_nums.append(line_num)
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

        filtered_lines = []
        filtered_nums = []
        reverse_sorted_lines = sorted_lines[::-1]
        filtered_lines.append(reverse_sorted_lines[0])
        line_nums = []
        labels = []
        layers = []
        line_nums.append(reverse_sorted_lines[0].split("=")[0][1:])
        labels.append(re.split("=|\(", reverse_sorted_lines[0])[1].strip())
        layers.append(0)
        layer = 0

        layerdicts = []
        layerdicts.append({"id_num": line_nums[0], "layer": 0})

        edges = []

        while(True):
            count = 0
            for line in filtered_lines:
                parent_num = line.split("=")[0][1:]
                refs = []
                split_line = re.split("\(|,|\)", line)[1:-1]
                for part in split_line:
                    if "#" in part:
                        refs.append(part[1:])
                for line_comp in reverse_sorted_lines:
                    line_num = line_comp.split("=")[0][1:]
                    if line_num in refs:
                        if line_comp not in filtered_lines:
                            filtered_lines.append(line_comp)
                            line_nums.append(line_num)
                            label = re.split("=|\(", line_comp)[1].strip()
                            labels.append(label)
                            edges.append((line_num, parent_num))
                            for ldict in layerdicts:
                                if ldict["id_num"] == parent_num:
                                    layer = ldict["layer"] + 1
                                    layerdicts.append({"id_num": line_num, "layer": layer})
                            count += 1
            if count == 0:
                break
        filtered_lines = filtered_lines[::-1]
        labels = labels[::-1]
        line_nums = line_nums[::-1]
        edges = edges[::-1]


        for line in filtered_lines:
            print(line[:-1])
        print("\n{} total lines.\n".format(len(filtered_lines)))
        print("\n")

        plot(line_nums, labels, layerdicts, edges, file_in)


    elif choice not in labels:
        try:
            num_choice = int(choice)
            refs = []
            print("")
            invalid = True
            for line in STEP_LINES:
                line_num = re.split("#|=", line)[1]
                if choice == line_num:
                    print(line[0:-2])
                    invalid = False
                elif ("#" + choice + ",") in line:
                    refs.append(line)
                elif ("#" + choice + " ") in line:
                    refs.append(line)
            if invalid:
                print("Not a valid line number.\n")
                continue
            p = input("Type p to print all references. Any other key returns to menu. ")
            if p == "p":
                for line in refs:
                    print(line[0:-2])
            print("")
        except ValueError:
            print("")
            print("Not a valid label.")
            print("")
            continue
    else:
        print("")
        counter = 0
        for line in STEP_LINES:
            if choice in line:
                counter += 1
        print("There are {} total lines with the label {}.".format(counter, choice))
        p = input("Type p to print all lines. Any other key returns to menu. ")
        if p == "p":
            for line in STEP_LINES:
                if choice in line:
                    print(line[0:-2])
            print("")
        else:
            continue

