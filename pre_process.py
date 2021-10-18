import glob
import os
from shutil import copy

import time
import math
import random
import pickle
import torch

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

# Local install
try:
    import glovar
    from tree import TreeNode, Tree

# coLab install
except(ModuleNotFoundError):
    from AI_engineering_management.treelstm.tree import TreeNode, Tree
    from AI_engineering_management.treelstm import glovar

# Return list of all files in a folder
def findFiles(path): return glob.glob(path)

# Read a file's contents
def readFile(path):
    f = open(path, "r")
    d = f.readlines()
    f.close()
    return d

# Concatenate two strings, removing any spaces
def concatLine(current, new): 
    current += new.replace(" ","")
    return current

# Process STEP nodes into standard form - nodes can cover more than one STEP line
def procLine(current_line, line):
    if current_line == "":
        # '#' represents start of line from main body of file
        if line[0] == "#":
            current_line = concatLine(current_line, line)
    else:
        current_line = concatLine(current_line, line)
    return current_line

# Return node ID number, category and children (parameters) for one STEP line
def getValues(step_line):
    node_id = int(step_line.split("=")[0][1:])
    node_label = step_line.split("=")[1].split("(")[0]
    children = step_line.split("(")[-1].strip(";").replace(")", "").split(",")
    return node_id, node_label, children

# Return time taken in minutes and seconds
def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s/60)
    s -= m * 60
    return m, s

# Randomly sort STEP dataset into train, test and validate sets at the ratio 7:2:1
def sortDataset(source):

    # Destination folders
    train_dir = os.path.join(glovar.DATA_DIR, "TRAIN/")
    test_dir = os.path.join(glovar.DATA_DIR, "TEST/")
    val_dir = os.path.join(glovar.DATA_DIR, "VAL/")

    val_files = findFiles(val_dir + "/*.STEP")
    num_val = len(val_files)
    if num_val > 0:
        print("Dataset already sorted.")
        return False

    print("Sorting dataset into train, test and val datasets")

    # Search all subdirectories in source directory
    # (STEP files are generated in subdirectories based on number of features)
    for d in findFiles(source + "/*/"):
        files = findFiles(d + "/STEP/*.STEP")
        random.shuffle(files)
        num_files = len(files)
        print("{} features: {} files".format(d[-2], num_files))
        if num_files > 0:
            lim_1 = int(0.7 * num_files)
            lim_2 = int(0.9 * num_files)
            for i in range(lim_1):
                copy(files[i], train_dir)
            for i in range(lim_1, lim_2):
                copy(files[i], test_dir)
            for i in range(lim_2, num_files):
                copy(files[i], val_dir)

# Return a tree object for each STEP file in the dataset
# Input can be raw STEP files or a data file from running this function previously
def procDataset(dataset):
    start = time.time()
    trees = []

    data_file = os.path.join(glovar.DATA_DIR, dataset + '_DATA')

    # File contains list of node categories seen in the dataset
    labels_file = os.path.join(glovar.DATA_DIR, 'NODE_LABELS.txt')

    # Check if this dataset has already been processed
    # If so, simply load the data file
    try:
        with open(data_file, "rb") as fp:
            tree_list = pickle.load(fp)
        print("Reading from stored dataset")
        for tree in tree_list:
            t = Tree(tree.pop(0))
            t.nodesFromList(tree)
            trees.append(t)

        setLabels(trees, labels_file)

        m, s = timeSince(start)
        print("Done. Processed dataset read in {}m, {}s.".format(m,round(s)))
        return trees

    except(FileNotFoundError):
        print("No processed dataset found.")

    data_dir = os.path.join(glovar.DATA_DIR, dataset)
    files = findFiles(data_dir + "\\*.STEP")
    num_models = len(files)

    if num_models == 0:
        print("No STEP dataset found.")
        return False

    print("Processing {} dataset ({} files)...".format(dataset, num_models))

    # For each STEP file in the dataset
    for filepath in files:

        model_id = filepath.split("\\")[-1].split(".")[0]
        t = Tree(model_id)

        STEP_DATA = readFile(filepath)
        STEP_LINES = []
        line_ids = []
        current_line = ""

        # STEP_LINES = list of all nodes from STEP file in standardised text form
        for line in STEP_DATA:
            current_line = procLine(current_line, line)
            if ";" in current_line:
                STEP_LINES.append(current_line.replace("\n",""))
                current_line = ""

        # Convert STEP lines into tree nodes
        for line in STEP_LINES:
            node_id, node_label, children = getValues(line)
            n = TreeNode(node_id, node_label)
            line_ids.append(node_id)

            # Add children to new tree node
            for child in children:
                try:

                    # Coordinate points - initially set as one child
                    # Will be reformatted later
                    try:
                        child = float(child)
                        n.addChild(child)

                    except ValueError:

                        # Child is another STEP line
                        if child[0] == "#":
                            n.addChild(int(child[1:]))

                        # Ignore children which add no information
                        elif child == "\'NONE\'" or child == "\'\'" or child == "":
                            pass

                        # Child is an additional parameter, not a STEP line
                        else:
                            n.addChild(child)

                except IndexError:
                    break

            # Add the new node to the tree
            t.addNode(n)

        coords = {}

        # Find the lowest id number not used in the STEP file
        counter = 1
        while(True):
            if counter not in line_ids:
                break
            counter += 1

        # Generate nodes with new ids for coordinate points and extra parameters
        for node in t.nodes:

            children = node.children.copy()

            for child in children:

                # STEP lines will be int, coord points will be float
                # Process ids for additional parameters
                if type(child) is str:

                    # If a node with this label already exists, connect to it
                    if t.getNodeID(child):
                        node.updateChild(child, t.getNodeID(child))

                    # If not, assign the next id number and create a new node
                    else:
                        child_node = TreeNode(counter, child)
                        t.addNode(child_node)
                        node.updateChild(child, counter)
                        line_ids.append(counter)
                        counter += 1

                # Process coordinate points
                if type(child) is float:
                    
                    # If this value has been seen before, connect to the node string
                    if child in coords:
                        node.updateChild(child, coords[child])

                    # If not, generate a new node string
                    else:

                        # Format coord point as string with 5dp, starting with + or -
                        coord_str = '{0:.5f}'.format(child)
                        if coord_str[0] != '-':
                            coord_str = '+' + coord_str

                        # Coord as direct string of nodes, one for each digit
                        coord_start = counter
                        for digit in coord_str:
                            digit_node = TreeNode(counter, digit)
                            t.addNode(digit_node)
                            line_ids.append(counter)
                            counter += 1
                        for i in range(coord_start + 1, counter):
                            dn = t.getNode(i)
                            dn.addChild(i - 1)
                        node.updateChild(child, counter - 1)
                        coords[child] = counter - 1

        # Once all nodes are added, format tree
        t.setParents()
        top_node = t.filterNodes()
        t.trim()
        trees.append(t)

    setLabels(trees, labels_file)

    tree_list = []

    # Convert trees into list form and write to data file for later access
    for t in trees:
        tree_list.append(t.ToList())
    with open(data_file, "wb") as fp:
        pickle.dump(tree_list, fp)

    m, s = timeSince(start)
    print("Done. {} files processed in {}m, {}s.".format(num_models,m,round(s)))
    return trees

# Update node labels file containing list of all node labels used in every tree
# Use index of labels in the file to assign category ids to each node
def setLabels(trees, labels_file):
    with open(labels_file, 'a+') as f:
        f.seek(0)
        labels = f.read().split('\n')
        new_labels = []

        for tree in trees:
            for node in tree.nodes:
                new_labels.append(node.category)
        unique_labels = list(set(new_labels))
        unique_labels.sort()
        
        for l in unique_labels:
            if l not in labels:
                f.write(l + '\n')
                labels.append(l)

        for tree in trees:
            for node in tree.nodes:
                node.setCatIndex(labels.index(node.category))

# Return list of node labels for the dataset
def getLabels(labels_file):
    with open(labels_file, 'r') as f:
        return f.read().split('\n')

# Return one-hot vector given length and index of high value
def getWordVector(vec_len, index):
    #word_vec = torch.zeros(1, vec_len).cuda()
    word_vec = torch.zeros(1, vec_len)
    word_vec[0][index] = 1
    return word_vec

# Return embedding matrix containing one-hot vectors for each node label
def getEmbeddings():
    labels_file = os.path.join(glovar.DATA_DIR, 'NODE_LABELS.txt')
    labels = getLabels(labels_file)
    #embedding_matrix = torch.zeros(0, len(labels)).cuda()
    embedding_matrix = torch.zeros(0, len(labels))
    for i in range(len(labels)):
        embedding_matrix = torch.cat([embedding_matrix, getWordVector(len(labels), i)], 0)
    return embedding_matrix


# Simplified 3d plot of a model, containing only coordinate points
def plotCoords(step_file):
    step_lines = readFile(step_file)
    x_coords = []
    y_coords = []
    z_coords = []
    vertex_points = []
    vertex_ids = []

    for line in step_lines:
        if 'VERTEX_POINT' in line:
            vertex_id = line.split('=')[0].strip()
            children = line.replace(')', '(').split('(')[1]
            child_id = children.split(',')[1].strip()
            vertex_points.append([vertex_id, child_id])

    for line in step_lines:
        if 'CARTESIAN_POINT' in line:
            point_id = line.split('=')[0].strip()
            split_line = line.replace(')', '(').split('(')
            point_str = split_line[2].replace(' ', '').split(',')
            coord = [float(c) for c in point_str]
            for v in vertex_points:
                if v[1] == point_id:
                    x_coords.append(coord[0])
                    y_coords.append(coord[1])
                    z_coords.append(coord[2])
                    vertex_ids.append(v[0])

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(x_coords, y_coords, z_coords)
    plt.show()
