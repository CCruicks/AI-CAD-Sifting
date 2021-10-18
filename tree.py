import torch

# Class for single STEP node
class TreeNode:

    def __init__(self, node_id, cat):
        self.id = node_id
        self.category = cat
        self.cat_id = -1
        self.parents = []
        self.children = []
        self.done = False
        self.ready = False

    # Return list of attributes for data saving
    def toList(self):
        return [self.id, self.category, self.cat_id, self.parents, self.children]

    # Add parent node to update the connections between nodes
    def addParent(self, parent_id):
        self.parents.append(parent_id)

    # Add child node to update the connections between nodes
    def addChild(self, child_id):
        self.children.append(child_id)

    # Update when a child node's ID changes
    def updateChild(self, old_id, new_id):
        i = self.children.index(old_id)
        self.children[i] = new_id

    # Return node ID if it is a certain category
    def checkLabel(self, label):
        if label == self.category:
            return self.id
        else:
            return False

    # Set category ID
    def setCatIndex(self, index): self.cat_id = index

    # Set node as done during forward pass through recursive structure
    def setDone(self): 
        self.done = True
        self.ready = False

    # Set node as ready during forward pass through recursive structure
    def setReady(self): self.ready = True

    # Reset node between forward passes
    # Nodes with no children are ready to be processed first
    def reset(self):
        self.done = False
        if len(self.children) > 0:
            self.ready = False
        else:
            self.ready = True

# Class for tree structure representing one STEP file
class Tree:

    def __init__(self, name):
        self.name = name
        self.class_ids = [int(x) for x in name.split("_")[1:]]
        self.nodes = []
        self.top_node_id = 0

    # Return tree name and list of nodes in list form for data saving
    def ToList(self):
        tree_list = []
        tree_list.append(self.name)
        for n in self.nodes:
            tree_list.append(n.toList())
        return tree_list

    # Rebuild nodes from list form when reading saved data
    def nodesFromList(self, node_list):
        for node in node_list:
            n = TreeNode(node[0], node[1])
            n.setCatIndex(node[2])
            for p in node[3]:
                n.addParent(p)
            for c in node[4]:
                n.addChild(c)
            self.nodes.append(n)

    # Add new node to the tree
    def addNode(self, node):
        self.nodes.append(node)

    # Return ID of a given category of node if it exists
    def getNodeID(self, label):
        for node in self.nodes:
            if node.checkLabel(label):
                return node.checkLabel(label)
        return False

    # Return the node with a chosen ID if it exists
    def getNode(self, node_id):
        for node in self.nodes:
            if node.id == node_id:
                return node
        return False

    # STEP lines have reference only to the node's children
    # Update the tree structure so nodes also have a list of parents
    def setParents(self):
        for parent_node in self.nodes:
            for child_id in parent_node.children:
                child_node = self.getNode(child_id)
                if parent_node.id not in child_node.parents:
                    child_node.addParent(parent_node.id)

    # Return total number of nodes in the tree
    def countNodes(self): return len(self.nodes)

    # Return the top node of the tree
    # 'Closed shell' node has been chosen as it represents the full model geometry
    def getTopNode(self):
        top_id = self.getNodeID("CLOSED_SHELL")
        return self.getNode(top_id)

    # Return a list of nodes containing a chosen top node
    # and all those it directly depends on
    def getSubTree(self, top_node):
        subtree_nodes = []
        subtree_nodes.append(top_node)
        while(True):
            counter = 0
            for node in subtree_nodes:
                for child_id in node.children:
                    child_node = self.getNode(child_id)
                    if child_node not in subtree_nodes:
                        subtree_nodes.append(child_node)
                        counter += 1
            if counter == 0:
                break
        return subtree_nodes

    # Filter out nodes which are not necessary to build the model geometry
    def filterNodes(self):
        top_node = self.getTopNode()
        self.top_node_id = top_node.id
        self.nodes = self.getSubTree(top_node)
        return top_node

    # Check if a given node is the top node in the tree
    def isTopNode(self, node):
        if node.id == self.top_node_id: return True
        else: return False

    # Remove unecessary nodes and references to them as children/parents
    def trim(self):
        while(True):
            exit = True
            for node in self.nodes:
                delete = True
                for c in node.children.copy():
                    if not self.getNode(c):
                        node.children.remove(c)
                        exit = False
                for p in node.parents.copy():
                    if not self.getNode(p):
                        node.parents.remove(p)
                        exit = False
                if (len(node.parents) > 0 and node.category) or self.isTopNode(node):
                    delete = False
                if delete:
                    self.nodes.remove(node)
                    exit = False
            if exit:
                break

    # Reset all nodes between forward passes
    def resetNodes(self):
        for n in self.nodes: n.reset()

    # Check if a node  has already been processed during forward pass
    def getNodeDone(self, node_id):
        node = self.getNode(node_id)
        return node.done

    # Set a node as ready if all its children have been processed in forward pass
    def setNodeReady(self, node):
        if node.ready: return

        for child_id in node.children:
            if not self.getNodeDone(child_id):
                return
        
        node.setReady()

    # Set all nodes which can be processed next as ready during forward pass
    def setReadyNodes(self):
        for node in self.nodes:
            if not node.done:
                self.setNodeReady(node)

    def getReadyNodes(self, prev_nodes):
        """ Load node info for the next layer of the forward pass
        Args:
            prev_nodes: List of processed nodes from the previous layer
        Returns:
            ready_nodes: List of all nodes ready to be processed next
            child_ids: List of prev_nodes indexes of all children for each ready node
            hold_nodes: List of processed nodes still needed for the next layer
            hold_ids: List of prev_nodes indexes of all hold nodes
    """

        ready_nodes = []
        hold_nodes = []
        hold_ids = []
        child_ids = []

        # Identify nodes to be processed next
        for node in self.nodes:
            if node.ready:
                children = []
                ready_nodes.append(node)
                for child_id in node.children:
                    child = self.getNode(child_id)

                    # Error catching for debug purposes
                    try:
                        children.append(prev_nodes.index(child))
                    except ValueError:
                        print("Child not found.")
                        print("Parent: {} | {}".format(node.id, node.category))
                        print("Child: {} | {}".format(child.id, child.category))
                        print(len(prev_nodes))
                child_ids.append(children)

        # Identify previously processed nodes which will still be needed later
        for node in prev_nodes:
            for parent_id in node.parents:
                    parent = self.getNode(parent_id)
                    if not parent.ready and prev_nodes.index(node) not in hold_ids:
                        hold_nodes.append(node)
                        hold_ids.append(prev_nodes.index(node))
                        if parent in ready_nodes:
                            ready_nodes.remove(parent)

        return ready_nodes, child_ids, hold_nodes, hold_ids

    # Mark nodes processed most recently as done
    def setDoneNodes(self, done_nodes):
        for node in done_nodes:
            node.setDone()

    # Return True and reset all nodes if the whole tree has been processed
    def getTreeDone(self):
        for node in self.nodes:
            if not node.done:
                return False
        self.resetNodes()
        return True
    
    # Return true if no nodes have been processed since reset
    def getTreeReset(self):
        for node in self.nodes:
            if node.done:
                return False
        return True

# Class representing 'forest' containing trees for each STEP file in a batch
class Forest:

    def __init__(self):
        self.trees = []
        self.classes = []

    # Add new tree and the list of feature classes present in it to the forest
    def addTree(self, tree):
        self.trees.append(tree)
        self.classes.append(tree.class_ids)

