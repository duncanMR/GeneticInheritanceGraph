import itertools

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import tskit
from IPython.display import clear_output
from IPython.display import display
from IPython.display import SVG
from ipywidgets import Button
from ipywidgets import HBox
from ipywidgets import IntSlider
from ipywidgets import Output
from ipywidgets import VBox


def translate_coordinate_along_edge(edge, x, direction):
    if direction == "up":
        if edge.child_left <= x < edge.child_right:
            parent_length = edge.parent_right - edge.parent_left
            child_length = edge.child_right - edge.child_left
            if parent_length == -child_length:
                # inversion
                return edge.parent_left - (x - edge.child_left)
            elif edge.child_left != edge.parent_left:
                # shift
                return x - (edge.child_right - edge.parent_right)
            else:
                # no change
                return x

    elif direction == "down":
        if (
            edge.parent_left <= x < edge.parent_right
            or edge.parent_left > x >= edge.parent_right
        ):
            parent_length = edge.parent_right - edge.parent_left
            child_length = edge.child_right - edge.child_left
            if parent_length == -child_length:
                # inversion
                return edge.child_left + (edge.parent_left - x)
            elif edge.child_left != edge.parent_left:
                # shift
                return x + (edge.child_right - edge.parent_right)
            else:
                # no change
                return x
    else:
        raise ValueError(f"Invalid direction {direction}")


def plot_local_tree(ts, starting_node, tree_type="Static", size=(400, 400)):
    mut_labels = {}
    for mut in ts.mutations():
        mut_labels[mut.id] = ",".join(str(val) for val in mut.metadata["omega"])
    style = (
        ".node > .lab {font-size: 20px; font-family: Arial}"
        ".mut > .lab {font-size: 16px; font-family: Arial}"
        f".n{starting_node} .sym {{fill: blue}}"
        f".node.n{starting_node} > .lab {{fill: blue}}"
    )

    svg = ts.first().draw_svg(
        mutation_labels=mut_labels,
        size=size,
        style=style,
        x_label=f"{tree_type}ally constructed ts",
    )
    display(SVG(svg))


def int_to_letter(index):
    return chr(index + 65)


def letter_to_int(letter):
    return ord(letter.upper()) - 65


def add_mutation(node, tables, x=None):
    if x is None:
        metadata = {"omega": list()}
    else:
        metadata = {"omega": [x]}

    tables.mutations.add_row(
        site=0,
        node=node,
        derived_state="1",
        parent=-1,
        metadata=metadata,
    )


def process_sample_node(u):
    if isinstance(u, str) and len(u) == 1 and u.isalpha():
        return letter_to_int(u)
    elif isinstance(u, int):
        return u
    else:
        return "The variable is not a single alphabetic character."


class NodeExplorer:
    """
    A class for navigating the nodes of a GIG
    """

    def __init__(self, gig, starting_node):
        self.gig = gig
        self.starting_node = starting_node
        self.pi = np.full(self.gig.num_nodes, -1, dtype=int)
        self.omega = [set() for _ in range(self.gig.num_nodes)]
        self.visited_internal = set()
        self.visited_sample = set()
        self.visited_edges = set()
        self.step_data = []

        # Initialise tskit
        tables = tskit.TableCollection(sequence_length=1)
        tables.nodes.metadata_schema = tskit.MetadataSchema.permissive_json()
        tables.mutations.metadata_schema = tskit.MetadataSchema.permissive_json()
        tables.sites.add_row(
            position=0,
            ancestral_state="0",
        )
        for _, row in self.gig.nodes.iterrows():
            tables.nodes.add_row(
                time=row["time"],
                flags=row["flags"],
                individual=row["individual"],
                metadata={"omega": []},
            )
        self.tables = tables

    def update(self, edge, direction, x_in, x_out, capture=False):
        """
        Move along an edge to a different node.
        """

        omega = self.omega
        pi = self.pi
        c = edge.child
        p = edge.parent
        edge_id = edge.id
        u = self.starting_node
        visited_sample = self.visited_sample.copy()
        visited_sample.discard(u)
        visited_internal = self.visited_internal.copy()
        visited_edges = self.visited_edges.copy()
        tables = self.tables.copy()

        if direction == "up":
            n_out = p
            if c == u and len(visited_sample) == 0:
                # We are at the start
                add_mutation(node=u, tables=tables, x=x_in)
        else:
            n_out = c
        omega_out = omega[n_out].union([x_out])

        if pi[c] == -1:
            # Haven't added the edge yet
            tables.edges.add_row(left=0, right=1, parent=p, child=c)
            pi[c] = p

        if capture is True:
            matching_mut = np.where(tables.mutations.node == n_out)[0]
            if len(matching_mut) == 0:
                # Haven't touch the output node yet
                add_mutation(node=n_out, tables=tables, x=None)
            # Update omega values in tree tables
            tables.nodes[n_out] = tables.nodes[n_out].replace(
                metadata={"omega": list(omega_out)}
            )
            mut_index = np.where(tables.mutations.node == n_out)[0][0]
            tables.mutations[mut_index] = tables.mutations[mut_index].replace(
                metadata={"omega": list(omega_out)}
            )
        self.tables = tables

        current_step = {
            "edge": edge,
            "direction": direction,
            "x_in": x_in,
            "x_out": x_out,
            "visited_internal": visited_internal,
            "visited_sample": visited_sample,
            "visited_edges": visited_edges,
            "tables": tables,
        }
        self.step_data.append(current_step)
        self.omega[n_out] = omega_out
        self.pi = pi
        self.visited_edges.add((edge_id, direction))
        if tables.nodes.flags[n_out] == 1:
            self.visited_sample.add(n_out)
        else:
            self.visited_internal.add(n_out)

    def get_step_data(self, step):
        """
        Retrieve the data for a given step in the traversal.
        """
        if step < len(self.step_data):
            return self.step_data[step]
        else:
            raise IndexError("Step index out of range")

    def plot_gig_highlight(
        self,
        step,
        ax,
        G,
        pos,
        sample_nodes,
        edges,
        fontsize=8,
        current_edge_color="black",
        current_node_color="red",
        visited_color="#3f84fc",
    ):
        """
        Given various data from a GIG plot, return graphics elements which
        highlight a particular edge and associated x values.
        """
        step_data = self.get_step_data(step)
        edge = step_data["edge"]
        direction = step_data["direction"]
        visited_internal = list(step_data["visited_internal"])
        visited_sample = list(step_data["visited_sample"])
        visited_edges = list(step_data["visited_edges"])
        x_in = step_data["x_in"]
        x_out = step_data["x_out"]
        p = edge.parent
        c = edge.child
        edge_id = edge.id
        node_shape = "o"

        if direction == "up":
            n_in = c
            n_out = p
        else:
            n_in = p
            n_out = c
            if n_out in sample_nodes:
                node_shape = "s"

        new_elements = []

        def highlight_edge(edge_id, direction, color="black", linewidth=2):
            edge = edges.loc[edge_id]
            p = edge.parent
            c = edge.child
            curve = edges.loc[edge_id, "curve"]
            edge_attributes = {}
            edge_attributes[
                "parent_interval"
            ] = f"[{edge.parent_left}, {edge.parent_right})"
            edge_attributes[
                "child_interval"
            ] = f"[{edge.child_left}, {edge.child_right})"
            label_offset = curve * 10

            if direction == "up":
                arrowstyle = "->"
            else:
                arrowstyle = "<-"

            highlighted_edge = ax.annotate(
                "",
                xy=pos[p],
                xycoords="data",
                xytext=pos[c],
                textcoords="data",
                arrowprops=dict(
                    arrowstyle=arrowstyle,
                    color=color,
                    shrinkA=10,
                    shrinkB=10,
                    patchA=None,
                    patchB=None,
                    connectionstyle=f"arc3,rad={0.2*curve}",
                    linewidth=linewidth,
                ),
            )
            # Add edge label texts
            label1 = ax.text(
                (pos[p][0] - pos[c][0]) * 0.3 + label_offset + pos[c][0],
                (pos[p][1] - pos[c][1]) * 0.3 + pos[c][1],
                edge_attributes["child_interval"],
                ha="center",
                bbox=dict(facecolor="white", edgecolor="none", pad=0.1),
                fontsize=fontsize,
                color=color,
            )
            label2 = ax.text(
                (pos[p][0] - pos[c][0]) * 0.7 + label_offset + pos[c][0],
                (pos[p][1] - pos[c][1]) * 0.7 + pos[c][1],
                edge_attributes["parent_interval"],
                ha="center",
                bbox=dict(facecolor="white", edgecolor="none", pad=0.1),
                fontsize=fontsize,
                color=color,
            )
            new_elements.extend([highlighted_edge, label1, label2])

        highlight_edge(edge_id, direction, color=current_edge_color, linewidth=2)

        for edge_id, direction in visited_edges:
            highlight_edge(edge_id, direction, color=visited_color, linewidth=1)

        for node in visited_internal:
            node_plot = nx.draw_networkx_nodes(
                G,
                pos,
                nodelist=[node],
                ax=ax,
                node_size=500,
                node_color=visited_color,
                node_shape="o",
            )
            new_elements.append(node_plot)

        for node in visited_sample:
            node_plot = nx.draw_networkx_nodes(
                G,
                pos,
                nodelist=[node],
                ax=ax,
                node_size=500,
                node_color=visited_color,
                node_shape="s",
            )
            new_elements.append(node_plot)

        # Add node highlight
        highlighted_node = nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=[n_out],
            ax=ax,
            node_size=500,
            node_color=current_node_color,
            node_shape=node_shape,
        )
        new_elements.append(highlighted_node)

        # Show x values
        text1 = ax.text(
            pos[n_in][0] + 14 + len(str(x_in)) * 3,
            pos[n_in][1],
            f"{x_in}",
            ha="center",
            va="center",
            fontsize=fontsize + 2,
            color=current_edge_color,
        )
        text2 = ax.text(
            pos[n_out][0] + 14 + len(str(x_out)) * 3,
            pos[n_out][1],
            f"{x_out}",
            ha="center",
            va="center",
            fontsize=fontsize + 2,
            color=current_edge_color,
        )
        new_elements.extend([text1, text2])

        return new_elements

    def plot_tree(self, step, size=(400, 400)):
        step_data = self.get_step_data(step)
        tables = step_data["tables"]
        starting_node = self.starting_node
        tables.sort()
        ts = tables.tree_sequence()
        mut_labels = {}
        for mut in ts.mutations():
            mut_labels[mut.id] = ",".join(str(val) for val in mut.metadata["omega"])
        node_labels = {}
        for node in ts.nodes():
            node_labels[node.id] = int_to_letter(node.id)

        style = (
            ".node > .lab {font-size: 20px; font-family: Arial}"
            ".node.leaf > .lab {transform: translateY(14px)}"
            ".mut > .lab {font-size: 20px; font-family: Arial; fill: blue}"
            ".mut .sym {display: none}"
            f".n{starting_node} .sym {{fill: #ff7b29}}"
            f".node.n{starting_node} > .lab {{fill: #ff7b29}}"
        )
        plot = ts.first().draw_svg(
            mutation_labels=mut_labels,
            node_labels=node_labels,
            size=(size[0], size[1] * 1.4),
            style=style,
        )
        return plot


class GIG:
    """
    A GIG is an ARG in which edges are annotated with coordinate
    transform functions. Nodes in a GIG are integers, and represent
    specific ancestral genomes (identical to an ARG). Edges
    represent a genetic inheritance relationship between two
    genomes. Finally, coordinate transforms define what parts
    of a genome were inherited and the effects of any structural
    variants that may have occured.
    """

    def __init__(self, table_group):
        # TODO instantiate the model by reading in some data and filling
        # in the instance variables as required to define the operations
        # below
        self.intervals = table_group.intervals._df()
        self.nodes = table_group.nodes._df()
        self.node_explorer = NodeExplorer(self, 0)
        self.tables = self.node_explorer.tables
        pass

    @property
    def num_nodes(self):
        """
        Return the number of nodes in the GIG
        """
        return len(self.nodes)

    def parents(self, u):
        """
        Return the parents of node u in the graph.
        """
        all_parents = self.intervals[self.intervals.child == u]["parent"]
        return all_parents.drop_duplicates()

    def children(self, u):
        """
        Return the children of node u in the graph.
        """
        all_children = self.intervals[self.intervals.parent == u]["child"]
        return all_children.drop_duplicates()

    def translate_coordinate(self, p, c, x_in, direction, capture=False):
        """
        Transforms a coordinate in one node to a (possibly empty) set of
        coordinates in an adjacent node. If direction is "up", then the
        x value is assumed to be in the coordinate space of the child node;
        if "down", it is assumed to be in the coordinate space of the parent.

        The NodeExplorer class records the inputs and outputs of the
        transformation as a tree is constructed from the graph;
        if capture is True, then the edge between p and c is recorded
        in the edges table of the tree sequence being constructed.
        """
        edge_family = self.intervals.query("parent == @p and child == @c")
        X = []
        for _, edge in edge_family.iterrows():
            x_out = translate_coordinate_along_edge(edge, x_in, direction)
            if x_out is not None:
                self.node_explorer.update(
                    edge=edge,
                    direction=direction,
                    x_in=x_in,
                    x_out=x_out,
                    capture=capture,
                )
                X.append(x_out)

        return set(X)

    def local_tree_top_down(self, u, x):
        """
        Return the local tree for node u at position x in its coordinate space
        as a tuple (pi, omega). Pi is an oriented tree, such that pi[v] is the
        parent of node v, and pi[v] = -1 if u is a root. Omega defines the
        the set of genome coordinate corresponding to x in the coordinate space
        of v for each node v in the tree.
        """
        pi = np.full(self.num_nodes, -1, dtype=int)
        omega = [set() for _ in range(self.num_nodes)]
        self.node_explorer = NodeExplorer(self, u)

        at_root = False
        c = u
        while not at_root:
            at_root = True
            for p in self.parents(c):
                X = self.translate_coordinate(
                    p=p, c=c, x_in=x, direction="up", capture=False
                )
                assert len(X) in [0, 1]
                if len(X) == 1:
                    x = X.pop()
                    c = p
                    at_root = False
                    pi[c] = p
                    break
        omega[-1] = {x}

        # Traverse downwards from the root, updating the tree as we go.
        stack = [(p, x)]
        while len(stack) > 0:
            p, x = stack.pop()
            for c in self.children(p):
                X = self.translate_coordinate(
                    p=p, c=c, x_in=x, direction="down", capture=True
                )
                for x_val in X - omega[c]:
                    omega[c].add(x_val)
                    pi[c] = p
                    stack.append((c, x_val))
        return self.node_explorer.pi, omega, self.node_explorer.tables

    def local_tree_bottom_up(self, u, x, method="bfs"):
        """
        Bottom up traversal with depth-first traversal down siblings
        """
        pi = np.full(self.num_nodes, -1, dtype=int)
        omega = [set() for _ in range(self.num_nodes)]
        self.node_explorer = NodeExplorer(self, u)
        if method == "bfs":
            position = 0
        elif method == "dfs":
            position = -1
        else:
            raise ValueError(f"Invalid method {method}")

        # Recursive helper function for upward traversal
        def traverse_upward(c, x):
            parents = self.parents(c)
            if len(parents) == 0:
                return
            for p in parents:
                X = self.translate_coordinate(
                    p=p, c=c, x_in=x, direction="up", capture=True
                )
                assert len(X) in [0, 1]
                if len(X) == 1:
                    pi[c] = p
                    x = X.pop()
                    traverse_upward(p, x)
                    traverse_downward(p, x)

        # Function for downward traversal of sibling subtrees
        def traverse_downward(p, x):
            stack = [(p, x)]
            while len(stack) > 0:
                p, x = stack.pop(position)
                for c in self.children(p):
                    if pi[c] == -1:
                        X = self.translate_coordinate(
                            p=p, c=c, x_in=x, direction="down", capture=True
                        )
                        omega[c] = omega[c].union(X)
                        for x_val in omega[c]:
                            pi[c] = p
                            stack.append((c, x_val))

        # Start the recursive traversal
        traverse_upward(u, x)
        tables = self.node_explorer.tables.copy()

        return pi, omega, tables

    def local_tree_bottom_up_sibling(self, u, x):
        """
        Bottom up traversal checking sibling subtrees
        """
        pi = np.full(self.num_nodes, -1, dtype=int)
        omega = [set() for _ in range(self.num_nodes)]
        self.node_explorer = NodeExplorer(self, u)

        # Recursive helper function for upward traversal
        def traverse_upward(c, x):
            parents = self.parents(c)
            if len(parents) == 0:
                return
            for p in parents:
                X = self.translate_coordinate(
                    p=p, c=c, x_in=x, direction="up", capture=True
                )
                assert len(X) in [0, 1]
                if len(X) == 1:
                    pi[c] = p
                    omega[p] = omega[p].union(X)
                    x = X.pop()
                    break

            for n in self.children(p):
                if n != c:
                    X = self.translate_coordinate(
                        p=p, c=n, x_in=x, direction="down", capture=True
                    )
                    for x_val in X:
                        if x_val not in omega[n]:
                            omega[n].add(x_val)
                            pi[n] = p
                            traverse_downward(p=n, x=x_val)

            traverse_upward(c=p, x=x)

        # Function for downward traversal of sibling subtrees
        def traverse_downward(p, x):
            for c in self.children(p):
                X = self.translate_coordinate(
                    p=p, c=c, x_in=x, direction="down", capture=True
                )
                for x_val in X:
                    if x_val not in omega[c]:
                        omega[c].add(x_val)
                        pi[c] = p
                        traverse_downward(p=c, x=x_val)

        # Start the recursive traversal
        traverse_upward(u, x)
        tables = self.node_explorer.tables.copy()

        return pi, omega, tables

    def local_tree_tskit(self, u, x):
        """
        Returns a tskit tree sequence object based on the local tree for node
        u at position x.
        """
        pi, omega, tables_dynamic = self.local_tree_bottom_up(u, x)
        tables = tskit.TableCollection(sequence_length=1)
        node_table = tables.nodes
        edge_table = tables.edges
        mutation_table = tables.mutations
        sites_table = tables.sites
        node_table.metadata_schema = tskit.MetadataSchema.permissive_json()
        mutation_table.metadata_schema = tskit.MetadataSchema.permissive_json()

        sites_table.add_row(
            position=0,
            ancestral_state="0",
        )

        for index, row in self.nodes.iterrows():
            omega_val = omega[index]
            node_table.add_row(
                time=row["time"],
                flags=row["flags"],
                individual=row["individual"],
                metadata={"omega": list(omega_val)},
            )
            if omega_val != set():
                mutation_table.add_row(
                    site=0,
                    node=index,
                    derived_state="1",
                    parent=-1,
                    metadata={"omega": list(omega_val)},
                )

        for child, parent in enumerate(pi):
            if parent != -1:
                edge_table.add_row(left=0, right=1, parent=parent, child=child)
        tables.sort()
        ts = tables.tree_sequence()

        tables_dynamic.sort()
        ts_dynamic = tables_dynamic.tree_sequence()
        return ts, ts_dynamic

    def qc_trees(self, u, x, size=(400, 300)):
        ts, ts_dynamic = self.local_tree_tskit(u, x)
        plot_local_tree(ts, u, tree_type="Static", size=size)
        plot_local_tree(ts_dynamic, u, tree_type="Dynamic", size=size)

    def plot_gig(self, starting_node, size=(10, 10), fontsize=8, dpi=100):
        """
        Plot the entire GIG using networkx
        """
        node_color = "#949494"
        nodes = self.nodes
        edges = self.intervals
        edges["parent_interval"] = [
            f"[{e.parent_left}, {e.parent_right})" for e in edges.itertuples()
        ]
        edges["child_interval"] = [
            f"[{e.child_left}, {e.child_right})" for e in edges.itertuples()
        ]
        edges["id"] = range(len(edges))
        edges["curve"] = 0

        # Create directed graph
        G = nx.from_pandas_edgelist(
            edges,
            source="parent",
            target="child",
            edge_attr=True,
            create_using=nx.MultiDiGraph(),
        )
        # Position and draw nodes
        fig, ax = plt.subplots(1, 1, figsize=size, dpi=dpi)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)
        sample_nodes = nodes[nodes["flags"] == 1].index
        other_nodes = nodes[nodes["flags"] == 0].index
        A = nx.nx_agraph.to_agraph(G)
        with A.subgraph(name="samples", rank="max") as s:
            s.add_nodes_from(sample_nodes)
        A.layout(prog="dot")

        pos = {
            n: [float(x) for x in A.get_node(n).attr["pos"].split(",")]
            for n in G.nodes()
        }
        pos = {
            node_id: np.array([-pos[node_id][0], nodes.loc[node_id].time])
            for node_id in range(self.num_nodes)
        }

        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=sample_nodes,
            ax=ax,
            node_size=500,
            node_shape="s",
            node_color=node_color,
        )
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=[starting_node],
            ax=ax,
            node_size=500,
            node_shape="s",
            node_color="#ff7b29",
        )
        nx.draw_networkx_nodes(
            G, pos, nodelist=other_nodes, ax=ax, node_size=500, node_color=node_color
        )
        labels = {node: int_to_letter(node) for node in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels, ax=ax)

        # Draw edges
        for (parent, child), edge_data in itertools.groupby(
            G.edges(data=True), key=lambda x: (x[0], x[1])
        ):
            edge_data = list(edge_data)
            curves = [0] if len(edge_data) == 1 else np.linspace(-1, 1, len(edge_data))
            for edge, curve in zip(edge_data, curves):
                edge_attributes = edge[2]
                edges.loc[edge_attributes["id"], "curve"] = curve
                ax.annotate(
                    "",
                    xy=pos[parent],
                    xycoords="data",
                    xytext=pos[child],
                    textcoords="data",
                    arrowprops=dict(
                        arrowstyle="-",
                        color="0.35",
                        shrinkA=10,
                        shrinkB=10,
                        patchA=None,
                        patchB=None,
                        connectionstyle=f"arc3,rad={0.2*curve}",
                    ),
                )
                label_offset = curve * 10
                ax.text(
                    (pos[parent][0] - pos[child][0]) * 0.3
                    + label_offset
                    + pos[child][0],
                    (pos[parent][1] - pos[child][1]) * 0.3 + pos[child][1],
                    edge_attributes["child_interval"],
                    ha="center",
                    bbox=dict(facecolor="white", edgecolor="none", pad=0.1),
                    fontsize=fontsize,
                    color="0.35",
                )
                ax.text(
                    (pos[parent][0] - pos[child][0]) * 0.7
                    + label_offset
                    + pos[child][0],
                    (pos[parent][1] - pos[child][1]) * 0.7 + pos[child][1],
                    edge_attributes["parent_interval"],
                    ha="center",
                    bbox=dict(facecolor="white", edgecolor="none", pad=0.1),
                    fontsize=fontsize,
                    color="0.35",
                )

        return fig, ax, G, pos, sample_nodes, edges

    def plot_gig_traversal(
        self,
        u,
        x,
        method="bottom_up_bfs",
        gig_size=(600, 700),
        tree_size=(600, 700),
        dpi=100,
        fontsize=8,
    ):
        """
        Plot GIG traversal algorithm, showing how the local tree
        is constructed in the process.
        """
        u = process_sample_node(u)
        size_inches = (gig_size[0] / dpi, gig_size[1] / dpi)

        # Base GIG plot
        fig, ax, G, pos, sample_nodes, edges = self.plot_gig(
            starting_node=u, size=size_inches, dpi=dpi, fontsize=fontsize
        )
        plt.close()

        # Perform traversal
        if method == "bottom_up_bfs":
            self.local_tree_bottom_up(u, x, method="bfs")
        elif method == "bottom_up_dfs":
            self.local_tree_bottom_up(u, x, method="dfs")
        elif method == "bottom_up_sibling":
            self.local_tree_bottom_up_sibling(u, x)
        elif method == "top_down":
            self.local_tree_top_down(u, x)
        else:
            raise ValueError(f"Invalid method {method}")

        # Construct the visualisation
        num_steps = len(self.node_explorer.step_data) - 1
        step_slider = IntSlider(
            value=0, min=0, max=num_steps, step=1, description="Step"
        )
        gig_output = Output()
        tree_output = Output()
        highlight_elements = []

        def on_step_change(change):
            step = change["new"]

            with gig_output:
                clear_output(wait=True)
                for artist in highlight_elements:
                    artist.remove()
                highlight_elements.clear()

                # Assuming the method to extract the relevant step data
                # from node_explorer
                new_elements = self.node_explorer.plot_gig_highlight(
                    step,
                    ax=ax,
                    G=G,
                    pos=pos,
                    sample_nodes=sample_nodes,
                    edges=edges,
                    fontsize=fontsize,
                    current_edge_color="black",
                    current_node_color="red",
                    visited_color="#3f84fc",
                )
                if new_elements is not None:
                    highlight_elements.extend(new_elements)
                display(fig)

            with tree_output:
                clear_output(wait=True)
                svg_tree = self.node_explorer.plot_tree(step, size=tree_size)
                display(SVG(svg_tree))

        step_slider.observe(on_step_change, names="value")

        # Construct the buttons
        left_button = Button(description="<", tooltip="Previous Step")
        right_button = Button(description=">", tooltip="Next Step")

        def on_left_button_clicked(b):
            if step_slider.value > 0:
                step_slider.value -= 1

        def on_right_button_clicked(b):
            if step_slider.value < step_slider.max:
                step_slider.value += 1

        left_button.on_click(on_left_button_clicked)
        right_button.on_click(on_right_button_clicked)

        display(
            VBox(
                [
                    HBox([gig_output, tree_output]),
                    HBox([left_button, step_slider, right_button]),
                ]
            )
        )
        on_step_change({"new": 0})
