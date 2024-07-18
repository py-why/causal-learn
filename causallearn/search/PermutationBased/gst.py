class GSTNode:

    def __init__(self, tree, add=None, score=None):
        if score is None: score = -tree.score.score_nocache(tree.vertex, [])
        # if score is None: score = -tree.score.score(tree.vertex, [])
        self.tree = tree
        self.add = add
        self.grow_score = score
        self.shrink_score = score
        self.branches = None
        self.remove = None

    def __lt__(self, other):
        return self.grow_score < other.grow_score

    def grow(self, available, parents):
        self.branches = []
        for add in available:
            parents.append(add)
            score = -self.tree.score.score_nocache(self.tree.vertex, parents)
            # score = -self.tree.score.score(self.tree.vertex, parents)
            parents.remove(add)
            branch = GSTNode(self.tree, add, score)
            if score > self.grow_score: self.branches.append(branch)
        self.branches.sort()

    def shrink(self, parents):
        self.remove = []
        while True:
            best = None
            for remove in [parent for parent in parents]:
                parents.remove(remove)
                score = -self.tree.score.score_nocache(self.tree.vertex, parents)
                # score = -self.tree.score.score(self.tree.vertex, parents)
                parents.append(remove)
                if score > self.shrink_score:
                    self.shrink_score = score
                    best = remove
            if best is None: break
            self.remove.append(best)
            parents.remove(best)

    def trace(self, prefix, available, parents):
        if self.branches is None: self.grow(available, parents)
        for branch in self.branches:
            available.remove(branch.add)
            if branch.add in prefix:
                parents.append(branch.add)
                return branch.trace(prefix, available, parents)
        if self.remove is None:
            self.shrink(parents)
            return self.shrink_score
        for remove in self.remove: parents.remove(remove)
        return self.shrink_score


class GST:

    def __init__(self, vertex, score):
        self.vertex = vertex
        self.score = score
        self.root = GSTNode(self)
        self.forbidden = [vertex]
        self.required = []

    def trace(self, prefix, parents=None):
        if parents is None: parents = []
        available = [i for i in range(self.score.data.shape[1]) if i not in self.forbidden]
        return self.root.trace(prefix, available, parents)

    def set_knowledge(self, forbidden, required):
        self.forbidden = forbidden  # not implemented
        self.required = required  # not implemented
        self.reset()

    def reset(self):
        self.root = GSTNode(self)