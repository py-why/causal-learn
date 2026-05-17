class DSU:
    def __init__(self, N):
        self.root = [i for i in range(N)]
        
    def find(self, k):
        if self.root[k] == k:
            return k
        
        fa = self.find(self.root[k])
        self.root[k] = fa
        
        return fa
    
    def union(self, a, b):
        x = self.find(a)
        y = self.find(b)
        if x != y:
            self.root[y] = x
        return