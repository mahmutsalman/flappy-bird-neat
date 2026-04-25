import math
import random
from math import ceil


# ─────────────────────────────────────────────
# 1. ACTIVATION FUNCTION
# ─────────────────────────────────────────────

def sigmoid(x):
    try:
        return 1.0 / (1.0 + math.exp(-4.9 * x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0


# ─────────────────────────────────────────────
# 2. INNOVATION TRACKER
# ─────────────────────────────────────────────

class InnovationTracker:
    def __init__(self):
        self._map = {}
        self._counter = 0

    def get(self, in_node, out_node):
        key = (in_node, out_node)
        if key not in self._map:
            self._map[key] = self._counter
            self._counter += 1
        return self._map[key]


# ─────────────────────────────────────────────
# 3. GENES
# ─────────────────────────────────────────────

class NodeGene:
    def __init__(self, node_id, node_type):
        self.id = node_id
        self.type = node_type  # 'i', 'h', 'o'

    def copy(self):
        return NodeGene(self.id, self.type)


class ConnGene:
    def __init__(self, inn, out, w, on, inno):
        self.inn = inn
        self.out = out
        self.w = w
        self.on = on
        self.inno = inno

    def copy(self):
        return ConnGene(self.inn, self.out, self.w, self.on, self.inno)


# ─────────────────────────────────────────────
# 4. GENOME
# ─────────────────────────────────────────────

class Genome:
    def __init__(self, n_inputs, n_outputs, tracker):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.tracker = tracker
        self.nodes = []
        self.conns = []
        self.fitness = 0.0

        for i in range(n_inputs):
            self.nodes.append(NodeGene(i, 'i'))
        for i in range(n_outputs):
            self.nodes.append(NodeGene(n_inputs + i, 'o'))

    def initialize(self):
        inputs = [n for n in self.nodes if n.type == 'i']
        outputs = [n for n in self.nodes if n.type == 'o']
        for inp in inputs:
            for out in outputs:
                w = random.uniform(-2.0, 2.0)
                inno = self.tracker.get(inp.id, out.id)
                self.conns.append(ConnGene(inp.id, out.id, w, True, inno))

    def activate(self, inputs):
        vals = {i: inputs[i] for i in range(self.n_inputs)}

        hidden = [n for n in self.nodes if n.type == 'h']
        outputs = [n for n in self.nodes if n.type == 'o']
        process_order = hidden + outputs

        for _ in range(3):
            for node in process_order:
                s = 0.0
                for c in self.conns:
                    if c.on and c.out == node.id and c.inn in vals:
                        s += vals[c.inn] * c.w
                vals[node.id] = sigmoid(s)

        return [vals.get(n.id, 0.0) for n in outputs]

    def _mutate_weights(self):
        for c in self.conns:
            if random.random() < 0.9:
                c.w += random.gauss(0, 0.2)
                c.w = max(-4.0, min(4.0, c.w))
            else:
                c.w = random.uniform(-2.0, 2.0)

    def _mutate_add_connection(self):
        non_outputs = [n for n in self.nodes if n.type != 'o']
        non_inputs = [n for n in self.nodes if n.type != 'i']
        existing = {(c.inn, c.out) for c in self.conns}

        for _ in range(20):
            src = random.choice(non_outputs)
            dst = random.choice(non_inputs)
            if src.id == dst.id:
                continue
            if (src.id, dst.id) in existing:
                continue
            inno = self.tracker.get(src.id, dst.id)
            self.conns.append(ConnGene(src.id, dst.id, random.uniform(-2.0, 2.0), True, inno))
            return

    def _mutate_add_node(self):
        enabled = [c for c in self.conns if c.on]
        if not enabled:
            return
        c = random.choice(enabled)
        c.on = False

        new_id = max(n.id for n in self.nodes) + 1
        new_node = NodeGene(new_id, 'h')
        self.nodes.append(new_node)

        inno1 = self.tracker.get(c.inn, new_id)
        inno2 = self.tracker.get(new_id, c.out)
        self.conns.append(ConnGene(c.inn, new_id, 1.0, True, inno1))
        self.conns.append(ConnGene(new_id, c.out, c.w, True, inno2))

    def mutate(self):
        r = random.random()
        if r < 0.80:
            self._mutate_weights()
        elif r < 0.87:
            self._mutate_add_connection()
        elif r < 0.90:
            self._mutate_add_node()

    def distance(self, other):
        own = {c.inno: c for c in self.conns}
        other_ = {c.inno: c for c in other.conns}

        hi1 = max(own.keys(), default=0)
        hi2 = max(other_.keys(), default=0)
        mx = min(hi1, hi2)

        excess = 0
        disjoint = 0
        weight_diff = 0.0
        matching = 0

        for inno, c in own.items():
            if inno in other_:
                weight_diff += abs(c.w - other_[inno].w)
                matching += 1
            elif inno <= mx:
                disjoint += 1
            else:
                excess += 1

        for inno in other_:
            if inno not in own:
                if inno <= mx:
                    disjoint += 1
                else:
                    excess += 1

        N = max(len(self.conns), len(other.conns), 1)
        avg_w = weight_diff / matching if matching > 0 else 0.0
        return (excess / N) + (disjoint / N) + 0.4 * avg_w

    def crossover(self, other):
        child = Genome(self.n_inputs, self.n_outputs, self.tracker)
        child.nodes = []

        own_ids = {n.id for n in self.nodes}
        for n in self.nodes:
            child.nodes.append(n.copy())
        for n in other.nodes:
            if n.type == 'h' and n.id not in own_ids:
                child.nodes.append(n.copy())

        other_conns = {c.inno: c for c in other.conns}
        for c in self.conns:
            if c.inno in other_conns:
                oc = other_conns[c.inno]
                new_c = c.copy() if random.random() < 0.5 else oc.copy()
                if not c.on or not oc.on:
                    new_c.on = random.random() > 0.75
                child.conns.append(new_c)
            else:
                child.conns.append(c.copy())

        return child

    def clone(self):
        g = Genome(self.n_inputs, self.n_outputs, self.tracker)
        g.nodes = [n.copy() for n in self.nodes]
        g.conns = [c.copy() for c in self.conns]
        g.fitness = self.fitness
        return g


# ─────────────────────────────────────────────
# 5. SPECIES
# ─────────────────────────────────────────────

class Species:
    def __init__(self, rep):
        self.rep = rep
        self.members = [rep]
        self.best = 0.0
        self.staleness = 0
        self.avg = 0.0

    def update_stats(self):
        if not self.members:
            return
        self.avg = sum(m.fitness for m in self.members) / len(self.members)
        cur_best = max(m.fitness for m in self.members)
        if cur_best > self.best:
            self.best = cur_best
            self.staleness = 0
        else:
            self.staleness += 1

    def cull(self):
        self.members.sort(key=lambda g: g.fitness, reverse=True)
        keep = max(1, ceil(len(self.members) / 2))
        self.members = self.members[:keep]

    def spawn(self):
        p1 = random.choice(self.members)
        if len(self.members) > 1 and random.random() < 0.75:
            p2 = random.choice(self.members)
            fitter, weaker = (p1, p2) if p1.fitness >= p2.fitness else (p2, p1)
            child = fitter.crossover(weaker)
        else:
            child = p1.clone()
        child.mutate()
        return child


# ─────────────────────────────────────────────
# 6. POPULATION
# ─────────────────────────────────────────────

COMPATIBILITY_THRESHOLD = 3.0


class Population:
    def __init__(self, size, n_inputs, n_outputs):
        self.size = size
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.generation = 1
        self.best_fitness = 0.0
        self.tracker = InnovationTracker()
        self.species = []

        self.genomes = []
        for _ in range(size):
            g = Genome(n_inputs, n_outputs, self.tracker)
            g.initialize()
            g.mutate()
            self.genomes.append(g)

    def _speciate(self):
        for sp in self.species:
            sp.rep = random.choice(sp.members) if sp.members else sp.rep
            sp.members = []

        for g in self.genomes:
            placed = False
            for sp in self.species:
                if g.distance(sp.rep) < COMPATIBILITY_THRESHOLD:
                    sp.members.append(g)
                    placed = True
                    break
            if not placed:
                self.species.append(Species(g))

        self.species = [sp for sp in self.species if sp.members]

    def evolve(self):
        self._speciate()

        for sp in self.species:
            sp.cull()
            sp.update_stats()

        for g in self.genomes:
            if g.fitness > self.best_fitness:
                self.best_fitness = g.fitness

        if len(self.species) > 1:
            self.species = [
                sp for sp in self.species if sp.staleness < 15
            ]
            if not self.species:
                self.species = self.species[:1]

        total = sum(max(sp.avg, 0.01) for sp in self.species)

        next_gen = []
        for sp in self.species:
            if sp.members:
                elite = max(sp.members, key=lambda g: g.fitness).clone()
                next_gen.append(elite)

        while len(next_gen) < self.size:
            r = random.uniform(0, total)
            chosen = self.species[0]
            for sp in self.species:
                r -= max(sp.avg, 0.01)
                if r <= 0:
                    chosen = sp
                    break
            next_gen.append(chosen.spawn())

        self.genomes = next_gen[:self.size]
        self.generation += 1
