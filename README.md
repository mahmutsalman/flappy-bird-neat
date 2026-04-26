# Flappy Bird — NEAT AI

A Flappy Bird clone where 50 birds learn to play using a from-scratch implementation of the **NEAT** (NeuroEvolution of Augmenting Topologies) algorithm. No ML libraries — just pure Python + Pygame.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue) ![Pygame](https://img.shields.io/badge/Pygame-2.5%2B-green)

---

## What is NEAT?

NEAT evolves both the **weights** and the **structure** of neural networks over generations. Each bird is its own neural network. Birds that fly further survive and reproduce; their offspring inherit and mutate the winning network topology.

Key ideas:
- **Speciation** — similar genomes are grouped so novel structures get time to develop before competing globally
- **Innovation numbers** — track gene history across generations to enable meaningful crossover between different topologies
- **Structural mutations** — add a new node or connection, not just tweak weights

---

## Neural Network: Inputs & Output

Each bird's network receives **5 inputs** each frame:

| Input | Description |
|---|---|
| `bird_y / screen_height` | Normalised vertical position |
| `velocity / 15` | Normalised vertical velocity |
| `(pipe_x − bird_x) / screen_width` | Horizontal distance to next pipe |
| `pipe_gap_top / screen_height` | Normalised top of the gap |
| `pipe_gap_bottom / screen_height` | Normalised bottom of the gap |

**Output**: single value — if > 0.5, the bird flaps.

---

## Fitness Function

```
fitness = frames_survived / 60  +  pipes_cleared × 200
```

Surviving longer contributes, but clearing pipes is weighted heavily to drive useful behaviour.

---

## Project Structure

```
FlappyBird/
├── main.py          # Pygame game loop, rendering, Bird & Pipe classes
├── neat_core.py     # Full NEAT implementation (Genome, Species, Population)
├── requirements.txt
└── visual-neat-*.html   # Presentation slides explaining the algorithm
```

---

## Getting Started

```bash
# 1. Clone
git clone <repo-url>
cd FlappyBird

# 2. Install dependency
pip install -r requirements.txt

# 3. Run
python main.py
```

### Controls

| Key | Action |
|---|---|
| `SPACE` | Skip current generation |
| `ESC` | Quit |

---

## Info Panel

The right-hand panel shows live stats every generation:

- **Generation** — current generation number
- **Alive** — birds still flying out of 50
- **Species** — number of active species
- **Best Fitness** — all-time best fitness score
- **Pipes This Gen** — most pipes cleared this generation
- **Species Distribution** — colour-coded bar chart per species

---

## Implementation Details (`neat_core.py`)

| Component | Description |
|---|---|
| `InnovationTracker` | Global counter assigning unique IDs to new connections/nodes |
| `NodeGene` / `ConnGene` | Gene primitives for nodes and connections |
| `Genome` | Full network: activate, mutate (weights / add-connection / add-node), crossover, clone |
| `Species` | Groups similar genomes; culls bottom half each generation; promotes elites |
| `Population` | Manages speciation, fitness sharing, and generational turnover |

Activation function: `sigmoid(x) = 1 / (1 + e^(−4.9x))`

Compatibility distance: `δ = E/N + D/N + 0.4 × W̄`  
where E = excess genes, D = disjoint genes, N = max genes, W̄ = average weight difference of matching genes.

---

## Presentation Slides

The `visual-neat-*.html` files are standalone browser slides used to explain the algorithm visually:

| File | Topic |
|---|---|
| `visual-neat-overview.html` | High-level overview of NEAT |
| `visual-neat-4steps.html` | The 4-step evolutionary loop |
| `visual-neat-crossover.html` | Crossover with innovation numbers |
