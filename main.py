import pygame
import random
import math
from neat_core import Population

# ─────────────────────────────────────────────
# WINDOW & LAYOUT CONSTANTS
# ─────────────────────────────────────────────
GAME_W, GAME_H = 550, 720
INFO_W         = 380
WIN_W          = GAME_W + INFO_W
WIN_H          = GAME_H
FPS            = 60
BETWEEN_GEN_MS = 700

# ─────────────────────────────────────────────
# PHYSICS CONSTANTS
# ─────────────────────────────────────────────
GRAVITY    =  0.38
FLAP_VEL   = -8.5
MAX_FALL   =  12.0
PIPE_SPEED =  3.2
PIPE_W     =  56
GAP_SIZE   =  160
GROUND_H   =  34
BIRD_R     =  13
POP_SIZE   =  50
BIRD_X     =  95

# ─────────────────────────────────────────────
# COLOUR PALETTE
# ─────────────────────────────────────────────
C_SKY_TOP    = ( 10,  18,  45)
C_SKY_BOT    = ( 25,  65, 130)
C_GROUND     = (101,  67,  33)
C_GROUND_TOP = (130,  90,  45)
C_PIPE_DARK  = (  5,  72,  55)
C_PIPE_MID   = ( 16, 185, 129)
C_PIPE_LIGHT = (  8, 140, 100)
C_INFO_BG    = ( 12,  17,  32)
C_PANEL_LINE = ( 28,  38,  58)
C_WHITE      = (255, 255, 255)
C_GRAY       = ( 90, 105, 130)
C_ACCENT     = ( 96, 165, 250)
C_GREEN      = ( 52, 211, 153)
C_YELLOW     = (251, 191,  36)
C_PURPLE     = (167, 139, 250)
C_RED        = (252,  90,  90)


# ─────────────────────────────────────────────
# HSL → RGB HELPER
# ─────────────────────────────────────────────

def _hsl_to_rgb(h, s, l):
    if s == 0:
        v = int(l * 255)
        return (v, v, v)

    def hue2rgb(p, q, t):
        t = t % 1.0
        if t < 1/6:
            return p + (q - p) * 6 * t
        if t < 1/2:
            return q
        if t < 2/3:
            return p + (q - p) * (2/3 - t) * 6
        return p

    q = l * (1 + s) if l < 0.5 else l + s - l * s
    p = 2 * l - q
    r = hue2rgb(p, q, h / 360 + 1/3)
    g = hue2rgb(p, q, h / 360)
    b = hue2rgb(p, q, h / 360 - 1/3)
    return (int(r * 255), int(g * 255), int(b * 255))


BIRD_COLORS = [_hsl_to_rgb((i * 137) % 360, 0.85, 0.62) for i in range(POP_SIZE)]


# ─────────────────────────────────────────────
# BACKGROUND RENDERING
# ─────────────────────────────────────────────

def make_sky(w, h):
    surf = pygame.Surface((w, h))
    for y in range(h):
        t = y / h
        r = int(C_SKY_TOP[0] + (C_SKY_BOT[0] - C_SKY_TOP[0]) * t)
        g = int(C_SKY_TOP[1] + (C_SKY_BOT[1] - C_SKY_TOP[1]) * t)
        b = int(C_SKY_TOP[2] + (C_SKY_BOT[2] - C_SKY_TOP[2]) * t)
        pygame.draw.line(surf, (r, g, b), (0, y), (w, y))
    return surf


def make_cloud(r):
    surf = pygame.Surface((r * 4, r * 2), pygame.SRCALPHA)
    color = (255, 255, 255, 45)
    pygame.draw.circle(surf, color, (r * 2, r), r)
    pygame.draw.circle(surf, color, (int(r * 2 + r * 0.9), int(r - r * 0.3)), int(r * 0.7))
    pygame.draw.circle(surf, color, (int(r * 2 - r * 0.9), int(r - r * 0.2)), int(r * 0.65))
    return surf


def _rot_pt(x, y, cos_a, sin_a):
    return (x * cos_a - y * sin_a, x * sin_a + y * cos_a)


# ─────────────────────────────────────────────
# CLASS Bird
# ─────────────────────────────────────────────

class Bird:
    def __init__(self, genome, idx):
        self.genome = genome
        self.idx = idx
        self.x = float(BIRD_X)
        self.y = GAME_H * 0.45
        self.vy = 0.0
        self.alive = True
        self.score = 0
        self.frames = 0
        self.color = BIRD_COLORS[idx % POP_SIZE]
        self.passed_pipes = set()
        self._wing_t = 0.0

    def think(self, pipes):
        next_pipe = None
        for p in pipes:
            if p.x + PIPE_W > self.x - 5:
                next_pipe = p
                break
        if next_pipe is None:
            return
        inputs = [
            self.y / GAME_H,
            self.vy / 15.0,
            (next_pipe.x - self.x) / GAME_W,
            next_pipe.gap_y / GAME_H,
            (next_pipe.gap_y + GAP_SIZE) / GAME_H,
        ]
        output = self.genome.activate(inputs)
        if output[0] > 0.5:
            self.vy = FLAP_VEL

    def update(self):
        self.vy += GRAVITY
        self.vy = max(-12.0, min(MAX_FALL, self.vy))
        self.y += self.vy
        self.frames += 1
        self._wing_t += 0.25

    def check_dead(self, pipes):
        if self.y < BIRD_R or self.y > GAME_H - GROUND_H - BIRD_R:
            self.alive = False
            return
        for p in pipes:
            if (self.x + BIRD_R > p.x and self.x - BIRD_R < p.x + PIPE_W):
                if (self.y - BIRD_R < p.gap_y or self.y + BIRD_R > p.gap_y + GAP_SIZE):
                    self.alive = False
                    return

    def draw(self, surf):
        if not self.alive:
            return

        angle_deg = max(-30, min(45, self.vy * 3))
        angle_rad = math.radians(-angle_deg)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)

        cx, cy = int(self.x), int(self.y)

        # Body
        pygame.draw.circle(surf, self.color, (cx, cy), BIRD_R)
        dark = tuple(max(0, c - 55) for c in self.color)
        pygame.draw.circle(surf, dark, (cx, cy), BIRD_R, 1)

        # Wing (animated)
        wing_off = int(4.5 + 3.5 * math.sin(self._wing_t))
        wx, wy = _rot_pt(-4, wing_off, cos_a, sin_a)
        wing_surf = pygame.Surface((14, 7), pygame.SRCALPHA)
        wing_color = tuple(max(0, c - 30) for c in self.color)
        pygame.draw.ellipse(wing_surf, wing_color, (0, 0, 14, 7))
        wing_pos = (cx + int(wx) - 7, cy + int(wy) - 3)
        surf.blit(wing_surf, wing_pos)

        # Eye white
        ex, ey = _rot_pt(5, -4, cos_a, sin_a)
        pygame.draw.circle(surf, C_WHITE, (cx + int(ex), cy + int(ey)), 5)
        # Pupil
        px, py = _rot_pt(6, -4, cos_a, sin_a)
        pygame.draw.circle(surf, (20, 20, 30), (cx + int(px), cy + int(py)), 3)

        # Beak
        beak_pts = [(11, -1), (18, 1), (11, 3)]
        rotated = [(cx + int(_rot_pt(bx, by, cos_a, sin_a)[0]),
                    cy + int(_rot_pt(bx, by, cos_a, sin_a)[1]))
                   for bx, by in beak_pts]
        pygame.draw.polygon(surf, (245, 158, 11), rotated)


# ─────────────────────────────────────────────
# CLASS Pipe
# ─────────────────────────────────────────────

class Pipe:
    def __init__(self, x):
        self.x = float(x)
        self.gap_y = random.uniform(85, GAME_H - GROUND_H - 85 - GAP_SIZE)

    def update(self):
        self.x -= PIPE_SPEED

    def draw(self, surf):
        px = int(self.x)
        gap_y = int(self.gap_y)

        # Top pipe body
        pygame.draw.rect(surf, C_PIPE_DARK, (px, 0, PIPE_W, gap_y))
        # Top pipe stripe
        pygame.draw.rect(surf, C_PIPE_MID, (px + 5, 0, 8, gap_y))
        # Top pipe cap
        pygame.draw.rect(surf, C_PIPE_LIGHT, (px - 4, gap_y - 16, PIPE_W + 8, 18))

        bot_y = gap_y + GAP_SIZE
        bot_h = GAME_H - GROUND_H - bot_y

        # Bottom pipe body
        pygame.draw.rect(surf, C_PIPE_DARK, (px, bot_y, PIPE_W, bot_h))
        # Bottom pipe stripe
        pygame.draw.rect(surf, C_PIPE_MID, (px + 5, bot_y, 8, bot_h))
        # Bottom pipe cap
        pygame.draw.rect(surf, C_PIPE_LIGHT, (px - 4, bot_y, PIPE_W + 8, 18))


# ─────────────────────────────────────────────
# DRAW INFO PANEL
# ─────────────────────────────────────────────

def draw_info(surf, fonts, pop, alive_count, max_pipes, gen_best):
    big, med, sm = fonts
    panel_x = GAME_W
    panel_rect = pygame.Rect(panel_x, 0, INFO_W, WIN_H)
    pygame.draw.rect(surf, C_INFO_BG, panel_rect)

    y = 18
    # Header
    header = big.render("NEAT × Flappy Bird", True, C_ACCENT)
    surf.blit(header, (panel_x + (INFO_W - header.get_width()) // 2, y))
    y += header.get_height() + 8
    pygame.draw.line(surf, C_PANEL_LINE, (panel_x, y), (panel_x + INFO_W, y), 1)
    y += 10

    stats = [
        ("GENERATION",      str(pop.generation),          C_WHITE),
        ("ALIVE",           f"{alive_count}/{POP_SIZE}",  C_GREEN),
        ("SPECIES",         str(len(pop.species)),         C_WHITE),
        ("BEST FITNESS",    f"{pop.best_fitness:.1f}",     C_YELLOW),
        ("PIPES THIS GEN",  str(max_pipes),                C_PURPLE),
        ("GEN BEST",        f"{gen_best:.1f}",             C_ACCENT),
    ]

    for label, value, color in stats:
        lbl_surf = sm.render(label, True, C_GRAY)
        surf.blit(lbl_surf, (panel_x + 16, y))
        y += lbl_surf.get_height() + 2
        val_surf = med.render(value, True, color)
        surf.blit(val_surf, (panel_x + 16, y))
        y += val_surf.get_height() + 10

    pygame.draw.line(surf, C_PANEL_LINE, (panel_x, y), (panel_x + INFO_W, y), 1)
    y += 10

    sp_label = sm.render("SPECIES DISTRIBUTION", True, C_GRAY)
    surf.blit(sp_label, (panel_x + 16, y))
    y += sp_label.get_height() + 6

    bar_w = INFO_W - 32
    for i, sp in enumerate(pop.species[:9]):
        bar_color = _hsl_to_rgb((i * 55) % 360, 0.75, 0.58)
        fill = max(4, int(bar_w * len(sp.members) / POP_SIZE))
        pygame.draw.rect(surf, C_PANEL_LINE, (panel_x + 16, y, bar_w, 14), border_radius=3)
        pygame.draw.rect(surf, bar_color, (panel_x + 16, y, fill, 14), border_radius=3)
        tag = sm.render(f"S{i+1}: {len(sp.members)}  stale={sp.staleness}", True, C_GRAY)
        surf.blit(tag, (panel_x + 16 + fill + 4, y))
        y += 18

    y = WIN_H - 108
    pygame.draw.line(surf, C_PANEL_LINE, (panel_x, y), (panel_x + INFO_W, y), 1)
    y += 8

    hints = [
        "Bird = neural network",
        "5 inputs → 1 output (flap?)",
        "Mutation adds nodes/connections",
        "Species protect new innovations",
        "ESC quit   SPACE skip gen",
    ]
    for hint in hints:
        h_surf = sm.render(hint, True, C_GRAY)
        surf.blit(h_surf, (panel_x + 16, y))
        y += h_surf.get_height() + 3


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("Flappy Bird — NEAT AI")
    clock = pygame.time.Clock()

    big = pygame.font.SysFont("Courier New", 28, bold=True)
    med = pygame.font.SysFont("Courier New", 15, bold=True)
    sm  = pygame.font.SysFont("Courier New", 12)
    fonts = (big, med, sm)

    sky_surf = make_sky(GAME_W, GAME_H)

    cloud_data = []
    for _ in range(7):
        r = random.randint(22, 40)
        csurf = make_cloud(r)
        cx = random.randint(0, GAME_W)
        cy = random.randint(40, GAME_H // 2)
        speed = random.uniform(0.3, 0.7)
        cloud_data.append([csurf, cx, cy, speed])

    pop = Population(POP_SIZE, 5, 1)

    birds = []
    pipes = []
    max_pipes = 0
    gen_best = 0.0
    ground_scroll = 0.0
    between_gens = False
    between_timer = 0

    def start_generation():
        nonlocal birds, pipes, max_pipes, gen_best
        birds = [Bird(g, i) for i, g in enumerate(pop.genomes)]
        pipes = [Pipe(GAME_W + 70)]
        max_pipes = 0
        gen_best = 0.0

    start_generation()

    running = True
    skip_gen = False

    while running:
        dt = clock.tick(FPS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if event.key == pygame.K_SPACE:
                    skip_gen = True

        if between_gens:
            between_timer -= dt
            if between_timer <= 0:
                between_gens = False
                skip_gen = False
                start_generation()
        else:
            # Spawn new pipe
            if pipes[-1].x < GAME_W - 240:
                pipes.append(Pipe(GAME_W + 20))

            # Remove off-screen pipes
            pipes = [p for p in pipes if p.x > -PIPE_W - 20]

            # Update pipes
            for p in pipes:
                p.update()

            # Update birds
            alive_count = 0
            for bird in birds:
                if not bird.alive:
                    continue
                alive_count += 1
                bird.think(pipes)
                bird.update()
                bird.check_dead(pipes)

                for p in pipes:
                    pid = id(p)
                    if pid not in bird.passed_pipes and p.x + PIPE_W < bird.x:
                        bird.score += 1
                        bird.passed_pipes.add(pid)

                if bird.score > max_pipes:
                    max_pipes = bird.score

                fit = bird.frames / 60.0 + bird.score * 200.0
                if fit > gen_best:
                    gen_best = fit

            all_dead = alive_count == 0

            if all_dead or skip_gen:
                for bird in birds:
                    bird.genome.fitness = bird.frames / 60.0 + bird.score * 200.0
                pop.evolve()
                between_gens = True
                between_timer = BETWEEN_GEN_MS

            # Ground scroll
            ground_scroll = (ground_scroll + PIPE_SPEED) % 60

        # ── DRAW ──
        # 1. Sky gradient
        screen.blit(sky_surf, (0, 0))

        # 2. Clouds
        for c in cloud_data:
            csurf, cx, cy, speed = c
            screen.blit(csurf, (int(cx), cy))
            c[1] -= speed
            if c[1] < -csurf.get_width():
                c[1] = GAME_W + 10

        # 3. Pipes
        for p in pipes:
            p.draw(screen)

        # 4. Birds
        for bird in birds:
            bird.draw(screen)

        # 5. Scrolling ground
        ground_y = GAME_H - GROUND_H
        pygame.draw.rect(screen, C_GROUND, (0, ground_y, GAME_W, GROUND_H))
        pygame.draw.rect(screen, C_GROUND_TOP, (0, ground_y, GAME_W, 5))
        stripe_color = (80, 52, 26)
        for sx in range(-60 + int(ground_scroll), GAME_W, 60):
            pygame.draw.rect(screen, stripe_color, (sx, ground_y + 8, 28, 4))
            pygame.draw.rect(screen, stripe_color, (sx + 14, ground_y + 18, 28, 4))

        # 6. HUD
        alive_now = sum(1 for b in birds if b.alive)
        hud_surf = pygame.Surface((250, 54), pygame.SRCALPHA)
        hud_surf.fill((0, 0, 0, 120))
        screen.blit(hud_surf, (8, 8))
        line1 = med.render(f"Gen {pop.generation}   Alive {alive_now}/{POP_SIZE}", True, C_WHITE)
        line2 = med.render(f"Pipes {max_pipes}   Best {pop.best_fitness:.1f}", True, C_YELLOW)
        screen.blit(line1, (14, 14))
        screen.blit(line2, (14, 34))

        # 7. Between-gens overlay
        if between_gens:
            overlay = pygame.Surface((GAME_W, WIN_H), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 140))
            screen.blit(overlay, (0, 0))
            msg = big.render("Evolving next generation…", True, C_ACCENT)
            screen.blit(msg, ((GAME_W - msg.get_width()) // 2,
                              (WIN_H - msg.get_height()) // 2))

        # Info panel
        draw_info(screen, fonts, pop, alive_now, max_pipes, gen_best)

        # Divider line between game and panel
        pygame.draw.line(screen, C_PANEL_LINE, (GAME_W, 0), (GAME_W, WIN_H), 1)

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
