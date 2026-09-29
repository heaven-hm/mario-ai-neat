"""Render five top-scored and two highly connected genomes as SVG diagrams."""

import csv
import hashlib
from collections import defaultdict
from pathlib import Path
from xml.etree import ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
DATABASE = ROOT / "mario_ai_neat.db"
OUTPUT_DIRECTORY = ROOT / "docs" / "images"
SVG = "http://www.w3.org/2000/svg"
ET.register_namespace("", SVG)

SIGNALS = (
    "speed X", "speed Y", "grounded", "size", "power", "enemy dx",
    "enemy dy", "enemy speed", "enemy type", "item visible", "item dx",
    "item dy", "item type", "gap ahead", "enemy near", "bias",
)
ACTIONS = ("run", "jump + run", "retreat", "brake", "hop", "walk")


def element(parent, tag, **attributes):
    return ET.SubElement(parent, f"{{{SVG}}}{tag}", {key.replace("_", "-"): str(value)
                                                   for key, value in attributes.items()})


def label(parent, x, y, value, *, size=14, color="#EAF4FF", weight="normal"):
    item = element(parent, "text", x=x, y=y, fill=color, font_size=size,
                   font_family="Arial, Helvetica, sans-serif", font_weight=weight,
                   dominant_baseline="middle")
    item.text = value


def parse_database():
    with DATABASE.open(newline="", encoding="utf-8") as database_file:
        rows = list(csv.reader(database_file))
    generation = int(rows[0][1])
    fitness = {int(row[1]): float(row[2]) for row in rows if row[0] == "G"}
    genes = defaultdict(list)
    for row in rows:
        if row[0] == "N":
            genes[int(row[1])].append((int(row[2]), int(row[3]), float(row[4]), row[5] == "1"))
    return generation, fitness, genes


def positions(hidden_nodes):
    points = {}
    for input_id in range(1, 170):
        grid_index = input_id - 1
        points[input_id] = (44 + (grid_index % 13) * 13,
                            103 + (grid_index // 13) * 13)
    for input_id in range(170, 186):
        points[input_id] = (335, 73 + (input_id - 170) * 23)
    for hidden_index, node_id in enumerate(hidden_nodes):
        points[node_id] = (695, 235 + (hidden_index - (len(hidden_nodes)-1)/2) * 70)
    for action_index in range(1, 7):
        points[1000000 + action_index] = (955, 106 + (action_index - 1) * 55)
    return points


def draw_genome(genome_index, generation, fitness, all_genes, database_hash):
    enabled = [gene for gene in all_genes if gene[3]]
    hidden_nodes = sorted({node for source, target, _, _ in enabled
                           for node in (source, target) if 185 < node < 1000000})
    points = positions(hidden_nodes)
    used_inputs = {source for source, _, _, _ in enabled if source <= 185}

    root = ET.Element(f"{{{SVG}}}svg", {"viewBox": "0 0 1150 470", "width": "1150",
                                       "height": "470", "role": "img",
                                       "aria-label": f"Saved neural network for genome {genome_index}"})
    element(root, "title").text = f"Genome {genome_index} from Mario AI NEAT generation {generation}"
    element(root, "desc").text = (f"All {len(enabled)} enabled connections, drawn from the "
                                  f"saved database. Teal means a positive weight and coral means negative.")
    element(root, "rect", x=0, y=0, width=1150, height=470, fill="#102D4A")
    element(root, "rect", x=24, y=86, width=199, height=199, rx=8,
            fill="#193B57", stroke="#547B91", stroke_width=1)
    element(root, "rect", x=317, y=56, width=165, height=376, rx=8,
            fill="#193B57", stroke="#547B91", stroke_width=1)

    label(root, 28, 31, f"GENOME {genome_index}", size=23, color="#55E7E2", weight="bold")
    label(root, 260, 31, f"FITNESS {fitness:.2f}   |   {len(enabled)} ENABLED LINKS",
          size=15, weight="bold")
    label(root, 29, 72, "13 × 13 nearby cells", size=13, color="#B8DAE8")
    label(root, 320, 48, "Player and world inputs", size=13, color="#B8DAE8")
    label(root, 663, 65, "Hidden", size=13, color="#B8DAE8")
    label(root, 930, 74, "Actions", size=13, color="#B8DAE8")

    # Draw every enabled gene before nodes and labels so crossings stay behind text.
    for source, target, weight, _ in enabled:
        if source not in points or target not in points:
            raise ValueError(f"Unmapped gene in genome {genome_index}: {source} -> {target}")
        source_x, source_y = points[source]
        target_x, target_y = points[target]
        element(root, "line", x1=source_x, y1=source_y, x2=target_x, y2=target_y,
                stroke="#51D4C9" if weight >= 0 else "#FF766B",
                stroke_width=round(1.1 + min(abs(weight), 3) * 0.75, 2),
                stroke_opacity="0.68", stroke_linecap="round")

    for input_id in range(1, 170):
        x, y = points[input_id]
        color = "#78E5B5" if input_id in used_inputs else "#335672"
        element(root, "rect", x=x-3, y=y-3, width=6, height=6, fill=color)
    center_x, center_y = points[85]
    element(root, "rect", x=center_x-7, y=center_y-7, width=14, height=14,
            fill="none", stroke="#F9CF50", stroke_width=2)

    for input_id in range(170, 186):
        x, y = points[input_id]
        color = "#F9CF50" if input_id == 185 else (
            "#78E5B5" if input_id in used_inputs else "#5A7890")
        element(root, "rect", x=x-5, y=y-5, width=10, height=10, fill=color)
        label(root, x+16, y, SIGNALS[input_id-170], size=13,
              color="#EAF4FF" if input_id in used_inputs else "#91ACBE")

    for node_id in hidden_nodes:
        x, y = points[node_id]
        element(root, "circle", cx=x, cy=y, r=11, fill="#F9CF50",
                stroke="#102D4A", stroke_width=2)
        label(root, x+19, y, str(node_id), size=14, weight="bold")

    for action_index, name in enumerate(ACTIONS, start=1):
        x, y = points[1000000 + action_index]
        element(root, "rect", x=x-8, y=y-8, width=16, height=16, rx=2,
                fill="#79D7D3", stroke="#102D4A", stroke_width=2)
        label(root, x+20, y, name, size=17, weight="bold")

    label(root, 28, 450, "TEAL  + positive weight", size=12, color="#51D4C9")
    label(root, 222, 450, "CORAL  − negative weight", size=12, color="#FF766B")
    label(root, 468, 450, "Bright input = connected; colors do not show live activation",
          size=12, color="#ACC7D6")
    label(root, 1020, 450, f"DB {database_hash[:8]}", size=11, color="#ACC7D6")

    destination = OUTPUT_DIRECTORY / f"network-genome-{genome_index}.svg"
    ET.indent(root, space="  ")
    ET.ElementTree(root).write(destination, encoding="utf-8", xml_declaration=True)
    print(f"{destination.relative_to(ROOT)}: {len(enabled)} enabled links")


def main():
    generation, fitness, genes = parse_database()
    database_hash = hashlib.sha256(DATABASE.read_bytes()).hexdigest()
    best_five = sorted(fitness, key=lambda genome_index: (-fitness[genome_index], genome_index))[:5]
    enabled_counts = {genome_index: sum(gene[3] for gene in genome_genes)
                      for genome_index, genome_genes in genes.items()}
    most_connected = sorted((genome_index for genome_index in fitness
                             if genome_index not in best_five),
                            key=lambda genome_index: (-enabled_counts.get(genome_index, 0),
                                                      -fitness[genome_index], genome_index))[:2]
    selected = best_five + most_connected
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    for genome_index in selected:
        draw_genome(genome_index, generation, fitness[genome_index],
                    genes[genome_index], database_hash)


if __name__ == "__main__":
    main()
