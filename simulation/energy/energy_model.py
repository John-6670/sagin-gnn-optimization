def compute_energy(node, load):
    base_power = 1.0
    dynamic = 0.5 * load
    return base_power + dynamic