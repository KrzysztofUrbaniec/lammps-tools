
ELEMENT_TO_MASS_MAP = {
    'H': 1.008,
    'Li': 4.002,
    'Be': 9.012,
    'B': 10.81,
    'C': 12.011,
    'N': 14.007,
    'O': 15.999,
    'F': 18.998,
    'Ne': 20.180,
    'Na': 22.989,
    'Mg': 24.305,
    'Al': 26.981,
    'Si': 28.085,
    'P': 30.973,
    'S': 32.06,
    'Cl': 35.45,
    'Ar': 39.95,
    'K': 39.098,
    'Ca': 40.078,
    'Sc': 44.955,
    'Ti': 47.867,
    'V': 50.942,
    'Cr': 51.996,
    'Mn': 54.938,
    'Fe': 55.845,
    'Co': 58.933,
    'Ni': 58.693,
    'Cu': 63.546,
    'Zn': 65.38
}

def print_progress_bar(iteration, total, bar_length=50):
    percent = "{:.1f}".format(100 * (iteration / float(total)))
    filled_length = int(bar_length * iteration // total)
    bar = '=' * filled_length + ' ' * (bar_length - filled_length)
    print(f'\r[{bar}] {percent}% Complete', end='', flush=True)