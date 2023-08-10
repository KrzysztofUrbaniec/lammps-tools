
ELEMENT_TO_MASS_MAP = {
    'H' : 1.008,
    'O' : 15.999
}

def print_progress_bar(iteration, total, bar_length=50):
    percent = "{:.1f}".format(100 * (iteration / float(total)))
    filled_length = int(bar_length * iteration // total)
    bar = '=' * filled_length + ' ' * (bar_length - filled_length)
    print(f'\r[{bar}] {percent}% Complete', end='', flush=True)