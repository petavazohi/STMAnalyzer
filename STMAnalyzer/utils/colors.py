import distinctipy

def generate_colors(n_spectra):
    colors = distinctipy.get_colors(n_spectra)
    if (1.0, 1.0, 0.0) in colors:
        colors = generate_colors(n_spectra+1)
        if (1.0, 1.0, 0.0) in colors:
            colors.remove((1.0, 1.0, 0.0))
        else:
            colors.pop(0)
    return colors
