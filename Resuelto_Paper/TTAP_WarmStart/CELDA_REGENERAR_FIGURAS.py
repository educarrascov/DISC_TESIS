# Sólo visualización: no importa HiGHS ni cambia el runtime del experimento.
from pathlib import Path
from urllib.request import urlopen
import hashlib, importlib.util, sys

PUBLICATION_REVISION = '7cbb2b3636f4daba115dbce906f0359aa6c32566'
PUBLICATION_SHA256 = '2a36a34b0d5e1c87eeb8a3adc257780a21eebd6e454356d705025c0cf55996f6'
_publication_name = 'paper_publication_figures.py'
_publication_candidates = [
    Path.cwd() / _publication_name,
    Path.cwd() / 'TTAP_WarmStart' / _publication_name,
    Path.cwd() / 'Resuelto_Paper' / 'TTAP_WarmStart' / _publication_name,
    Path.cwd().parent / 'TTAP_WarmStart' / _publication_name,
]
_publication_bytes = next((p.read_bytes() for p in _publication_candidates
    if p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest() == PUBLICATION_SHA256), None)
if _publication_bytes is None:
    _publication_url = ('https://raw.githubusercontent.com/educarrascov/DISC_TESIS/'
        + PUBLICATION_REVISION + '/Resuelto_Paper/TTAP_WarmStart/' + _publication_name)
    with urlopen(_publication_url, timeout=60) as response:
        _publication_bytes = response.read()
if hashlib.sha256(_publication_bytes).hexdigest() != PUBLICATION_SHA256:
    raise RuntimeError('La visualización descargada no coincide con la revisión publicada.')
_publication_folder = Path.cwd() / 'ttap_publication_helpers'
_publication_folder.mkdir(exist_ok=True)
_publication_path = _publication_folder / ('paper_publication_' + PUBLICATION_SHA256[:12] + '.py')
_publication_path.write_bytes(_publication_bytes)
_publication_spec = importlib.util.spec_from_file_location('ttap_paper_figures_' + PUBLICATION_SHA256[:12], _publication_path)
ttap_figures = importlib.util.module_from_spec(_publication_spec)
sys.modules[_publication_spec.name] = ttap_figures
_publication_spec.loader.exec_module(ttap_figures)
print('Visualización cargada: sólo lectura de resultados, sin optimizaciones.')

# Ajusta aquí la estética; no se alteran los resultados.
TTAP_STYLE = ttap_figures.style_config(png_dpi=600)
# TTAP_STYLE['png_dpi'] = 2000  # Mismo PDF vectorial; mayor memoria para PNG.
# TTAP_STYLE['colors'].update({'GBStart': '#D55E00', 'GDStart': '#009E73'})

if 'RUN_DIR' not in globals():
    raise RuntimeError('Pega esta celda al final del notebook A o B con RUN_DIR ya definido.')
TTAP_REPLOT = ttap_figures.replot_run(
    RUN_DIR, style=TTAP_STYLE, quality_cohort='Cold', include_budget_figures=True)
from IPython.display import display, Image
print('Figuras y tablas:', TTAP_REPLOT['directory'])
display(TTAP_REPLOT['coverage'])
for _publication_plot in ['quality', 'effort', 'clock']:
    display(Image(filename=str(TTAP_REPLOT['figures'][_publication_plot]['preview'])))
