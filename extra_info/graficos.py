import matplotlib.pyplot as plt
import pandas as pd


path ="/Users/rsanchis/datarus/www/master/practicas-arf/Moldes_MantenimientoPreventivo"
filename="limpiezas.csv"
limpiezas = pd.read_csv(path+'/data/'+filename, header=0) 


ax=limpiezas.query('0<hora_diff<2500').hora_diff.hist(bins=100)
fig=ax.get_figure()
fig.savefig(path+'/horas_diff_hist.png', dpi=100, bbox_inches='tight')

ax2=limpiezas.query('0<piezas<150').piezas.hist(bins=100)
fig2=ax2.get_figure()
fig2.savefig(path+'/piezas.png', dpi=100, bbox_inches='tight')
