import json

with open('app_notebook.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        new_source = []
        for line in cell['source']:
            # _misprice_markers
            line = line.replace('xs.append(a["ks"])', 'xs.append(np.log(a["ks"]))')
            line = line.replace('hovertemplate="%{text}<br>K/S: %{x:.2f}<br>DTE: %{y:.0f}<extra></extra>",', 'hovertemplate="%{text}<br>Log(K/S): %{x:.2f}<br>DTE: %{y:.0f}<extra></extra>",')
            
            # _draw_vol_surface
            line = line.replace('x=x, y=[y[0]] * len(x), z=z[0], mode="lines+markers",', 'x=np.log(x), y=[y[0]] * len(x), z=z[0], mode="lines+markers",')
            line = line.replace('x=x, y=y, z=z, colorscale=colorscale, colorbar=dict(title=title),', 'x=np.log(x), y=y, z=z, colorscale=colorscale, colorbar=dict(title=title),')
            line = line.replace('hovertemplate="K/S: %{x:.2f}<br>DTE: %{y:.0f}d<br>Vol: %{z:.1f}%<extra></extra>",', 'hovertemplate="Log(K/S): %{x:.2f}<br>DTE: %{y:.0f}d<br>Vol: %{z:.1f}%<extra></extra>",')
            
            line = line.replace('xaxis_title="Moneyness (K/S)", yaxis_title="DTE", zaxis_title=title,', 'xaxis_title="Log Moneyness ln(K/S)", yaxis_title="DTE", zaxis_title=title,')
            
            new_source.append(line)
        cell['source'] = new_source

with open('app_notebook.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
