with open('app_notebook_code.py', 'r', encoding='utf-8') as f:
    text = f.read()

text = text.replace('xs.append(a["ks"])', 'xs.append(np.log(a["ks"]))')
text = text.replace('hovertemplate="%{text}<br>K/S: %{x:.2f}<br>DTE: %{y:.0f}<extra></extra>",', 'hovertemplate="%{text}<br>Log(K/S): %{x:.2f}<br>DTE: %{y:.0f}<extra></extra>",')
text = text.replace('x=x, y=[y[0]] * len(x), z=z[0], mode="lines+markers",', 'x=np.log(x), y=[y[0]] * len(x), z=z[0], mode="lines+markers",')
text = text.replace('x=x, y=y, z=z, colorscale=colorscale, colorbar=dict(title=title),', 'x=np.log(x), y=y, z=z, colorscale=colorscale, colorbar=dict(title=title),')
text = text.replace('hovertemplate="K/S: %{x:.2f}<br>DTE: %{y:.0f}d<br>Vol: %{z:.1f}%<extra></extra>",', 'hovertemplate="Log(K/S): %{x:.2f}<br>DTE: %{y:.0f}d<br>Vol: %{z:.1f}%<extra></extra>",')
text = text.replace('xaxis_title="Moneyness (K/S)", yaxis_title="DTE", zaxis_title=title,', 'xaxis_title="Log Moneyness ln(K/S)", yaxis_title="DTE", zaxis_title=title,')

text = text.replace('dict(x=1, y=1, xref="paper", yref="y", text="sticky-strike", showarrow=False, font=dict(color="#8f96a3", size=10), xanchor="right")', 'dict(x=1, y=1, xref="paper", yref="y", text="sticky-strike", showarrow=False, font=dict(color="#27ae60", size=10), xanchor="right")')
text = text.replace('dict(x=1, y=0, xref="paper", yref="y", text="sticky-delta", showarrow=False, font=dict(color="#27ae60", size=10), xanchor="right")', 'dict(x=1, y=0, xref="paper", yref="y", text="sticky-delta", showarrow=False, font=dict(color="#8f96a3", size=10), xanchor="right")')

with open('app_notebook_code.py', 'w', encoding='utf-8') as f:
    f.write(text)
