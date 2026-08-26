with open('app_notebook_code.py', 'r', encoding='utf-8') as f:
    text = f.read()

text = text.replace(
    'dict(x=1, y=1, xref="paper", yref="y", text="sticky-strike", showarrow=False, font=dict(color="#8f96a3", size=10), xanchor="right")',
    'dict(x=1, y=1, xref="paper", yref="y", text="sticky-strike", showarrow=False, font=dict(color="#27ae60", size=10), xanchor="right")'
)
text = text.replace(
    'dict(x=1, y=0, xref="paper", yref="y", text="sticky-delta", showarrow=False, font=dict(color="#27ae60", size=10), xanchor="right")',
    'dict(x=1, y=0, xref="paper", yref="y", text="sticky-delta", showarrow=False, font=dict(color="#8f96a3", size=10), xanchor="right")'
)

with open('app_notebook_code.py', 'w', encoding='utf-8') as f:
    f.write(text)
