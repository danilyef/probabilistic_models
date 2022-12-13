import numpy as np
from IPython.display import HTML, display

def print_table(probability_table: np.ndarray, variable_names: str) -> None:
    """
    Prints a probability distribution table.

    Parameters
    ----------
    probability_table : np.ndarray
        The probability distribution table
    variable_names : str
        A string containing the variable names, e.g., 'CDE'.

    Returns
    -------
    None
    """
    
    assert type(probability_table) is np.ndarray, 'probability_table must be a NumPy array'
    assert probability_table.ndim > 0, 'probability_table must be a table'
    assert type(variable_names) is str, 'variable_names must a string'
    assert len(variable_names) == probability_table.ndim, f'Length of variable_names and dimensions of probability_table must be equal. Your FJDT has {probability_table.ndim} dimensions, but the variable name string is of length {len(variable_names)}.'
    
    num_values = np.product(probability_table.shape[1:])
    shapes = probability_table.shape[1:]
    
    html = '<table>'
    for r, s in enumerate(shapes):
        
        html += '<tr>'
        html += '<td> </td>'
        
        if len(shapes[r:]) > 1:
            length = np.product(shapes[r+1:])
        else:
            length = 1
            
        row = ['$' + variable_names[r+1].lower() + '_' + str(i) + '$' for i in range(s) for _ in range(length)]
        row = row * (num_values // len(row))
        
        for c in row:
            html += '<td>{}</td>'.format(c)
        
        html += '</tr>' 
    
    for i, row in enumerate(probability_table):
        row = row.reshape(-1).tolist()
        
        html += '<tr>'
        html += '<td>{}</td>'.format('$' + variable_names[0].lower() + '_' + str(i) + '$')
        for c in row:
            html += '<td>{:.3f}</td>'.format(c)
        html += '</tr>'
    
    html += '</table>'
    
    display(HTML(html))