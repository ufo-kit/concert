"""Prettytable"""
import prettytable


def get_default_table(field_names):
    """Return a prettytable styled for use in the shell. *field_names* is a
    list of table header strings."""
    table = prettytable.PrettyTable(field_names)
    table.border = True
    table.hrules = prettytable.ALL
    table.vertical_char = ' '
    table.junction_char = '-'
    for name in field_names:
        table.align[name] = 'l'
    return table
