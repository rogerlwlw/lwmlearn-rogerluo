# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 16:58:53 2020

@author: roger luo
"""


def hide_all_code():
    '''
    '''
    from IPython.display import HTML
    import IPython.core.display as di
    di.display_html('''
        <script>jQuery(function() 
        {if (jQuery("body.notebook_app").length == 0) 
        { jQuery(".input_area").toggle(); 
        jQuery(".prompt").toggle();}
        });</script>
        ''',
                    raw=True)
    CSS = """#notebook div.output_subarea {max-width:100%;}"""  #changes output_subarea width to 100% (from 100% - 14ex)
    HTML('<style>{}</style>'.format(CSS))

    return


def hide_code_button():
    '''
    '''
    import ipywidgets as widgets
    from IPython.display import display, HTML
    javascript_functions = {False: "hide()", True: "show()"}
    button_descriptions = {False: "Show code", True: "Hide code"}

    def toggle_code(state):
        output_string = "<script>$(\"div.input\").{}</script>"
        output_args = (javascript_functions[state], )
        output = output_string.format(*output_args)
        display(HTML(output))

    def button_action(value):
        state = value.new
        toggle_code(state)
        value.owner.description = button_descriptions[state]

    state = False
    toggle_code(state)
    button = widgets.ToggleButton(state,
                                  description=button_descriptions[state])
    button.observe(button_action, "value")
    display(button)
    return
