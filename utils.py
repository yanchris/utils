import sys
sys.path.append('..')
from md.market_data import MDClient
from model.gen_history import gen_model_data

import requests
import itertools
import pandas as pd
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
from plotly.subplots import make_subplots

from dash.dependencies import Input, Output

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

date_from = '2020-02-03'
date_to = '2020-12-30'

sm_symbols = ['SM2101', 'SM2105']
rb_symbols = ['RB2010', 'RB2101', 'RB2105']
i_symbols = ['I2101', 'I2105']
ma_symbols = ['MA2101', 'MA2105']


black_serial_name_mapping = {
    'SM0': '锰硅连续',
    'RB0': '螺纹钢连续',
    'ZC0': '动力煤连续',
    'I0': '铁矿石连续',
    'J0': '焦炭连续',
    'JM0': '焦煤连续',
    'SF0': '硅铁连续',
}
black_symbols = ['SM0', 'RB0', 'ZC0', 'I0', 'J0', 'JM0', 'SF0']
colours = {
    'SM0': '#0000FF',
    'RB0': '#00FF00',
    'ZC0': '#FF00FF',
    'I0': '#800080',
    'J0': '#00FFFF',
    'JM0': '#008080',
    'SF0': '#000080',
    }

def query_avg_data(symbols, date_from, date_to):
    if not symbols:
        return pd.DataFrame()

    para = {'symbol':symbols, 'start':date_from, 'end':date_to}
    r = requests.get('http://159.65.210.22:8869/avg', params=para)
    data = pd.DataFrame(r.json())
    data['date'] = pd.to_datetime(data['timestamp'], unit='s').dt.strftime('%Y-%m-%d')


    df = pd.concat([
        data[['symbol']],
        data[['price']],
        data[['date']]
    ], axis=1)

    df = df.drop_duplicates(subset=['symbol', 'date'], keep='last')

    df.index = df['date']

    df = pd.concat([df.loc[df['symbol'] == i, 'price'] for i in symbols],
                   axis=1, keys=symbols)

    df.sort_index(inplace=True, ascending=False)

    #df = df.dropna()
    return df

def add_diff_col(df):
    print(df.columns.values)
    columns = sorted(df.columns.values)
    for near, far in filter(lambda x: x[0]!=x[1], itertools.combinations_with_replacement(columns, 2)):
        key = f'{near[-2:]}/{far[-2:]}'
        df[key] = df[near] - df[far]
    return df

def gen_sm_df():
    df = query_avg_data(sm_symbols, date_from, date_to)
    #df['09/01'] = df['SM2009'] - df['SM2101']
    add_diff_col(df)

    return df

def gen_rb_df():
    df = query_avg_data(rb_symbols, date_from, date_to)
    add_diff_col(df)
    #df['10/01'] = df['RB2010'] - df['RB2101']

    return df

def gen_i_df():
    df = query_avg_data(i_symbols, date_from, date_to)
    add_diff_col(df)
    #df['09/10'] = df['I2009'] - df['I2010']
    #df['09/01'] = df['I2009'] - df['I2101']
    #df['10/01'] = df['I2010'] - df['I2101']

    return df

def gen_0509_df():
    i_0509 = ['I2005', 'I2009', 'SM2005', 'SM2009', 'RB2005', 'RB2009']
    date_from = '2020-03-01'
    date_to = '2020-05-01'
    df = query_avg_data(i_0509, date_from, date_to)
    df['I05/09'] = df['I2005'] - df['I2009']
    df['SM05/09'] = df['SM2005'] - df['SM2009']
    df['RB05/09'] = df['RB2005'] - df['RB2009']

    return df[['I2005', 'I2009', 'I05/09', 'SM2005', 'SM2009', 'SM05/09', 'RB2005', 'RB2009', 'RB05/09']], df.loc[:, ['I2005', 'I2009', 'I05/09']], df.loc[:, ['SM2005', 'SM2009', 'SM05/09']], df.loc[:, ['RB2005', 'RB2009', 'RB05/09']]

def gen_6month_df():
    sm_1901_1905 = ['SM1901', 'SM1905']
    date_from = '2018-06-01'
    date_to = '2019-01-31'
    df_1901_1905 = query_avg_data(sm_1901_1905, date_from, date_to)
    df_1901_1905 = df_1901_1905.dropna()
    add_diff_col(df_1901_1905)

    sm_1905_1909 = ['SM1905', 'SM1909']
    date_from = '2018-12-01'
    date_to = '2019-06-01'
    df_1905_1909 = query_avg_data(sm_1905_1909, date_from, date_to)
    df_1905_1909 = df_1905_1909.dropna()
    add_diff_col(df_1905_1909)

    sm_1909_2001 = ['SM1909', 'SM2001']
    date_from = '2019-03-01'
    date_to = '2019-10-01'
    df_1909_2001 = query_avg_data(sm_1909_2001, date_from, date_to)
    df_1909_2001 = df_1909_2001.dropna()
    add_diff_col(df_1909_2001)

    sm_2001_2005 = ['SM2001', 'SM2005']
    date_from = '2019-06-01'
    date_to = '2020-02-01'
    df_2001_2005 = query_avg_data(sm_2001_2005, date_from, date_to)
    df_2001_2005 = df_2001_2005.dropna()
    add_diff_col(df_2001_2005)

    sm_2005_2009 = ['SM2005', 'SM2009']
    date_from = '2019-12-01'
    date_to = '2020-06-01'
    df_2005_2009 = query_avg_data(sm_2005_2009, date_from, date_to)
    df_2005_2009 = df_2005_2009.dropna()
    add_diff_col(df_2005_2009)

    sm_2009_2101 = ['SM2009', 'SM2101']
    date_from = '2020-03-01'
    date_to = '2020-10-01'
    df_2009_2101 = query_avg_data(sm_2009_2101, date_from, date_to)
    df_2009_2101 = df_2009_2101.dropna()
    add_diff_col(df_2009_2101)


    return df_1901_1905, df_1905_1909, df_1909_2001, df_2001_2005, df_2005_2009, df_2009_2101


def gen_ma_df():
    df = query_avg_data(ma_symbols, date_from, date_to)
    add_diff_col(df)
    #df['09/01'] = df['MA2009'] - df['MA2101']
    #df['09/05'] = df['MA2009'] - df['MA2105']
    #df['01/05'] = df['MA2101'] - df['MA2105']

    return df

def gen_black_df(symbols=black_symbols):
    df = query_avg_data(symbols, date_from, date_to)

    return df

def generate_table(dataframe):
    return html.Table([
        html.Thead(
            html.Tr([html.Th('date')] + [html.Th(black_serial_name_mapping[col]) if col in black_serial_name_mapping else html.Th(col) for col in dataframe.columns])
        ),
        html.Tbody([
            html.Tr([html.Td(dataframe.index[i])] + [
                html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
            ]) for i in range(len(dataframe))
        ])
    ])

def gen_all_chart(df, name):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,vertical_spacing=0.02,horizontal_spacing=0.02)
    fig['layout']['margin'] = {'l': 30, 'r': 10, 'b': 50, 't': 25}
    for col in df.columns.values:
        if len(col) > 3 and '/' in col:
            fig.append_trace({'x':df.index,'y':df[col],'type':'bar','name':col},2,1)
        else:
            fig.append_trace({'x':df.index,'y':df[col],'type':'scatter','name':col,'mode': 'markers+lines'},1,1)

    return dcc.Graph(
        id = name,
        figure = fig
    )

def gen_diff_chart(df, name):
    fig = make_subplots(rows=1, cols=1, shared_xaxes=True,vertical_spacing=0.02,horizontal_spacing=0.02)
    fig['layout']['margin'] = {'l': 30, 'r': 10, 'b': 50, 't': 25}
    fig['layout'].update(title=name)
    for col in df.columns.values:
        if '/' in col:
            fig.append_trace({'x':df.index,'y':df[col],'type':'scatter','name':col,'mode': 'markers+lines'},1,1)

    return dcc.Graph(
        id = name,
        figure = fig
    )

def gen_model_chart():
    fig = make_subplots( rows=2, cols=2, vertical_spacing=0.12,horizontal_spacing=0.02, specs=[[{"type": "domain"}, {"type": "domain"}], [{"type": "domain"}, {"type": "domain"}]],
            subplot_titles=("预测区间分布","预测上限区间分布", "预测下限区间分布", "预测在区间内成功率"))
    fig['layout']['margin'] = {'l': 30, 'r': 10, 'b': 50, 't': 100}
    df = gen_model_data()

    prange = df['range'].value_counts(normalize=True) * 100
    pup = df['up'].value_counts(normalize=True) * 100
    plow = df['low'].value_counts(normalize=True) * 100
    pcorrect = df['correct'].value_counts(normalize=True) * 100


    labels = ['Oxygen','Hydrogen','Carbon_Dioxide','Nitrogen']
    values = [4500, 2500, 1053, 500]

    pie1 = go.Pie(labels=prange.index.tolist(), values=prange.values.tolist())
    pie2 = go.Pie(labels=pup.index.tolist(), values=pup.values.tolist())
    pie3 = go.Pie(labels=plow.index.tolist(), values=plow.values.tolist())
    pie4 = go.Pie(labels=pcorrect.index.tolist(), values=pcorrect.values.tolist())

    fig.append_trace(pie1, row=1, col=1)
    fig.append_trace(pie2, row=1, col=2)
    fig.append_trace(pie3, row=2, col=1)
    fig.append_trace(pie4, row=2, col=2)

    fig.update_layout(height=900, showlegend=True)

    return dcc.Graph(
        id = 'model_pie',
        figure = fig
    )

def sm_all_chart(df):
#    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,vertical_spacing=0.02,horizontal_spacing=0.02)
#    fig['layout']['margin'] = {'l': 30, 'r': 10, 'b': 50, 't': 25}
#
#    fig.append_trace({'x':df.index,'y':df.SM2009,'type':'scatter','name':'SM2009','mode': 'markers+lines'},1,1)
#    fig.append_trace({'x':df.index,'y':df.SM2101,'type':'scatter','name':'SM2101','mode': 'markers+lines'},1,1)
#    fig.append_trace({'x':df.index,'y':df['09/01'],'type':'bar','name':'09/01'},2,1)
#    fig['layout'].update(title='')
#
#    return dcc.Graph(
#        id = 'sm_all_in_one',
#        figure = fig
#    )
    return gen_all_chart(df, 'sm_all_in_one')

def rb_all_chart(df):
#    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,vertical_spacing=0.02,horizontal_spacing=0.02)
#    fig['layout']['margin'] = {'l': 30, 'r': 10, 'b': 50, 't': 25}
#
#    fig.append_trace({'x':df.index,'y':df.RB2010,'type':'scatter','name':'RB2010','mode': 'markers+lines'},1,1)
#    fig.append_trace({'x':df.index,'y':df.RB2101,'type':'scatter','name':'RB2101','mode': 'markers+lines'},1,1)
#    fig.append_trace({'x':df.index,'y':df['10/01'],'type':'bar','name':'10/01'},2,1)
#    fig['layout'].update(title='')
#
#    return dcc.Graph(
#        id = 'rb_all_in_one',
#        figure = fig
#    )
    return gen_all_chart(df, 'rb_all_in_one')

def i_all_chart(df):
#    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,vertical_spacing=0.02,horizontal_spacing=0.02)
#    fig['layout']['margin'] = {'l': 30, 'r': 10, 'b': 50, 't': 25}
#
#    fig.append_trace({'x':df.index,'y':df.I2009,'type':'scatter','name':'I2009','mode': 'markers+lines'},1,1)
#    fig.append_trace({'x':df.index,'y':df.I2010,'type':'scatter','name':'I2010','mode': 'markers+lines'},1,1)
#    fig.append_trace({'x':df.index,'y':df.I2101,'type':'scatter','name':'I2101','mode': 'markers+lines'},1,1)
#    fig.append_trace({'x':df.index,'y':df['09/10'],'type':'bar','name':'09/10'},2,1)
#    fig.append_trace({'x':df.index,'y':df['09/01'],'type':'bar','name':'09/01'},2,1)
#    fig.append_trace({'x':df.index,'y':df['10/01'],'type':'bar','name':'10/01'},2,1)
#    fig['layout'].update(title='')
#
#
#    return dcc.Graph(
#        id = 'i_all_in_one',
#        figure = fig
#    )
    return gen_all_chart(df, 'i_all_in_one')

def ma_all_chart(df):
#    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,vertical_spacing=0.02,horizontal_spacing=0.02)
#    fig['layout']['margin'] = {'l': 30, 'r': 10, 'b': 50, 't': 25}
#
#    fig.append_trace({'x':df.index,'y':df.MA2009,'type':'scatter','name':'MA2009','mode': 'markers+lines'},1,1)
#    fig.append_trace({'x':df.index,'y':df.MA2101,'type':'scatter','name':'MA2101','mode': 'markers+lines'},1,1)
#    fig.append_trace({'x':df.index,'y':df.MA2105,'type':'scatter','name':'MA2105','mode': 'markers+lines'},1,1)
#    fig.append_trace({'x':df.index,'y':df['09/01'],'type':'bar','name':'09/01'},2,1)
#    fig.append_trace({'x':df.index,'y':df['09/05'],'type':'bar','name':'09/05'},2,1)
#    fig.append_trace({'x':df.index,'y':df['01/05'],'type':'bar','name':'01/05'},2,1)
#    fig['layout'].update(title='')
#
#    return dcc.Graph(
#        id = 'ma_all_in_one',
#        figure = fig
#    )
    return gen_all_chart(df, 'ma_all_in_one')

def all_chart_0509(df, key):
#    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,vertical_spacing=0.02,horizontal_spacing=0.02)
#    fig['layout']['margin'] = {'l': 30, 'r': 10, 'b': 50, 't': 25}
#
#    fig.append_trace({'x':df.index,'y':df.I2009,'type':'scatter','name':'I2009','mode': 'markers+lines'},1,1)
#    fig.append_trace({'x':df.index,'y':df.I2010,'type':'scatter','name':'I2010','mode': 'markers+lines'},1,1)
#    fig.append_trace({'x':df.index,'y':df.I2101,'type':'scatter','name':'I2101','mode': 'markers+lines'},1,1)
#    fig.append_trace({'x':df.index,'y':df['09/10'],'type':'bar','name':'09/10'},2,1)
#    fig.append_trace({'x':df.index,'y':df['09/01'],'type':'bar','name':'09/01'},2,1)
#    fig.append_trace({'x':df.index,'y':df['10/01'],'type':'bar','name':'10/01'},2,1)
#    fig['layout'].update(title='')
#
#
#    return dcc.Graph(
#        id = 'i_all_in_one',
#        figure = fig
#    )
    return gen_all_chart(df, key)

def all_chart_6m(df, key):
    return gen_all_chart(df, key)

#def sm_all_scatter_fill_chart(df):
#    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,vertical_spacing=0.02,horizontal_spacing=0.02)
#    fig['layout']['margin'] = {'l': 30, 'r': 10, 'b': 50, 't': 25}
#
#    fig.append_trace({'x':df.index,'y':df.SM2009,'type':'scatter','name':'SM2009','mode': 'markers+lines'},1,1)
#    fig.append_trace({'x':df.index,'y':df.SM2101,'type':'scatter','name':'SM2101','mode': 'markers+lines'},1,1)
#    fig.append_trace({'x':df.index,'y':df['09/01'],'type':'scatter','name':'09/01', 'fill': 'tonexty'},2,1)
#    fig['layout'].update(title='')
#
#    return dcc.Graph(
#        id = 'all_in_one_area',
#        figure = fig
#    )
#
#def sm_all_scatter_chart(df):
#    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,vertical_spacing=0.02,horizontal_spacing=0.02)
#    fig['layout']['margin'] = {'l': 30, 'r': 10, 'b': 50, 't': 25}
#
#    fig.append_trace({'x':df.index,'y':df.SM2009,'type':'scatter','name':'SM2009','mode': 'markers+lines'},1,1)
#    fig.append_trace({'x':df.index,'y':df.SM2101,'type':'scatter','name':'SM2101','mode': 'markers+lines'},1,1)
#    fig.append_trace({'x':df.index,'y':df['09/01'],'type':'scatter','name':'09/01','mode': 'markers+lines'},2,1)
#    fig['layout'].update(title='Scatter&Scatter chart')
#
#    return dcc.Graph(
#        id = 'all_in_one_scatter',
#        figure = fig
#    )


#def sm_price_chart(dataframe):
#    scatters = [
#        go.Scatter(
#            x = dataframe.index,
#            y = dataframe.loc[:,rowname],
#            mode = 'markers+lines',
#            name = rowname ) for rowname in sm_symbols
#    ]
#
#    return dcc.Graph(
#        id = 'scatter',
#        figure = {
#            'data': scatters,
#            'layout': go.Layout(
#                title = "硅锰合约差值表",
#                xaxis = {'title': '日期'},
#                yaxis = {'title': '价格'},
#            )
#        }
#    )

#def sm_diff_chart(dataframe):
#    hist = [
#        go.Bar(
#            x = dataframe.index,
#            y = dataframe.loc[:,rowname],
#            name = rowname
#            ) for rowname in ['09/01']
#    ]
#
#    return dcc.Graph(
#        id = 'scatter',
#        figure = {
#            'data': hist,
#            'layout': go.Layout(
#                title = "硅锰合约差值表",
#                xaxis = {'title': '日期'},
#                yaxis = {'title': '差值'},
#            )
#        }
#    )



def generate_black_figure(black_symbols):
    ystep = 0.04
    df = gen_black_df(black_symbols)
    #fig = make_subplots(rows=1, cols=1, shared_xaxes=True,vertical_spacing=0.02,horizontal_spacing=0.02)
    fig = go.Figure()
    #fig['layout']['margin'] = {'l': 30, 'r': 10, 'b': 50, 't': 25}
    ypos = 1 - len(black_symbols) * ystep
    #https://www.rapidtables.com/web/color/RGB_Color.html

    layout_prop = {
        'title': "黑色系收盘价",
        'xaxis': {'title': '日期', 'domain': [0, ypos]},
    }

    y_num = None
    for idx, symbol in enumerate(black_symbols):
        if idx == 0:
            fig.add_trace(go.Scatter(
                **{'x':df.index,'y':df[symbol],'name':black_serial_name_mapping[symbol],'mode': 'markers+lines', 'marker':{'color':colours[symbol], 'line':{'color':colours[symbol]}} }
                )
            )
            layout_prop['yaxis'] = {'title':f'{black_serial_name_mapping[symbol]}收盘价 ', 'titlefont':{'color': colours[symbol]}, 'tickfont': {'color': colours[symbol]}}
        else:
            y_num = idx + 1

            fig.add_trace(go.Scatter(
                **{'x':df.index,'y':df[symbol],'name':black_serial_name_mapping[symbol],'mode': 'markers+lines', 'marker':{'color':colours[symbol], 'line':{'color':colours[symbol]}}, 'yaxis':f'y{y_num}'}
                )
            )

            layout_prop[f'yaxis{y_num}'] = {'title':f'{black_serial_name_mapping[symbol]}收盘价', 'side':'right', 'overlaying':'y', 'anchor':"free", 'position':ypos, 'titlefont':{'color': colours[symbol]}, 'tickfont': {'color': colours[symbol]} }
            ypos += ystep

    fig.update_layout(**layout_prop)
    return fig

app = dash.Dash(__name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)

content = html.Div([
    dcc.Tabs(id='tabs', value='tab-1', children=[
        dcc.Tab(label='锰硅差值线', value='tab-1'),
        dcc.Tab(label='螺纹钢差值线', value='tab-2'),
        dcc.Tab(label='铁矿石差值线', value='tab-3'),
        dcc.Tab(label='甲醇差值线', value='tab-4'),
        dcc.Tab(label='黑色系价格线', value='tab-5'),
        dcc.Tab(label='05/09差值线', value='tab-6'),
        dcc.Tab(label='交割月差值线', value='tab-7'),
        dcc.Tab(label='模型数据', value='tab-8'),
    ]),
    html.Div(id='tabs-content')
])

black_checklist = dcc.Checklist(
    id = 'check_list',
    options=[ {'label': black_serial_name_mapping[i], 'value': i} for i in black_symbols],
    value=black_symbols[:3]
)

black_figure = html.Div([
                    dcc.Graph(
                        id='black_fig',
                    )
               ])

@app.callback(Output('tabs-content', 'children'),
              [Input('tabs', 'value')])
def render_content(tab):
    if tab == 'tab-1':
        df = gen_sm_df()
        return html.Div(children=[
            sm_all_chart(df),
            html.H2('锰硅合约表'),
            generate_table(df),
        ])
    elif tab == 'tab-2':
        df = gen_rb_df()
        return html.Div(children=[
            rb_all_chart(df),
            html.H2('螺纹钢合约表'),
            generate_table(df),
        ])
    elif tab == 'tab-3':
        df = gen_i_df()
        return html.Div(children=[
            i_all_chart(df),
            html.H2('铁矿石合约表'),
            generate_table(df),
        ])
    elif tab == 'tab-4':
        df = gen_ma_df()
        return html.Div(children=[
            ma_all_chart(df),
            html.H2('甲醇合约表'),
            generate_table(df),
        ])
    elif tab == 'tab-5':
        df = gen_black_df()
        body = html.Div(children=[
            html.Div([
                dcc.Graph(
                    id='black_fig',
                )
            ]),
            html.H4('显示合约'),
                black_checklist,
                html.H2('黑色系连续合约表'),
                generate_table(df),
        ])

        return body

    elif tab == 'tab-6':
        df, i_df, sm_df, rb_df = gen_0509_df()
        return html.Div(children=[
            all_chart_0509(i_df, 'i_0509'),
            all_chart_0509(sm_df, 'sm_0509'),
            all_chart_0509(rb_df, 'df_0509'),
            html.H2('合约表'),
            generate_table(df),
        ])

    elif tab == 'tab-7':
        df_1901_1905, df_1905_1909, df_1909_2001, df_2001_2005, df_2005_2009, df_2009_2101 = gen_6month_df()
        return html.Div(children=[
            all_chart_6m(df_1901_1905, '1901/1905'),
            all_chart_6m(df_1905_1909, '1905/1909'),
            all_chart_6m(df_1909_2001, '1909/2001'),
            all_chart_6m(df_2001_2005, '2001/2005'),
            all_chart_6m(df_2005_2009, '2005/2009'),
            all_chart_6m(df_2009_2101, '2009/2101'),
        ])
    elif tab == 'tab-8':
        return html.Div(children=[
            html.Div([
                html.P('说明:'),
                html.P('1). "|"在数值的左边表示上限,比如|70表示上限范围在70%-80%的预测结果，比如某条预测结果: “74%概率收盘价低于 xxxx”'),
                html.P('2). "|"在数值的右边边表示下限,比如50|表示下限范围在50%-60%的预测结果，比如某条预测结果: “59%概率收盘价高于 xxxx”'),
                html.P('3). |70 60| 就表示预测结果在区间中，也就是预测消息中的 "7x%概率低于xxxx, 并且60%概率高于xxxx"'),
                html.P('预测区间分布饼图 -- 预测结果在预测消息的分布。|60 60|表示价格在上下限都是6x%的区间内，而单边的结果比如“|70”, 只是预测对了价格低于某个值，但是没有预测对价格高于哪个值'),
                html.P('预测上限区间分布饼图 -- 成功的上限预测。图示中的“|50=19.3%,|60=30.3,|70=24.4%...”, 那么说70%的概率价格低于xxxx的成功率有 19.3%+30.3%+24.4%,以此类推'),
                html.P('预测下限区间分布饼图 -- 和预测上限区间分布饼图类似，说明的是下限预测的成功率'),
                html.P('预测在区间内成功率饼图 -- 价格在区间内和不在区间内的比率'),
                ]),
            gen_model_chart(),
        ])

@app.callback(
    dash.dependencies.Output('black_fig', 'figure'),
    [Input('check_list', 'value')])
def update_graph(chk_value):
    print('check value is:', chk_value)
    return generate_black_figure(chk_value)


app.layout = content

if __name__ == '__main__':
    app.run_server('0.0.0.0', 8868, debug=True)
