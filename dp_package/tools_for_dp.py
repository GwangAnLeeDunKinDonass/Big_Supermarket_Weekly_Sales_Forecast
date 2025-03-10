#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from tqdm import tqdm
from datetime import datetime
import numpy as np
import pandas as pd
import math
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, message="pandas.Int64Index is deprecated and will be removed from pandas in a future version. Use pandas.Index with the appropriate dtype instead.")

import altair as alt
alt.data_transformers.enable('default', max_rows=None)
import seaborn as sns
import matplotlib.pyplot as plt  

import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.metrics import root_mean_squared_error, r2_score

import datapane as dp

'''
df_join : 보고서 작성용 데이터 작성 코드
Input : {directory: Raw Data가 있는 디렉토리 (str) (Default: './data/walmart')}
Output: Join 및 인코딩 & 표준화를 거친 후, Store_Num에 따라 필터링된 데이터 (pd.DataFrame)
'''
def df_join(directory='./data/walmart'):
    
    
    # 데이터 로드
    walmart_tr=pd.read_csv(f'{directory}/train.csv')
    walmart_st=pd.read_csv(f'{directory}/stores.csv')
    walmart_ft=pd.read_csv(f'{directory}/features.csv')
    
    # 키에 맞춰 JOIN
    walmart_trst=pd.merge(walmart_tr,walmart_st, on='Store', how='left')
    df=pd.merge(walmart_trst,walmart_ft, on=['Store','Date','IsHoliday'], how='inner')
    
    # Date 관련 컬럼 추가
    df["Date"] = pd.to_datetime(df["Date"])
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['Week'] = df['Date'].apply(lambda x: x.isocalendar()[1])
    
    # Label Encoding 한 컬럼 추가
    le1 = LabelEncoder()
    le2 = LabelEncoder()

    df['IsHoliday_le'] = le1.fit_transform(df['IsHoliday'])
    df['Year_le'] = le2.fit_transform(df['Year'])
    
    # Standart Scaling 한 컬럼 추가
    scaler1 = StandardScaler()
    scaler2 = StandardScaler()
    
    scaler1.fit(walmart_st[['Size']])
    df['Size_sd'] = scaler1.transform(df[['Size']])
    
    columns_to_scale = ['Temperature', 'Unemployment','Fuel_Price', 'CPI']
    scaler2.fit(walmart_ft[columns_to_scale])
    col_names = ['Temperature_sd', 'Unemployment_sd','Fuel_Price_sd', 'CPI_sd']
    df[col_names] = scaler2.transform(df[columns_to_scale])
    
    return df

'''
store_summary : Datapane 내 report 상단에 배치될 Store Summary 블록 생성 함수
Input : {df : df_join에서 return된 데이터프레임 (pd.DataFrame),
        year_week : 연, 주로 이루어진 리스트 (list)}
Output : Store Summary 블록, Dept 유니크 리스트 (list)
'''
def store_summary(df, year_week):
    # 불러온 스토어 번호 호출
    store = df.loc[0,'Store']
    
    # 블록에 들어갈 변수 선언
    yearly_sales=round(df.loc[(df['Year']==year_week[0])&(df['Week']<=year_week[1]),'Weekly_Sales'].sum()/1000,2) # 연누계
    # 해당 주에 포함되는 데이터만 추출
    weekly_df=df[(df['Year']==year_week[0]) & (df['Week']==year_week[1])].reset_index(drop=True)
    # weekly_df를 Dept로 그루핑
    dept_sales=weekly_df.groupby(['Dept'],as_index=False)['Weekly_Sales'].sum()
    # 주간 매출이 가장 높은 Dept와 낮은 Dept 추출
    max_dept=int(dept_sales[dept_sales['Weekly_Sales']==max(dept_sales['Weekly_Sales'])]['Dept'])
    min_dept=int(dept_sales[dept_sales['Weekly_Sales']==min(dept_sales['Weekly_Sales'])]['Dept'])
    # 주간 총매출
    weekly_total_sales=round(weekly_df['Weekly_Sales'].sum()/1000,2)
    
    
    # 1주 전에 포함되는 데이터만 추출
    if year_week[1]>1:
        one_weekly_df=df[(df['Year']==year_week[0]) & (df['Week']==year_week[1]-1)].reset_index(drop=True)
    else:
        one_weekly_df=df[(df['Year']==year_week[0]-1) & (df['Week']==52)].reset_index(drop=True)
    # 2주전 주간 총매출
    if len(one_weekly_df)>0:
        one_weekly_total_sales=round(one_weekly_df['Weekly_Sales'].sum()/1000,2)
        week_updown=round((weekly_total_sales-one_weekly_total_sales)/one_weekly_total_sales * 100)
        if week_updown>=0:
            week_updown={'value': f'$ {one_weekly_total_sales:,}',
                         'change': f'{str(abs(week_updown))}%',
                         'is_upward_change' : True}
        else:
            week_updown={'value': f'$ {one_weekly_total_sales:,}',
                         'change': f'{str(abs(week_updown))}%',
                         'is_upward_change' : False}
    else:
        week_updown={'value': '지난 주 정보 없음'}
    
    # 1년 전에 포함되는 데이터만 추출
    last_weekly_df=df[(df['Year']==year_week[0]-1) & (df['Week']==year_week[1])].reset_index(drop=True)
    # 작년 동일 주간 총매출
    if len(last_weekly_df)>0:
        last_weekly_total_sales=round(last_weekly_df['Weekly_Sales'].sum()/1000,2)
        year_updown=round((weekly_total_sales-last_weekly_total_sales)/last_weekly_total_sales * 100)
        if year_updown>=0:
            year_updown={'value': f'$ {last_weekly_total_sales:,}',
                         'change': f'{str(abs(year_updown))}%',
                         'is_upward_change' : True}
        else:
            year_updown={'value': f'$ {last_weekly_total_sales:,}',
                         'change': f'{str(abs(year_updown))}%',
                         'is_upward_change' : False}
    else:
        year_updown={'value': '작년 정보 없음'}
        
    store_summary_block = dp.Group(
                            dp.Group(
                                dp.BigNumber(heading='연누계 (천달러 단위)', value=f'$ {yearly_sales:,}'),
                                dp.BigNumber(heading='Best Dept', value=f'{str(max_dept)}'),
                                dp.BigNumber(heading='Worst Dept', value=f'{str(min_dept)}'),
                                columns=3
                            ),
                            dp.Group(
                                dp.BigNumber(heading='주간 총 매출 (천달러 단위)', value=f'$ {weekly_total_sales:,}'),
                                dp.BigNumber(heading='지난 주 총 매출 (천달러 단위)', **week_updown),
                                dp.BigNumber(heading='작년 동기 총 매출 (천달러 단위)', **year_updown),
                                columns=3
                            )
                        )
    
    # 해당 Store가 가지고 있는 Dept 리스트로 추출
    dept_uniq=sorted(list(dept_sales['Dept'].unique()))
    dept_uniq_copy=dept_uniq[:]
    for dept in dept_uniq_copy:
        fake_df=df[(df['Dept']==dept) & (df['Store']==store)]
        if year_week[1]>=12:
            elder_date = fake_df[(fake_df['Year']==year_week[0])&(fake_df['Week']<=year_week[1])&(fake_df['Week']>=year_week[1]-11)].reset_index(drop=True)
        else:
            elder_date1 = fake_df[(fake_df['Year']==year_week[0])&(fake_df['Week']<=year_week[1])&(fake_df['Week']>=1)]
            elder_date2 = fake_df[(fake_df['Year']==year_week[0]-1)&(fake_df['Week']<=52)&(fake_df['Week']>=52-(11-year_week[1]))]
            elder_date = pd.concat([elder_date2,elder_date1]).reset_index(drop=True)

        if year_week[1]<=44:
            newer_date = fake_df[(fake_df['Year']==year_week[0])&(fake_df['Week']>=year_week[1]+1)&(fake_df['Week']<=year_week[1]+8)].reset_index(drop=True)
        else:
            newer_date1 = fake_df[(fake_df['Year']==year_week[0])&(fake_df['Week']>=year_week[1]+1)&(fake_df['Week']<=52)]
            newer_date2 = fake_df[(fake_df['Year']==year_week[0]+1)&(fake_df['Week']>=1)&(fake_df['Week']<=(year_week[1]-44))]
            newer_date = pd.concat([newer_date1,newer_date2]).reset_index(drop=True)
        
        fake_df = pd.concat([elder_date,newer_date]).reset_index(drop=True)
        if len(fake_df)!=20:
            dept_uniq.remove(dept)
    return store_summary_block, dept_uniq

'''
sales_predict : 해당 시점에서 13~1주전 과거 데이터와 해당 시점+7주 후 미래 plot을 그리기 위한 데이터 프레임 추출 함수
Input : {df : df_join에서 return된 데이터프레임 (pd.DataFrame),
        year_week : 연, 주로 이루어진 리스트 (list),
        dept_num : Dept 번호 (int)}
Output : Plot을 그리기 위한 데이터프레임 (pd.DataFrame)
'''
def sales_predict(df, year_week, dept_num=None):
    # dept_num 인자 받았을 시, 해당 Dept로 필터링
    if dept_num == None:
        df = df.groupby(['Date'],as_index=False).agg({'Store':'min',
                                                       'Weekly_Sales':'sum',
                                                       'Year':'min',
                                                       'Week':'min',
                                                       'IsHoliday_le':'min',
                                                       'Size_sd':'min',
                                                       'Temperature_sd':'min',
                                                       'CPI_sd':'min'})
    else:
        df = df[df['Dept']==dept_num].reset_index(drop=True)
        
    # 입력주부터 11주 전까지의 데이터 추출 > 12주차 이전까지는 작년 데이터도 끌어옴
    if year_week[1]>=12:
        elder_date = df[(df['Year']==year_week[0])&(df['Week']<=year_week[1])&(df['Week']>=year_week[1]-11)].reset_index(drop=True)
    else:
        elder_date1 = df[(df['Year']==year_week[0])&(df['Week']<=year_week[1])&(df['Week']>=1)]
        elder_date2 = df[(df['Year']==year_week[0]-1)&(df['Week']<=52)&(df['Week']>=52-(11-year_week[1]))]
        elder_date = pd.concat([elder_date2,elder_date1]).reset_index(drop=True)
    elder_date['Pred']=0
    elder_date=pd.concat([elder_date,elder_date.loc[[11]]]).reset_index(drop=True)
    elder_date.loc[12,'Pred']=1
    
    # 입력내주부터 8주 후까지의 데이터 추출 > 44주차 이후까지는 내년 데이터도 끌어옴
    if year_week[1]<=44:
        newer_date = df[(df['Year']==year_week[0])&(df['Week']>=year_week[1]+1)&(df['Week']<=year_week[1]+8)].reset_index(drop=True)
    else:
        newer_date1 = df[(df['Year']==year_week[0])&(df['Week']>=year_week[1]+1)&(df['Week']<=52)]
        newer_date2 = df[(df['Year']==year_week[0]+1)&(df['Week']>=1)&(df['Week']<=(year_week[1]-44))]
        newer_date = pd.concat([newer_date1,newer_date2]).reset_index(drop=True)
    newer_date['Pred']=1
    # 예측을 위한 데이터 생성 및 모델 불러오기 & 예측
    df_feature = newer_date[['Store','Dept','Year','Week','IsHoliday_le','Size_sd','Temperature_sd','CPI_sd']]
    
    with open('best_model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    pred = model.predict(df_feature)
        
    # newer_date에 predict값 삽입 후, elder_date와 병합
    newer_date['Weekly_Sales']=pred
    df_for_plot = pd.concat([elder_date,newer_date]).reset_index(drop=True)
    df_for_plot['S']='S'
    return df_for_plot

'''
dept_summary : Datapane 내 report 중단에 배치될 Dept Summary 블록 생성 함수
Input : {df : df_join에서 return된 데이터프레임 (pd.DataFrame),
        year_week : 연, 주로 이루어진 리스트 (list),
        dept_num : Dept 번호 (int)}
Output : Dept Summary 블록
+ Dept Total Summary도 만들어야함
'''
def dept_summary(dept_num, df, year_week):
    # store 금주 예측
    df_s = df[(df['Year']==year_week[0]) & (df['Week']==year_week[1])].reset_index(drop=True)
    df_s = df_s[['Store','Dept','Year','Week','IsHoliday_le','Size_sd','Temperature_sd','CPI_sd']]
    
    with open('best_model.pkl', 'rb') as f:
        model = pickle.load(f)
    s_pred = model.predict(df_s)
    store_pred = s_pred.sum()
    
    # 불러온 Dept로만 필터링
    dept_df=df[df["Dept"]==dept_num].reset_index(drop=True)
    
    # 블록에 들어갈 변수 선언
    yearly_sales=round(dept_df[dept_df['Year']==year_week[0]]['Weekly_Sales'].sum()/1000,2) # 연누계
    # 해당 주에 포함되는 데이터만 추출
    weekly_df=dept_df[(dept_df['Year']==year_week[0]) & (dept_df['Week']==year_week[1])].reset_index(drop=True)
    # 주간 총매출
    weekly_total_sales=round(weekly_df['Weekly_Sales'].sum(),2)
    # 지난주 총매출
    # weekly_df=dept_df[(dept_df['Year']==year_week[0]) & (dept_df['Week']==year_week[1]-2)].reset_index(drop=True)
    
    # 1주 전에 포함되는 데이터만 추출
    if year_week[1]>1:
        one_weekly_df=dept_df[(dept_df['Year']==year_week[0]) & (dept_df['Week']==year_week[1]-1)].reset_index(drop=True)
    else:
        one_weekly_df=dept_df[(dept_df['Year']==year_week[0]-1) & (dept_df['Week']==52)].reset_index(drop=True)
    # 1주전 주간 총매출
    if len(one_weekly_df)>0:
        one_weekly_total_sales=round(one_weekly_df['Weekly_Sales'].sum(),2)
        week_updown=round((weekly_total_sales-one_weekly_total_sales)/one_weekly_total_sales * 100)
        if week_updown>=0:
            week_updown={'value': f'$ {one_weekly_total_sales:,}',
                         'change': f'{str(abs(week_updown))}%',
                         'is_upward_change' : True}
        else:
            week_updown={'value': f'$ {one_weekly_total_sales:,}',
                         'change': f'{str(abs(week_updown))}%',
                         'is_upward_change' : False}
    else:
        week_updown={'value': '지난 주 정보 없음'}
        
    # 1년 전에 포함되는 데이터만 추출
    last_weekly_df=dept_df[(dept_df['Year']==year_week[0]-1) & (dept_df['Week']==year_week[1])].reset_index(drop=True)
    # 작년 동일 주간 총매출
    if len(last_weekly_df)>0:
        last_weekly_total_sales=round(last_weekly_df['Weekly_Sales'].sum(),2)
        year_updown=round((weekly_total_sales-last_weekly_total_sales)/last_weekly_total_sales * 100)
        if year_updown>=0:
            year_updown={'value': f'$ {last_weekly_total_sales:,}',
                         'change': f'{str(abs(year_updown))}%',
                         'is_upward_change' : True}
        else:
            year_updown={'value': f'$ {last_weekly_total_sales:,}',
                         'change': f'{str(abs(year_updown))}%',
                         'is_upward_change' : False}
    else:
        year_updown={'value': '작년 정보 없음'}
    
    # 시각화를 위한 데이터 프레임 준비 후 플롯 그리기
    df_for_plot=sales_predict(df, year_week, dept_num=dept_num)
    
    departure_pred=df_for_plot.loc[12,'Weekly_Sales']

    colors = ['#041E42','#FFC220']
    brush = alt.selection_interval()
    
    line_chart=(alt.Chart(df_for_plot).mark_line().encode(
            x='Date:T',
            y='Weekly_Sales:Q',
            color='Pred:N',
        ).properties(
            width=800,
            height=300,
            title='Weekly Sales and Predictions Over Time'
        ).add_params(brush))
    
    bar_chart = alt.Chart(df_for_plot).mark_bar(color='#041E42').encode(
            x='sum(Weekly_Sales):Q',
            y='sum:N'
        ).properties(
        width=800,
        height=100,
        title='Total Sales and Predictions for Selected Period'
    ).transform_filter(brush)
    
    plot = (line_chart & bar_chart)
    plot= plot.configure_range(category=alt.RangeScheme(colors))
    plot_html = plot.to_html()
    plot_html = f"""
        <div style="display: flex; justify-content: center; align-items: center;">
            <div style="width: 80%; max-width: 1200px;">
                {plot_html}
            </div>
        </div>
        """          

    df_for_table = df_for_plot.drop_duplicates(['Week'])
    df_for_table = df_for_table.copy()
    df_for_table.loc[:, 'Date'] = df_for_table['Date'].dt.strftime("%Y-%m-%d")
    df_for_table = df_for_table[['Date','Week','IsHoliday','Weekly_Sales','MarkDown1','MarkDown2','MarkDown3','MarkDown4','MarkDown5']]

    
    dept_summary_block = dp.Group(
                            dp.Group(
                                dp.BigNumber(heading='주간 총 매출', value=f'$ {weekly_total_sales:,}'),
                                dp.BigNumber(heading='지난 주 총 매출', **week_updown),
                                dp.BigNumber(heading='작년 동기 총 매출', **year_updown),
                                columns=3,
                            ),
                            dp.Group(
                                dp.BigNumber(heading='매장 전체 금주 예측치', value=f'$ {store_pred:,.2f}'),
                                dp.BigNumber(heading=f'{dept_num}번 부서 금주 예측치', value=f'$ {departure_pred:,.2f}'),
                                columns=2                            
                            ),
                            dp.Select(
                                blocks=[
                                    dp.HTML(plot_html, label="Graph"),
                                    dp.DataTable(df_for_table, label="Table")
                                ]
                            ),
                            label=str(dept_num)
                        )
    
    return dept_summary_block

'''
function_iter : Datapane 내 블록을 여러번 만들기 위해 만든 함수
Input : {function : 여러번 돌리고 싶은 함수,
        iteration : 몇 번 돌리고 싶은지 (int, list),
        *args : function의 인자들}
Output : Dept Summary 블록 (list)
'''
def function_iter(function, iteration, *args, **kwargs):
    lst=[]
    print('블록 추가 중')
    # iteration을 정수 형태로 받을 시, 1부터 iteration까지
    if type(iteration)==int:
        for i in tqdm(range(1,iteration+1)):
            lst.append(function(i,*args,**kwargs))
    # iteration을 리스트 형태로 받을 시, 리스트 원소들로 반복
    elif type(iteration)==list:
        for i in tqdm(iteration):
            lst.append(function(i,*args,**kwargs))
    return lst

'''
header_image : Datapane 내 상단 디자인을 만들기 위해 만든 함수
Input : {df : df_join에서 return된 데이터프레임 (pd.DataFrame),
        store_num : Store 번호 (int)
        year_week : 연, 주로 이루어진 리스트 (list),
        }
Output : 상단 블록 구성을 위한 html 언어 (str)
'''
def header_image(df, store_num, year_week):
    year = year_week[0]
    week = year_week[1]
    ishoilday = df.loc[(df['Year']==year)&(df['Week']==week),'IsHoliday_le'].values[0]
    # 슈퍼볼, 노동절, 추수감사절, 크리스마스
    if week == 6:
        text = "슈퍼볼 주간"
    elif week == 36:
        text = "노동절 주간"
    elif week == 47:
        text = "추수감사절 주간"
    elif week == 52:
        text = "크리스마스 주간"
    else:
        text = ""

    if ishoilday == 0:
        color = '#000000'
    else:
        color = '#eb5757'
    header_image = f"""
    <html>
        <style type='text/css'>
            #container {{
                background: #ffffff;
                padding: em;
                text-align: center;
            }}
            h1 {{
                color:#00336d;
                text-align:center;
                font-size:50px;
                font-family:verdana;
                margin-top:10px;
            }}
            span {{
                color:#57aeff;
                text-align:center;
                font-size:25px;
            }}
            #reportdate {{
                color:#000000;
                font-size:20px;
                text-align:center;
            }}
            #holiday {{
                color:{color};
                font-size:20px;
                text-align:center;
            }}
            hr {{
                border: none;
                height: 8px;
                background: #064e89;
                margin-top: 10px;
                margin-bottom: 5px;
            }}
        </style>
        <div id="container">
            <span><b>Walmart Store No.{store_num}</b></span><br>
            <h1> 주간 매출 보고서 </h1>            
            <span id = "reportdate"><b>- {year}년도 {week}주차 결산 -</b></span><br>
            <span id = "holiday"><b>{text}</b></span>
            <hr>
        </div>
    </html>
    """

    return header_image

