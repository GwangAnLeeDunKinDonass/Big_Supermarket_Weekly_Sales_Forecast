#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from datetime import datetime
import argparse

import datapane as dp

from dp_package.tools_for_dp import *

'''
weekly_report : report 저장하는 함수
Input : {store_num : Store 번호 (int) (Default: None > 입력 None으로 받을 시 유저에게 질문),
        year_week : 연, 주로 이루어진 리스트 (list) (Default: None > 입력 None으로 받을 시 유저에게 질문)
        directory: Raw Data가 있는 디렉토리 (str) (Default: './data/walmart')}
Output : -
'''
def weekly_report(store_num=None, year_week=None, directory='./data/walmart'):
    
    # year_week가 -1일 시, 현재 값으로
    try:
        if year_week==-1:
            current_datetime = datetime.now()
            year_week = [current_datetime.year,current_datetime.isocalendar()[1]]
    except:
        raise('[연, 주] 포맷으로 데이터를 입력해주세요')
    
    # store_num & year_week가 None으로 들어올 시 질문
    if store_num==None:
        store_num = int(input('보고서 생성을 원하는 매장의 번호를 입력해주세요 (1~45) : '))
    if year_week==None:
        y = int(input('보고서 생성을 원하는 연도를 입력해주세요 : '))
        if y==-1:
            current_datetime = datetime.now()
            year_week = [current_datetime.year,current_datetime.isocalendar()[1]]
        else:
            w = int(input('보고서 생성을 원하는 주차를 입력해주세요 : '))
        year_week=[y,w]

    print(f'{store_num}번 매장의 {year_week[0]}년 {year_week[1]}주차 보고서를 생성합니다.')
    
    # df 생성
    df=df_join(directory)
    
    # 특정 스토어 필터링
    df=df[df['Store']==store_num].reset_index(drop=True)
    
    store_sum, dept_uniq = store_summary(df,year_week)
    
    report=dp.Report(
                dp.HTML(header_image(df, store_num, year_week), label="Header Image"),
                dp.Blocks(
                    blocks=[store_sum]
                ),
                dp.Blocks(
                    '### 부서 분석',
                    dp.Select(
                        blocks=function_iter(dept_summary, dept_uniq, df, year_week),
                    type=dp.SelectType.DROPDOWN
                    )
                )
            )
    print('리포트 저장 중...')
    dp.save_report(report, path=f'report/월마트_{store_num}번매장_{year_week[0]}년도_{year_week[1]}주차.html')

def main():
    parser = argparse.ArgumentParser(description='Generate weekly report for a given store and week.')

    # 인자 추가
    parser.add_argument('--store_num', type=int, default=None, help='Store number (integer)')
    parser.add_argument('--year_week', type=int, nargs=2, default=None, help='Year and week in format YYYY WW')
    parser.add_argument('--directory', type=str, default='./data/walmart', help='Directory path')

    # 인자 파싱
    args = parser.parse_args()

    # weekly_report 함수 호출
    weekly_report(store_num=args.store_num, year_week=args.year_week, directory=args.directory)
    
if __name__ == '__main__':
    main()
