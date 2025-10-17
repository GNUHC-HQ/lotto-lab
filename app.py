import random
import pandas as pd
from flask import Flask, render_template, request, jsonify

# Flask 앱 생성
app = Flask(__name__)

# JSON 파일에서 로또 데이터 불러오기
try:
    # 날짜 컬럼을 datetime 객체로 읽어오면 좋지만, 여기서는 문자열로 충분합니다.
    lotto_df = pd.read_json('lotto_history.json')
    lotto_df = lotto_df.dropna(subset=['Date']).astype({'Date': str})
except FileNotFoundError:
    print("'lotto_history.json' 파일을 찾을 수 없습니다. 프로젝트 폴더에 파일이 있는지 확인하세요.")
    lotto_df = pd.DataFrame()


@app.route('/')
def home():
    """
    메인 페이지를 보여주는 함수.
    """
    # --- 1. 사용자 입력 처리 및 기본값 설정 ---
    start_str = request.args.get('start_round', '')
    end_str = request.args.get('end_round', '')

    # 생일 찾기용 입력
    birth_month_str = request.args.get('birth_month', '')
    birth_day_str = request.args.get('birth_day', '')

    analysis_form_submitted = 'start_round' in request.args or 'end_round' in request.args
    birthday_form_submitted = 'birth_month' in request.args or 'birth_day' in request.args

    analysis_section_open = analysis_form_submitted
    birthday_section_open = birthday_form_submitted

    # analysis_section_open = form_submitted

    latest_round = lotto_df.iloc[0]['No'] if not lotto_df.empty else 0

    start_round = int(start_str) if start_str.isdigit() else 1
    end_round = int(end_str) if end_str.isdigit() else latest_round

    # --- 2. 데이터 필터링 및 유효성 검사 ---
    error_message = None
    target_df = pd.DataFrame()

    if analysis_form_submitted and start_round > end_round:
        error_message = "종료 회차는 시작 회차보다 크거나 같아야 합니다."
        target_df = pd.DataFrame() # 에러 시 빈 데이터프레임
    else:
        # ✨ 핵심 수정: 통계분석 form이 제출되었거나, 아무 form도 제출되지 않은 초기 상태일 때만 필터링
        if analysis_form_submitted or not birthday_form_submitted:
             if not lotto_df.empty:
                target_df = lotto_df[
                    (lotto_df['No'] >= start_round) & (lotto_df['No'] <= end_round)
                ]
        # 생일 찾기만 제출된 경우, target_df는 전체 데이터(lotto_df)를 유지
        else:
             target_df = lotto_df

    # --- 3. 통계 데이터 생성 ---
    if not target_df.empty:
        number_columns = ['1', '2', '3', '4', '5', '6']
        existing_cols = [col for col in number_columns if col in target_df.columns]
        all_numbers = target_df[existing_cols].stack()
        number_counts = all_numbers.value_counts()
    else:
        number_counts = pd.Series()

    full_counts = pd.Series(0, index=range(1, 46), dtype=int)
    full_counts.update(number_counts)
    chart_labels = full_counts.index.tolist()
    chart_data = full_counts.values.tolist()

    # --- 3. 생일 로또 찾기 로직 (새로운 기능) ---
    birthday_lotto_results = []
    if birthday_form_submitted and birth_month_str.isdigit() and birth_day_str.isdigit():
        # "MM-DD" 형식으로 검색할 날짜 문자열 생성
        search_date = f"{int(birth_month_str):02d}-{int(birth_day_str):02d}"

        # 'Date' 열의 마지막 5자리('MM-DD')가 검색 날짜와 일치하는 모든 행을 찾음
        found_df = lotto_df[lotto_df['Date'].str.endswith(search_date)]

        # 찾은 결과를 리스트에 담음
        for index, row in found_df.iterrows():
            birthday_lotto_results.append({
                "round": row['No'],
                "date": row['Date'],
                "main": row[['1', '2', '3', '4', '5', '6']].tolist(),
                "bonus": row['B']
            })

    # --- 4. 기타 정보 준비 (최신 번호, 날짜 등) ---
    latest_winning_numbers = None
    if not lotto_df.empty:
        latest_row = lotto_df.iloc[0]
        latest_winning_numbers = {
            "round": latest_row['No'],
            "date": latest_row['Date'],
            "main": latest_row[['1', '2', '3', '4', '5', '6']].tolist(),
            "bonus": latest_row['B']
        }

    start_date = "N/A"
    end_date = "N/A"
    if not target_df.empty:
        end_date = target_df.iloc[-1]['Date']
        start_date = target_df.iloc[0]['Date']

    # --- 5. 최종 데이터 전달 ---
    return render_template(
        'index.html',
        chart_labels=chart_labels,
        chart_data=chart_data,
        latest_round=latest_round,
        start_date=start_date,
        end_date=end_date,
        current_start=start_str,
        current_end=end_str,
        latest_winning_numbers=latest_winning_numbers,
        analysis_section_open=analysis_section_open,
        error_message=error_message,
        birthday_section_open=birthday_section_open,
        birthday_lotto_results=birthday_lotto_results,
        current_month=birth_month_str,
        current_day=birth_day_str
    )


@app.route('/api/generate-numbers')
def generate_numbers():
    number_sets = []
    for _ in range(5):  # 5세트 생성
        # 1~45 숫자 중 6개를 중복 없이 뽑아 정렬
        numbers = sorted(random.sample(range(1, 46), 6))
        number_sets.append(numbers)

    # 생성된 번호 세트를 JSON 형태로 반환
    return jsonify(number_sets)

if __name__ == '__main__':
    app.run(debug=True)