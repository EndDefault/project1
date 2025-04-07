import numpy as np
from sklearn.model_selection import train_test_split

X, y = np.arange(120).reshape((30, 4)), list(range(30))
# np.arange(n)는 0부터 n-1까지의 정수 배열을 생성하는 NumPy 함수임.
# reshape((30, 4)) = 30행 4열로 만듬
# reshape는 앞의 주어진 요소들을 딱맞추어 나누지 못하면 오류 (많아도 적어도 안됨) -1은 자동으로 끝까지 쓰게 함

# print(X)

# print(y)

# print('X의 첫 5개 샘플:\n', X[:5, :], '\n')
# print('y의 첫 5개 샘플:\n', y[:5])

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.33,
                                                    random_state=1234)
# train_test_split() 은 데이터를 랜덤하게 훈련 데이터와 테스트 데이터로 분리하는 함수
# X -> 입력 데이터(feature)
# y -> 정답 데이터(labels)
# X_train, y_train -> 훈련(train) 데이터
# X_test, y_test -> 테스트(test) 데이터
# test_size = 0.33 -> 전체 데이터에서 33%를 테스트 데이터로 사용, 나머지 67%는 훈련 데이터로 사용
# random_state = 1234 -> 랜덤 시드를 고정하는 역할, 항상 같은 방식으로 데이터가 나뉘도록 설정
# X_train = test안에 들어간 데이터를 제외한 나머지 값을 넣는다
# X_test = X리스트 리스트 중 0.33을 넎음
# y_train = test안에 들어간 데이터를 제외한 나머지 값을 넣는다
# y_test = y리스트 리스트 중 0.33을 넎음

# print('데이터셋 분할:', len(X_train), len(y_train), len(X_test), len(y_test), '\n')
# print('X_train의 첫 5개 샘플:\n', X_train[:5, :], '\n')
# print('y_train의 첫 5개 샘플:\n', y_train[:5])
# 위에서

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

# MinMaxScaler는 각 특성(feature)의 값을 0과 1 사이로 정규화하는 스케일링 기법입니다.
# 이렇게 정규화를 하면, 모델이 데이터의 크기 차이로 인해 성능이 저하되는 것을 방지할 수 있습니다. 
# 예를 들어, 한 특성의 값이 1000이고 다른 특성의 값이 0~1이라면, 모델이 큰 값을 가진 특성에만 집중할 수 있습니다. 
# 정규화 후 모든 특성이 비슷한 범위에 있으면 모델이 효율적으로 학습할 수 있습니다.
# Xscaled = (X-Xmin)/(Xmax-Xmin) page.33

scaler.fit(X_train)
# fit() 메서드는 훈련 데이터(X_train)에 대해 최솟값과 최댓값을 계산합니다. 
# 이 값은 MinMaxScaler가 데이터를 0과 1 사이로 정규화하는 데 필요한 기준이 됩니다.
# 훈련 데이터에만 fit을 적용하는 이유는,
# 모델을 학습할 때 훈련 데이터의 특성을 기준으로 스케일링을 해야만 테스트 데이터가 훈련 데이터의 분포에 맞춰 평가될 수 있기 때문입니다.

# fit()은 주어진 데이터의 최소값과 최댓값을 계산하여, 정규화에 필요한 기준을 학습합니다.
# 이 값들은 훈련 데이터에서만 계산됩니다. 즉, X_train에 대해 최소값과 최대값을 찾고, 이를 나중에 변환 과정에서 사용합니다.
# fit로 주어진 최소값과 최댓값을 transform에 사용됨됨

X_test_scaled = scaler.transform(X_test)
# transform() 메서드는 fit()을 통해 계산된 최솟값과 최댓값을 사용하여 테스트 데이터(X_test)의 값을 0과 1 사이로 변환합니다.
# 테스트 데이터에 대한 변환만 transform을 사용하는 이유는,
# 테스트 데이터가 훈련 데이터의 스케일을 그대로 따라야 하기 때문입니다.
# 만약 X_test에 대해 fit을 적용하면, 
# 테스트 데이터의 분포에 따라 새로운 최솟값과 최댓값이 계산되어 훈련 데이터와 다른 방식으로 스케일링될 수 있습니다. 
# 이는 **데이터 유출(data leakage)**을 초래할 수 있습니다.

# transform()은 fit()에서 계산된 최소값과 최대값을 이용해 주어진 데이터 (보통 X_test)의 값을 0과 1 사이로 변환합니다.
# transform은 훈련 데이터에 대해서는 이미 학습된 기준을 적용하고, 테스트 데이터에 대해서도 훈련 데이터의 기준을 따라 변환합니다.
# 최대 최소값을 fit에서 가지고 온다

# 스케일링 (정규화):
# MinMaxScaler는 각 특성(feature)의 값을 0과 1 사이로 정규화하는 데 사용됩니다. 
# scaler.fit(X_train)은 훈련 데이터의 특성에 대해 스케일러를 학습시키는 과정입니다. 
# 학습된 스케일러는 X_test에 대해 transform 메서드를 사용하여 테스트 데이터의 각 특성값을 정규화합니다.

print(f"X_test_scaled의 첫 5개 샘플: \n{np.array2string(X_test_scaled[:5, :], precision =3, floatmode='fixed')}")
# np.array2string(): numpy 배열을 문자열로 변환하는 함수, 배열을 사람이 읽을 수 있는 형태로 출력(:5, :는 2개의 리스트를 각각 출력하도록 한것)
# precision = 3 : 출력되는 실수값의 소수점 이하 자릿수 3자리 제한
# floatmode = 'fixed' : 고정 소수점 표기법을 사용하여 실수 값을 출력하도록 설정합니다.
# floatmode = 'fixed' 예시 : 0.1 == 0.100 - 기본 표기법 0.1