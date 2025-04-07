import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

X,y = load_diabetes(return_X_y=True, as_frame=False)

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.33,
                                                    random_state=1234)
# 훈련 데이터 67%, 테스트 데이터 33%
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)
# y를 1열로 새우는 이유는 x의 요소들에 맞추어 대응하는 실제값이 1개이기 때문이다

n_train, n_test = X_train.shape[0], X_test.shape[0]
# numpy에서 shape를 이용하여 개수를 확인할 수 있음 shape[0] = 행의 개수, shape[1] = 열의 개수
# n_train = X_train의 행의 개수, n_test = X_test의 행의 개수
# 
X_train = np.append(np.ones((n_train, 1)), X_train, axis=1)
# np.ones((n_train, 1)) = 1로 채워진 열 백터 생성
# = n_train의 개수만큰 1로 채워진 행령을 생성
# 예제 = [[1],[1],[1],[1],[1],[1],[1],[1],[1],...]
# np.append(a, b, axis = 1) = b백터 안에 a를 집어 넣음 axis = 1은 열의 맨앞에 넣겠다는 의미
# 조건 b백터가 a백터의 행만큼 값을 가지고 있어야 함(안에 요소가 얼마나 있던 행만 맞추면 넣어짐)


w = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y_train
#최소 제곱법(OLS)은 y를 열벡터 X의 선형 결합과 오차항의 합으로 모델링한다 
# y = Xw + ϵ(엡실론) p.44
# y는 타겟 값 벡터,
# X는 입력 특성 행렬,
# w는 회귀 계수 벡터,
# ϵ은 오차입니다.
# 정규 방정식을 사용하여 회귀 계수 벡터w를 구하는 공식은 다음과 같습니다.
# w = (XTX)-1XTy
# 이 식은 정규 방정식을 사용하여 선형 회귀 모델의 **회귀 계수 벡터 w**를 계산하는 것입니다. 
# 선형 회귀 모델의 목표: 선형 회귀에서는 **입력 데이터(피처들)**와 목표 값(레이블) 사이의 관계를 모델링합니다.
#  예를 들어, X_train은 피처(독립 변수) 행렬이고, y_train은 이 피처들에 대한 실제 값(종속 변수)입니다.
# w = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y_train를 계산하는 이유는 모델의 최적의 회귀 계수를 구하여, 입력 데이터에 대해 최선의 예측을 할 수 있는 모델을 만들기 위해서입니다.  
# 최소 제곱법(OLS): 이 방식은 한 번의 계산으로 최적의 회귀 계수 w를 바로 구하는 방법입니다. 반복을 하지 않고 행렬 연산을 통해 정확한 해를 구할 수 있습니다.

#  = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y_train는 한 번의 계산을 통해 최적의 가중치(w)를 구하는 과정입니다. 
# 이 가중치는 선형 회귀 모델이 입력 데이터에 대해 예측한 값과 실제 값 사이의 오차를 최소화하는 최적의 파라미터입니다.

# w는 가중치로 

y_pred_train = X_train @ w
print(f'학습 데이터셋 MAE:{np.abs(y_pred_train - y_train).mean(): .3f}')
# abs = 절대값, mean() = 평균
X_test = np.append(np.ones((n_test, 1)), X_test, axis = 1)
y_pred = X_test @ w
print(f'테스트 데이터셋 MAE:{np.abs(y_pred - y_test).mean(): .3f}')

# 지금 이 코드는 오차범위를 알아내는 코드이다
# X_train = 임의의 데이터(y값에 영향을 주는는), y_train = 테스트 데이터의 실제값 = 기존에 있던거
# X_test = 새로운 임으의 데이터(x_train과 같은 요소를 가지고 있음), y_test = 새로운 테스트 데이터의 실제값값

# w는 가중치이다 이 가중치는 X_train의 열의 수만큼(요소만큼) 값이 생성됨
# 이 때 열의 앞에 1을 받았기 때문에 원래의 x_train보다 1개 더 많이 받았다고 할 수 있다
# y = w0 * 1 + w1*x1 + w2*x2 이렇게 처음 가중치는 그냥 더해지고 나머지는 그 요소의 가중치가 된다
# 이 w들은 요만큼 생성되면 이걸로 각행에 모두 사용한다
# 이 코드는 오차 범위들의 평균이므로 이 가중치들이 어느정도 틀리는지 알아보는 것이다.
# train의 가중치의 오류가 test에 어느정도로 작용하는가

# 문제 다중공선성 문제(변수들 끼리 너무 비슷한 정보가 많으면 문제가 생김)
# svd(특잇값 분해 방법) 다중공선성 뭄제를 해결하는 방법 