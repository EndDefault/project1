n, m = map(int, input().split())
answer = []
for i in range(n):
    answer.append(0)
for z in range(m):
    i, j, k = map(int, input().split())
    for x in range(i-1, j):
        answer[x] = k
for i in range(n):
    print(answer[i], end=" ")