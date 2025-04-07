z = int(input())
a = list(map(int, input().split()))
b = []
for i in range(0, z):
    b.append(a[i]/max(a)*100)
print(round(sum(b)/len(b), 2))