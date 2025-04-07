h, m = 0, 0
while True:
    h, m = map(int, input().split())
    if 0 <= h <= 23 or 0 <= m <= 59:
        break
m -= 45
if  m < 0:
    h -= 1
    m += 60
    if h < 0:
        h = 23
print(h, m)