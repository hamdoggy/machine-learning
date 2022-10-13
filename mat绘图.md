
```
plt.plot(x,y,labe='xx')
plt.show()
```
figure:创建窗口,重叠
```
plt.figure(num=2,figsize=(8,5))
plt.plot(x,y1,color='red',linewidth=1.0,linestyle='--')
plt.plot(x,y2)
plt.show()
```
坐标轴修改
```
plt.xlim((-1,2))
plt.ylim((-2,3))
plt.xlabel('qifei')
plt.ylabel('wuhu')

new_ticks=np.linspace(-1,2,5)
plt.xticks(new_ticks)
plt.yticks([-2,-1.8],['good','bad'])
```
坐标轴移到中间：
```
#gca=get current axis
ax=plt.gca()
#右上线消失
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.spines['bottom'].set_position(('data',0))#outward,axes
ax.spines['left'].set_position(('data',0))
plt.show()
```
图例:为plot的label
```
plt.legend()
```

散点图
```
t=np.arctan2(y,x)
plt.scatter(x,y,75,c=t)
```
