# 导入相应的库
from PIL import Image,ImageDraw
import numpy as np 
# 读取图像
big_image = Image.open("./img/big.jpg")
small_image = Image.open("./img/small.jpg")
print("成功读取图像")
# 将图片转换为灰度图
big_image_rgb = big_image
big_image = big_image.convert('L')
small_image = small_image.convert('L')

# 转换为矩阵格式
big = np.array(big_image)
small = np.array(small_image)

print("打印图像尺寸:")
print("big:",big.shape)
print("small:",small.shape)
#从small中随机取50个点,可能用不完,多了备用
rand_point = []
for i in range(50):
    rand_point.append((np.random.randint(0,small.shape[0]-1) ,np.random.randint(0,small.shape[1]-1)))

# 矩阵R 用来保存误差次数
R = np.zeros([big.shape[0]-small.shape[0],big.shape[1]-small.shape[1]])

sMean = np.mean(small) # 计算small的均值
print("正在进行计算，请稍后……")
for i in range(R.shape[0]):
    for j in range(R.shape[1]):
        loss = 0 #误差
        mean = np.mean(big[i:i+small.shape[0],j:j+small.shape[1]]) # s[i,j]的均值
        for n in range(len(rand_point)):
            point = rand_point[n] #从备选的随机点中取出一个随机点
            # 计算并叠加误差
            # 误差 = |备选区域中随机点的灰度值-备选区域均值 - small中随机点的灰度值 + small均值 |
            loss += np.abs( big[i+point[0],j+point[1]] - mean - small[point[0],point[1]] + sMean)
            # 当误差超过一定的限度，记录下使用的随机点个数
            if loss >= 150 or n==len(rand_point)-1: 
                R[i,j] = n
                break # 跳出n循环
# 找到迭代次数最多的点，对应的索引即为小图对应左上角位置坐标
index = np.unravel_index(R.argmax(), R.shape)
 
print("左上角左边：",index)
xy = [(index[1],index[0]),(index[1]+small.shape[1],index[0]+small.shape[0])]

# 在big图中画出对应的位置
big_image = big_image_rgb #将big还原为RGB图像
draw = ImageDraw.Draw(big_image)
draw.rectangle(xy,outline='red',width=3)
big_image.show()
print("已保存图像结果")
big_image.save("output.jpg")
