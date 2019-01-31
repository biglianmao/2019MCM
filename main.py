from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import queue

img = Image.open("n1-d.bmp").convert('L')

img_data = np.array(img)
rows,cols = img_data.shape
map_data = np.zeros(shape=img_data.shape)
map_data[img_data<20] = 2
map_data[img_data>200] = 1


img_safe = map_data != 0
img_blank = map_data != 1
img_wall = map_data != 2


# 找出门在哪里
core = np.delete(map_data,(0,rows-1),axis=0)
core = np.delete(core,(0,cols-1),axis=1)
mask_1 = np.hstack((np.delete(map_data,0,axis=1),[[100]]*rows))
mask_2 = np.hstack(([[100]]*rows,np.delete(map_data,cols-1,axis=1)))
mask_3 = np.vstack((np.delete(map_data,0,axis=0),[100]*cols))
mask_4 = np.vstack(([100]*cols,np.delete(map_data,rows-1,axis=0)))
img_door = ~(((map_data+mask_1) == 1) | ((map_data+mask_2) == 1) | ((map_data+mask_3) == 1) | ((map_data+mask_4) == 1))
# print(np.array(np.where(~img_door)).T.shape)
if np.array(np.where(~img_door)).T.shape[0] > 0:
    print('has door!!!')
else:
    print('no door!!!')
#     exit()


# 计算势能矩阵
energy_data = np.zeros(shape=img_data.shape)
energy_data[img_data<20] = 100000
energy_data[img_data>200] = 10000

q=queue.Queue()
# x:纵坐标
# y:横坐标
for x,y in np.array(np.where(~img_door)).T:
    if energy_data[x,y] != 0:
#         print(x,y)
        energy_data[x,y] = 1
        q.put((x,y))

while q.qsize() > 0:
    x,y = q.get()
    current_len = energy_data[x,y]
    for nx,ny in np.array([[-1,0],[1,0],[0,-1],[0,1]]) + (x,y):
        # 如果没有走过 == 10000
        if energy_data[nx,ny] == 10000:
            energy_data[nx,ny] = current_len + 1
            q.put((nx,ny))



# 在空地生成人
person_count = 100
blank = np.transpose(np.where(map_data == 1))
persons = blank[np.random.choice(blank.shape[0],person_count,replace=False)][:,[1,0]]
# causion：person坐标以图像坐标系表示，纵向为x，横向为y

############################
# 开始画图

fig = plt.figure(figsize=(12, 12))

ax_orig = plt.subplot2grid((3, 4), (0, 0), colspan = 1, rowspan = 1)
ax_orig.imshow(img, cmap='gray')
ax_orig.axis('off')
ax_orig.set_title('orig')
ax_safe = plt.subplot2grid((3, 4), (1, 0), colspan = 1, rowspan = 1)
ax_safe.imshow(img_safe, cmap='gray')
ax_safe.axis('off')
ax_safe.set_title('safe')
ax_wall = plt.subplot2grid((3, 4), (2, 0), colspan = 1, rowspan = 1)
ax_wall.imshow(img_wall, cmap='gray')
ax_wall.axis('off')
ax_wall.set_title('wall')
ax_blank = plt.subplot2grid((3, 4), (0, 1), colspan = 1, rowspan = 1)
ax_blank.imshow(img_blank, cmap='gray')
ax_blank.axis('off')
ax_blank.set_title('blank')
# plt.imshow(img)


ax_door = plt.subplot2grid((3, 4), (1, 1), colspan = 1, rowspan = 1)
ax_door.imshow(img_door, cmap='gray')
ax_door.axis('off')
ax_door.set_title('door')

# 现实下势能矩阵
# 调整下像素值
img_door = np.copy(energy_data)
img_door[img_data<20] = 1000
ax_energy = plt.subplot2grid((3, 4), (2, 1), colspan = 1, rowspan = 1)
ax_energy.imshow(img_door, cmap='gray')
ax_energy.axis('off')
ax_energy.set_title('energy')


# fig=plt.figure(figsize=(15, 15))
ax_map = plt.subplot2grid((3, 4), (0, 2), colspan = 2, rowspan = 3)
ax_map.imshow(img, cmap='gray')
# ax_map.axis('off')
ax_map.set_title('map')
ax_map.xaxis.tick_top()


s = ax_map.scatter(persons[:,0],persons[:,1],c='r',marker = 'o')

def gen_frame():
    step = 0
    while step<100:
        print(step)
        step += 1
        yield step
    print('over')
    return step
    pass

def update(n):
    global persons
    for v in persons:

        px = v[1]
        py = v[0]
        # 在势能矩阵上寻找下一步
        current_len = energy_data[px,py]
        next_xy = np.array([[-1,0],[1,0],[0,-1],[0,1],[-1,-1],[-1,1],[1,-1],[1,1]]) + (px,py)
        next_len = energy_data[tuple(next_xy.T)]
        next_len_min_v = np.amin(next_len)
        # 直接取一个，应该不会重复
        next_len_min_i = np.where(next_len==next_len_min_v)[0][0]

        if next_len_min_v < current_len:
            v[1] = next_xy[next_len_min_i][0]
            v[0] = next_xy[next_len_min_i][1]

    s.set_offsets(persons)
    return s
ani = animation.FuncAnimation(fig, update, frames=gen_frame,repeat=False)


plt.show()