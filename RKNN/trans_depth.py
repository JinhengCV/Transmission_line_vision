######jinheng新加######
######2D-3D测距#######
######################
###可能的误差来源：选取的匹配点数，PNP算法，标定######
from __future__ import division
import numpy as np
import cv2
# from ppdet.utils.visualizer import angpoints, height, width
# from PIL import Image, ImageDraw, ImageFont
import math


import os
from PIL import Image, ImageDraw, ImageFile, ImageFont
ImageFile.LOAD_TRUNCATED_IMAGES = True


def visualize_box_mask(im, results, labels, threshold=0.5):
    """
    Args:
        im (str/np.ndarray): path of image/np.ndarray read by cv2
        results (dict): include 'boxes': np.ndarray: shape:[N,6], N: number of box,
                        matix element:[class, score, x_min, y_min, x_max, y_max]
                        MaskRCNN's results include 'masks': np.ndarray:
                        shape:[N, im_h, im_w]
        labels (list): labels:['class1', ..., 'classn']
        threshold (float): Threshold of score.
    Returns:
        im (PIL.Image.Image): visualized image
    """
    if isinstance(im, str):
        im = Image.open(im).convert('RGB')
    elif isinstance(im, np.ndarray):
        im = Image.fromarray(im)
    if 'boxes' in results and len(results['boxes']) > 0:
        im = draw_box(im, results['boxes'], labels, threshold=threshold)
    return im


def get_color_map_list(num_classes):
    """
    Args:
        num_classes (int): number of class
    Returns:
        color_map (list): RGB color list
    """
    color_map = num_classes * [0, 0, 0]
    for i in range(0, num_classes):
        j = 0
        lab = i
        while lab:
            color_map[i * 3] |= (((lab >> 0) & 1) << (7 - j))
            color_map[i * 3 + 1] |= (((lab >> 1) & 1) << (7 - j))
            color_map[i * 3 + 2] |= (((lab >> 2) & 1) << (7 - j))
            j += 1
            lab >>= 3
    color_map = [color_map[i:i + 3] for i in range(0, len(color_map), 3)]
    return color_map

def draw_box(im, np_boxes, labels, threshold=0.5):
    """
    Args:
        im (PIL.Image.Image): PIL image
        np_boxes (np.ndarray): shape:[N,6], N: number of box,
                               matix element:[class, score, x_min, y_min, x_max, y_max]
        labels (list): labels:['class1', ..., 'classn']
        threshold (float): threshold of box
    Returns:
        im (PIL.Image.Image): visualized image
    """
    im = Image.fromarray(im)
    
    ######jinheng新加的全局变量定义
    angpoints = []
    height = []
    width = []
    _label = []
    ##################
    draw_thickness = min(im.size) // 320  ###jinheng draw_thickness = 4
    draw = ImageDraw.Draw(im)
    clsid2color = {}
    color_list = get_color_map_list(len(labels))
    expect_boxes = (np_boxes[:, 1] > threshold) & (np_boxes[:, 0] > -1)
    np_boxes = np_boxes[expect_boxes, :]

    for dt in np_boxes:
        clsid, bbox, score = int(dt[0]), dt[2:], dt[1]
        if clsid not in clsid2color:
            clsid2color[clsid] = color_list[clsid]
        color = tuple(clsid2color[clsid])

        if len(bbox) == 4:
            xmin, ymin, xmax, ymax = bbox
            ####jinheng 新加高度宽度###
            w = xmax - xmin
            h = ymax - ymin
            if ((xmin + (xmax - xmin)/2 < 215) or (xmin + (xmax - xmin)/2 > 2374)):  ##########jinheng 不显示在图中两边刚出现的目标框
                continue
            ##########################

            print('class_id:{:d}, confidence:{:.4f}, left_top:[{:.2f},{:.2f}],'
                  'right_bottom:[{:.2f},{:.2f}]'.format(
                      int(clsid), score, xmin, ymin, xmax, ymax))
            
            # draw.line(            ######原本的
            #         [(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin),
            #         (xmin, ymin)],
            #         #  width=draw_thickness,  ###原来的
            #         width=10,      ####jinheng 替换--加粗线
            #         fill=color)

            ###############jinheng新增---吊车检测变为红色，并加粗框线
            # draw bbox
            if labels[clsid] == 'crane':
                draw.line(
                    [(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin),
                    (xmin, ymin)],
                    #  width=draw_thickness,  ###原来的
                    width=8,      ####jinheng 替换---加粗线
                    fill=(0, 0, 255))     #PIL与opencv的RGB值相反
            else:
                draw.line(
                    [(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin),
                    (xmin, ymin)],
                    #  width=draw_thickness,  ###原来的
                    width=10,      ####jinheng 替换
                    fill=color)
            ##############################

        elif len(bbox) == 8:
            x1, y1, x2, y2, x3, y3, x4, y4 = bbox
            draw.line(
                [(x1, y1), (x2, y2), (x3, y3), (x4, y4), (x1, y1)],
                width=2,
                fill=color)
            xmin = min(x1, x2, x3, x4)
            ymin = min(y1, y2, y3, y4)

        # draw label
        text = "{} {:.4f}".format(labels[clsid], score)
        tw, th = draw.textsize(text)
        ###############jinheng新增---吊车检测变为红色
        if labels[clsid] == 'crane':
            draw.rectangle(
                [(xmin + 1, ymin - th), (xmin + tw + 1, ymin)], fill=(0, 0, 255))
        else:
            draw.rectangle(
                [(xmin + 1, ymin - th), (xmin + tw + 1, ymin)], fill=color)
        ####################

        # draw.rectangle(
        #     [(xmin + 1, ymin - th), (xmin + tw + 1, ymin)], fill=color)   ####原本的
        draw.text((xmin + 1, ymin - th), text, fill=(255, 255, 255))

        #############jinheng新加 输出点和高度############
        angpoints.append([[xmin, ymax], [xmax, ymax]])
        height.append([h])
        width.append([w])
        _label.append(labels[clsid])
        # print(angpoints, type(angpoints))
        # print(_label, type(_label))
        ###########################################
    return im, angpoints, height, width, _label  ##jinheng新加, angpoints, height, width

# def draw_box(j, fill1, img, bbox):
#     xmin, ymin, xmax, ymax = bbox
#     draw = ImageDraw.Draw(img)
#     draw.line(            ######原本的
#             [(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin),
#             (xmin, ymin)],
#             #  width=draw_thickness,  ###原来的
#             width=10,      ####jinheng 替换--加粗线
#             fill=fill1)


###需要自己实验，到底选取几个点，误差小####
point3s=np.array((
    [21.802820, -5.550117, 31.803055],[20.190720, -4.895717, 19.950855], [-6.330480, 6.493983, 31.724955],[-4.941480, 5.298983, 19.982954]
,[-2.352194, 91.333687, 7.974355],[-0.627794, 90.117989, 18.687654],[-17.744894, 97.274292, 7.952156],[19.601120, -4.591317, 44.623455]
 ,[-18.394594, 98.105492, 33.159157], [-4.226480, 5.063983, 44.758953], [16.479820, -2.948417, 50.086155],[-0.345494,90.713387,32.243355]

,[7.625506, -106.135811, -22.586544],[0.257106, -124.009308, -22.017044],[-88.211090,-28.428610,-22.573944],[-95.851395,-48.260010,-20.936445]
,[-41.864292, -130.213913, -33.637844],[-89.624290, -106.762108, -33.229244], [-3.218194, -145.256714, -33.52693], [-19.022694, -140.770309, -33.794746]
, [5.271106, -147.464310, -33.710346],[-0.300394, -140.292618, -33.495743], [-21.347094, -146.887711, -34.041245], [7.635806, -106.535713, -33.384644]
,[-23.862894, -92.200912, -33.864944], [-73.378090, -68.072708, -32.932545], [-69.663193, -123.902809, -32.890244]

    ),dtype=np.double)

point2s=np.array((
    [1353, 404],[1334, 513],[1063, 394],[1075, 505]
,[836, 492],[856, 402],[688, 490],[1333, 284]
,[701, 288],[1092, 277],[1300, 235],[863,292]#上面是铁塔点

,[1904, 927],[2022, 913],[235, 874],[135,845] ##到处漂浮的点
,[1289, 1221],[179,1107], [2283,1174], [1953, 1206]  #马路
, [2443,1160], [2202,1166], [2031, 1226], [1900, 1104] #马路
, [1344, 1109] , [438,1059], [613, 1193]  #马路

),dtype=np.double)

camera_intrinsic = np.mat(np.array(([1.785072740345520e+03, 0, 1.251186131686387e+03],[0, 1.784492358648050e+03, 7.325446146606907e+02],[0,0,1]),dtype=np.float))


line1 = np.array([[[-99.808891, -102.813011, -0.807545],[-97.054497, -98.439011, -1.166645],
                [-89.903397, -80.933311, -1.886145], [-82.391891, -62.391712, -2.575545], [-76.285095, -48.064713, -2.573345]]])
line1_2d = np.array([[[243,272], [281, 304], [366, 360], [447, 415], [520, 459]]])

line2 = np.array([[[-81.092392, -106.081711, -1.107745],[-78.729591, -100.262108, -1.604545], 
                [-74.942596, -90.204811, -1.828045], [-67.329094, -71.688911, -2.521045], [-60.291595, -54.159512, -3.104945]]])
line2_2d = np.array([[[332, 23], [347, 43], [504, 234], [672, 403], [725, 453]]])
# line2_2d = np.array([[[444,161], [496, 222], [602, 338], [672, 403], [725, 453]]])  

line3 = np.array([[[-55.705093, -130.204514, 2.387055], [-52.289192, -122.084908, 2.680455], 
                [-44.865192, -105.003212, 3.659755], [-37.235992, -86.974014, 4.736655], [-30.471893, -70.537910, 6.275655]]])
line3_2d = np.array([[[1035, 28], [1038, 44], [1067, 252], [1092, 330], [1089, 383]]]) 
# line3_2d = np.array([[[1078,211], [1067, 252], [1092, 330], [1089, 383], [1104, 473]]]) 

line4 = np.array([[[-32.646893, -119.030014, 3.624055], [-30.573294, -114.045914, 4.032655], 
                [-28.073494, -107.977608, 4.020755], [-22.389694, -94.271309, 5.153755], [-15.362694, -77.519608, 6.093855]]])
line4_2d = np.array([[[1651, 67], [1611, 127], [1611, 127], [1514, 286], [1411, 432]]])  
# line4_2d = np.array([[[1542,246], [1514, 286], [1514, 286], [1411, 432], [1362, 493]]])  

lines = np.concatenate((line1, line2, line3, line4), axis = 0)
lines_2d = np.concatenate((line1_2d, line2_2d, line3_2d, line4_2d), axis = 0)
# print(lines[1], lines.shape)

# 畸变系数 [k1,k2,p1,p2,k3]
dist=np.array(([-0.415606568235958, 0.194606016994383, -2.715864007905328e-04, -2.943252670932223e-04, -0.047941656495039]),dtype=np.double)
dist=dist.T
#dist=np.zeros((5,1))
found,r,t=cv2.solvePnP(point3s, point2s, camera_intrinsic, dist, None, None, False, cv2.SOLVEPNP_EPNP) #计算雷达相机外参,r-旋转向量，t-平移向量
# print(r)
r=cv2.Rodrigues(r)[0] #旋转向量转旋转矩阵####罗德里格斯(Rodrigues)旋转公式

def pixel_to_world(camera_intrinsics, r, t, img_points, Zw):
    K_inv = camera_intrinsics.I
    R_inv = np.asmatrix(r).I  #3*3
    R_inv_T = np.dot(R_inv, np.asmatrix(t))  #3*3 dot 3*1 = 3*1  ###R-1*t
    world_points = []
    coords = np.zeros((3, 1), dtype=np.float64) ##像素坐标齐次矩阵
    for img_point in img_points:
        coords[0] = img_point[0]
        coords[1] = img_point[1]
        coords[2] = 1.0
        cam_point = np.dot(K_inv, coords)  ###K-1*[u v 1]
        cam_R_inv = np.dot(R_inv, cam_point)  ###R-1*(K-1*[u v 1])
        scale = (Zw + R_inv_T[2][0]) / cam_R_inv[2][0]   #s即改点的深度值，可转化为Zw高度相关的变量
        scale_world = np.multiply(scale, cam_R_inv)  ##s*()
        # print('scale_world =', scale_world )
        # print('R_inv_T=', R_inv_T)
        world_point = np.asmatrix(scale_world) - np.asmatrix(R_inv_T)
        world_point = world_point.reshape(1, 3)
        world_points.append(world_point.tolist())

    world_points = np.squeeze(np.array(world_points))
    # print('world_points=',  '\n', world_points, world_points.shape)
    return world_points

# 计算欧式距离
def ranging(P1, P2, ax):
    a = np.subtract(P1, P2)
    dist = np.sqrt(np.sum(np.square(np.subtract(P1, P2)), axis = ax)) #一个点对多点，axis=1.多点对多点，axis=2
    return dist

def threeunderpoints(angpoints, height):
    #########获得底下的3个二维坐标
    cenpoints = np.zeros((1, angpoints.shape[0], 2), dtype=np.float64)
    underpoints = np.zeros((angpoints.shape[0], angpoints.shape[1] + 1, angpoints.shape[2]))
    i = 0
    for angpoint in angpoints:
        cenpoints[0, i, 0] = angpoints[i,0,0] + (angpoints[i,1,0] - angpoints[i,0,0])/2
        cenpoints[0, i, 1] = angpoints[i,0,1]
        underpoints[i, :, :] = np.concatenate((angpoints[i], cenpoints[:, i, :]), axis=0)  ###框地下的三个坐标，顺序：[左,右,中]
        i +=1
    cenpoints = np.squeeze(cenpoints, 0)
    cenpoints[:, 1] = np.subtract(cenpoints[:, 1], np.squeeze(height)) ###框上边的中心点像素坐标，用于可视化
   
    ######获得底下的3个三维坐标
    j = 0
    under_worldpoints = np.zeros((underpoints.shape[0], underpoints.shape[1], 3), dtype=np.float64)
    for underpoint in underpoints:
        under_worldpoints[j] = pixel_to_world(camera_intrinsic, r, t, underpoint, Zw = -33.4)
        j += 1
    return  cenpoints, under_worldpoints 


def objectpoint(angpoints, height, width, _label):  #####获取图像中所有目标框的顶部三个点的三维坐标
    cenpoints_1, under_worldpoints_1 = threeunderpoints(angpoints, height)   ####所有的中心点和底下点

    ######获得目标的实际3D高度(框的3D高度)
    act_width_list = []
    for under_worldpoint in under_worldpoints_1:
        act_width = ranging(under_worldpoint[0, :].reshape(1, 3), under_worldpoint[1, :].reshape(1, 3), 1)
        act_width_list.append(act_width.tolist())
    act_width = np.array(act_width_list)
    # print('act_width', act_width, 'width',width)
    act_height = height*(act_width/width)
    # print('act_height', act_height)

    ###当存在吊臂目标时，吊车的测距发生变化；最后需要删除吊臂的坐标高宽信息
    if 'jib' in _label:
        ####下面的等测试完，可删
        location_num_jibs = [i for i,v in enumerate(_label) if v=='jib']  ###吊臂在标签列表_label中的所有位置索引--是个列表
        new_angpoints = angpoints
        # new_angpoints = np.delete(angpoints, location_num_jibs, axis=0)
        # point_jib = [j for j in angpoints[location_num_jibs]]
        # location_num_car = [i for i,v in enumerate(_label) if v=='car'] ###删除

        # diff = np.zeros(np.array(location_num_jibs).shape)
        # print('np.argmax(diff)', new_angpoints, 'point_jib=', point_jib, location_num_car)
        
        ####当吊臂和吊车同时存在时，把吊车的底下三个坐标换为吊臂的三个x值
        if 'crane' in _label:
            # location_num_jibs = [i for i,v in enumerate(_label) if v=='jib']  ###吊臂在标签列表_label中的所有位置索引--是个列表
            location_num_cranes = [i for i,v in enumerate(_label) if v=='crane']  ###吊车在标签列表_label中的所有位置索引--是个列表
            # new_angpoints = angpoints
            #当图中吊车数量少于吊臂时。通过两个for，计算吊臂和吊车最顶上Y值的差，其最小值绝对值属于配对成功的吊车-吊臂，随后赋值
            if np.size(location_num_cranes) < np.size(location_num_jibs):
                for location_num_crane in location_num_cranes:
                    i = 0
                    diff = np.zeros(np.array(location_num_jibs).shape)
                    for location_num_jib in location_num_jibs:
                        ####顶部Y值的差的绝对值
                        diff[i] = abs(new_angpoints[location_num_crane, 0, 1] - height[location_num_crane, 0] - new_angpoints[location_num_jib, 0, 1] + height[location_num_jib, 0])
                        i +=1
                    diff_min = np.argmin(diff)
                    new_angpoints[location_num_cranes, :, 0] = new_angpoints[location_num_jibs[diff_min], :, 0]  ###匹配成功后就替换赋值
            #当图中吊车数量多于吊臂时
            if np.size(location_num_cranes) >= np.size(location_num_jibs):
                for location_num_jib in location_num_jibs:
                    j = 0
                    diff = np.zeros(np.array(location_num_jibs).shape)
                    for location_num_crane in location_num_cranes:
                        diff[j] = abs(new_angpoints[location_num_crane, 0, 1] - height[location_num_crane, 0] - new_angpoints[location_num_jib, 0, 1] + height[location_num_jib, 0])
                        j +=1
                    diff_min = np.argmin(diff)
                    new_angpoints[location_num_cranes[diff_min], :, 0] = new_angpoints[location_num_jib, :, 0]
         #去除jib的信息，因为不用给吊臂测距
        angpoints = np.delete(new_angpoints, location_num_jibs, axis=0)  #####把jib的横坐标换给crane的横坐标，并删去jib信息
        # angpoints = np.delete(angpoints, location_num_jibs, axis=0)                  ########
        height = np.delete(height, location_num_jibs, axis=0)
        width = np.delete(width, location_num_jibs, axis=0)
        act_height = np.delete(act_height, location_num_jibs, axis=0)         
        cenpoints, under_worldpoints = threeunderpoints(angpoints, height)   ####如果存在jib，就要产生(去除了jib框的)新的中心点和底部点
        # print('EEEEE',angpoints, '\n', height, '\n', width, _label)
        #############################
        ##############################
    else:
        cenpoints, under_worldpoints = cenpoints_1, under_worldpoints_1

    ######获得框顶部三个点的3D坐标#########
    toppoints = under_worldpoints    ########坐标的顺序是---左顶角、右顶角、中间
    for c in range(act_height[:, 0].shape[0]):
        toppoints[c,:,2] = under_worldpoints[c,:,2] + act_height[c, :]
    # print(' toppoints',  toppoints)
    return toppoints, cenpoints

######线上的点与每个框顶部点的三维距离#####
def dist(toppoints, lines):
    dist_allboxs = []
    for toppoint in toppoints:
        dist_oneboxs = []
        for line in lines:
            dist_onebox = []
            for world_point in toppoint:
                distance = ranging(world_point, line, 1)
                dist_onebox.append(distance.tolist()) #########单个框顶的三个点到单个线的距离#####
            dist_oneboxs.append(dist_onebox)          ###########单个框到所有线的距离############
        dist_allboxs.append(dist_oneboxs)             ###########所有框到所有线的距离############
    dist_allboxs = np.array(dist_allboxs)
    # print('distance=',  dist_allboxs)

    return dist_allboxs

######已知单张图片所有框到线点的距离，计算最小值及对应的像素坐标####
def mindist(allbox_LINEs):
    lineplist = []
    _min_list = []
    for allbox_LINE in allbox_LINEs:
        _min = np.min(allbox_LINE)   ###单个框中的最小距离值
        locate = np.where(allbox_LINE == _min)   ###最小距离对应的精确位置，元祖索引
        # locate_linep = lines[locate[0][0], locate[2][0], :] #locate[0][0]对应的线，locate[2][0]对应点 #3D点
        # linep_3d, _ = cv2.projectPoints(locate_linep, r, t, camera_intrinsic, 0)  #可替换为对应位置的二维点

        linep_2d = lines_2d[locate[0][0], locate[2][0], :] #locate[0][0]对应的线，locate[2][0]对应点 #3D点
        lineplist.append(list(np.squeeze(linep_2d)))
        _min_list.append(_min)
    linep_2d = np.array(lineplist)
    _min_list = np.array(_min_list)
    # print('3Dlocate_linep', locate_linep)
    return linep_2d, _min_list


#####PIL画虚线->https://stackoverflow.com/questions/64276513/draw-dotted-or-dashed-rectangle-from-pil
class DashedImageDraw(ImageDraw.ImageDraw):

    def thick_line(self, xy, direction, fill=None, width=0):
        #xy – Sequence of 2-tuples like [(x, y), (x, y), ...]
        #direction – Sequence of 2-tuples like [(x, y), (x, y), ...]
        if xy[0] != xy[1]:
            self.line(xy, fill = fill, width = width)
        else:
            x1, y1 = xy[0]            
            dx1, dy1 = direction[0]
            dx2, dy2 = direction[1]
            if dy2 - dy1 < 0:
                x1 -= 1
            if dx2 - dx1 < 0:
                y1 -= 1
            if dy2 - dy1 != 0:
                if dx2 - dx1 != 0:
                    k = - (dx2 - dx1)/(dy2 - dy1)
                    a = 1/math.sqrt(1 + k**2)
                    b = (width*a - 1) /2
                else:
                    k = 0
                    b = (width - 1)/2
                x3 = x1 - math.floor(b)
                y3 = y1 - int(k*b)
                x4 = x1 + math.ceil(b)
                y4 = y1 + int(k*b)
            else:
                x3 = x1
                y3 = y1 - math.floor((width - 1)/2)
                x4 = x1
                y4 = y1 + math.ceil((width - 1)/2)
            self.line([(x3, y3), (x4, y4)], fill = fill, width = 1)
        return   
        
    def dashed_line(self, xy, dash=(2,2), fill=None, width=0):
        #xy – Sequence of 2-tuples like [(x, y), (x, y), ...]
        for i in range(len(xy) - 1):
            x1, y1 = xy[i]
            x2, y2 = xy[i + 1]
            x_length = x2 - x1
            y_length = y2 - y1
            length = math.sqrt(x_length**2 + y_length**2)
            dash_enabled = True
            postion = 0
            while postion <= length:
                for dash_step in dash:
                    if postion > length:
                        break
                    if dash_enabled:
                        start = postion/length
                        end = min((postion + dash_step - 1) / length, 1)
                        self.thick_line([(round(x1 + start*x_length),
                                          round(y1 + start*y_length)),
                                         (round(x1 + end*x_length),
                                          round(y1 + end*y_length))],
                                        xy, fill, width)
                    dash_enabled = not dash_enabled
                    postion += dash_step
        return
########虚线


####可视化###
def _draw(image, linep2ds, _mins, vis_cenpoints):
    # draw = ImageDraw.Draw(image) ##之前的实线
    draw = DashedImageDraw(image)
    i = 0
    for _min in _mins:
        drawshape = [(linep2ds[i, 0], linep2ds[i, 1]), (vis_cenpoints[i, 0], vis_cenpoints[i, 1])]
        # draw.line(drawshape, fill=(0, 255, 0), width=3)  ##之前的实线
        if _min < 17 and _min >= 16:           ####200kV小于12米线条变为黄色---“注意”，发出注意信号，
            fill_1 = (0, 255, 255)
            fill_2 = (0, 255, 255)
        elif _min < 16 and _min >= 15:          ####200kV小于10米线条变为橙色---“预警”，发出预警信号，
            fill_1 = (0, 165, 255)
            fill_2 = (0, 165, 255)
        elif _min < 15:                        ####200kV小于8米线条变为红色---“告警”，发出告警信号，
            fill_1 = (0, 0, 255)
            fill_2 = (0, 0, 255)
        else:
            fill_1 = (0, 255, 0)
            fill_2 = (255, 0, 0)
        draw.dashed_line(drawshape, dash = (10, 10), fill=fill_1, width = 8)
        x1 = abs(linep2ds[i, 0] - vis_cenpoints[i, 0])/2 + min(linep2ds[i, 0], vis_cenpoints[i, 0])
        y1 = abs(linep2ds[i, 1] - vis_cenpoints[i, 1])/2 + min(linep2ds[i, 1], vis_cenpoints[i, 1])
        text = "{:.2f} m".format(_min)
        myfont = ImageFont.truetype('arial.ttf', size=65)
        draw.text((x1+10, y1), text, anchor="ls", font = myfont, fill=fill_2)
        # = draw_box2(i, fill_1, fill_2, image)  #############根据距离调节框的颜色
        # image.show()
        i += 1
    return image
        
    #############################
    ########2D-3D测距#############
    ##############################



def transmission( image_file, im_results, labels, threshold):
# def transmission(image, angpoints, height, width, _label):
    ####jinheng新加获取框顶的三个点3D坐标#####
    image, angpoints, height, width, _label = draw_box(image_file, im_results, labels, threshold=threshold)
    angpoints = np.array(angpoints)
    height = np.array(height)
    width = np.array(width)
    if angpoints.shape[0] != 0 and (('crane' in _label) or ('car'  in _label) or ('truck' in _label)):
        toppoints, vis_cenpoints = objectpoint(angpoints, height, width, _label)  #顶上的三个3D点和可视化框顶2D点
        
        ##############测距######################
        ######allbox_LINEs的第一维[]是各个框，第二维是[[]]四根线分别到框的多个距离,
        #######第三维是三个顶点到每根线的距离，第四维是点到线上两个点的距离#
        allbox_LINEs = dist(toppoints, lines)
        # print('allbox_LINE', allbox_LINEs)

        ##############每个框的最小距离及其对应的线及其像素坐标值########
        linep2ds, _mins = mindist(allbox_LINEs)
        # print('locate', linep2ds, '_mins', _mins, '\n', vis_cenpoints)

        image = _draw(image, linep2ds, _mins, vis_cenpoints)    
    return image
