######jinheng新加######
######2D-3D测距#######
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

from parameters.parameters_wuhan import camera_intrinsic, r, t, optimization, depth_map, segments, reps_concat

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
    使用 OpenCV 在输入图像上绘制检测框、类别标签和置信度（字体更小版本）。
    Args:
        im (numpy.ndarray): 输入图像（OpenCV格式，BGR通道）
        np_boxes (np.ndarray): 检测框矩阵，shape=[N,6]，每行 [class, score, x_min, y_min, x_max, y_max]
        labels (list): 类别名称列表
        threshold (float): 置信度阈值，小于该值的检测框不绘制
    Returns:
        im (numpy.ndarray): 绘制后的图像
        angpoints (list): 每个目标框底边的两个角点坐标
        height (list): 每个目标框的高度
        width (list): 每个目标框的宽度
        _label (list): 每个目标的类别名称
    """
    Xwide = im.shape[1]  # 获取图像宽度
    angpoints, height, width, _label = [], [], [], []

    # ---------- 绘制参数 ----------
    draw_thickness = 8
    if Xwide < 2000:
        draw_thickness = draw_thickness // 2  # 图像小则框线更细
    font_scale = 0.4 if Xwide < 2000 else 0.5  # 字体缩放比例（减小字体）
    thickness = 1 if Xwide < 2000 else 2        # 字体线宽

    # ---------- 初始化颜色映射 ----------
    clsid2color = {}
    color_list = get_color_map_list(len(labels))
    expect_boxes = (np_boxes[:, 1] > threshold) & (np_boxes[:, 0] > -1)
    np_boxes = np_boxes[expect_boxes, :]  # 过滤掉低置信度目标

    # ---------- 遍历所有检测框 ----------
    for dt in np_boxes:
        clsid, bbox, score = int(dt[0]), dt[2:], dt[1]
        if clsid not in clsid2color:
            clsid2color[clsid] = color_list[clsid]
        color = tuple(int(c) for c in clsid2color[clsid][::-1])  # RGB→BGR

        xmin, ymin, xmax, ymax = map(int, bbox)
        w, h = xmax - xmin, ymax - ymin

        # ---------- 判断类别，吊车特殊显示 ----------
        if labels[clsid] == 'crane':
            box_color = (0, 0, 255)  # 红色
        else:
            box_color = color

        # ---------- 绘制检测框 ----------
        cv2.rectangle(im, (xmin, ymin), (xmax, ymax), box_color, draw_thickness)

        # ---------- 绘制类别+置信度标签 ----------
        text = "{} {:.2f}".format(labels[clsid], score)

        # 获取文本大小（宽、高）
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

        # 标签背景矩形更紧凑
        text_bg_top = max(ymin - th - 2, 0)
        text_bg_bottom = ymin
        text_bg_right = xmin + tw + 2
        cv2.rectangle(im, (xmin, text_bg_top), (text_bg_right, text_bg_bottom), box_color, -1)

        # 绘制白色文本
        cv2.putText(im, text, (xmin + 1, max(ymin - 3, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255),
                    thickness, lineType=cv2.LINE_AA)

        # ---------- 保存框参数信息 ----------
        angpoints.append([[xmin, ymax], [xmax, ymax]])  # 框底边两个点
        height.append([h])
        width.append([w])
        _label.append(labels[clsid])

    return im, angpoints, height, width, _label



def pixel_to_world(img_points, depth):
    """
    从像素坐标和深度值计算世界坐标。
    
    参数:
    - camera_intrinsic: 相机内参矩阵 K
    - r: 旋转矩阵 R
    - t: 平移向量 T
    - img_points: 图像上的像素点坐标
    - depth_map: 深度值矩阵(从CSV文件中读取的深度图)
    
    返回:
    - world_points: 世界坐标系中的坐标
    """
    K_inv = np.linalg.inv(camera_intrinsic)  # 计算内参矩阵的逆矩阵
    R_inv = np.linalg.inv(r)  # 计算旋转矩阵的逆矩阵
    R_inv_T = np.dot(R_inv, t)  # 计算 R-1 * T

    # 初始化世界坐标数组
    world_points = np.zeros((img_points.shape[0], 3), dtype=np.float64)
    # print('depth_values =', depth)
    
    for i, img_point in enumerate(img_points):
        if depth == 0:
            continue  # 跳过没有深度值的点

        coords = np.array([img_point[0], img_point[1], 1.0]).reshape((3, 1))
        cam_point = np.dot(K_inv, coords)  # 计算 K-1 * [u v 1].T
        cam_R_inv = np.dot(R_inv, cam_point)  # 计算 R-1 * (K-1 * [u v 1].T)

        scale = depth
        scale_world = scale * cam_R_inv  # 计算 s * (R-1 * (K-1 * [u v 1].T))

        world_point = scale_world - R_inv_T  # 计算最终的世界坐标
        world_point = world_point.reshape(1, 3)
        world_points[i] = world_point
    return world_points

# 计算欧式距离
def ranging(P1, P2, ax):
    a = np.subtract(P1, P2)
    dist = np.sqrt(np.sum(np.square(np.subtract(P1, P2)), axis = ax)) #一个点对多点，axis=1.多点对多点，axis=2
    return dist


def remove_outliers(depth_values):
    """
    去除深度值中的异常值（过大、过小的值以及大于300米的值），
    使用四分位差（IQR）法来去除异常值。
    """
    if len(depth_values) < 4:
        return depth_values  # 如果数据点少，直接返回原始数据
    
    # 去除深度值超过300米的点
    depth_values = [d for d in depth_values if d <= 300]
    
    if not depth_values:  # 检查深度值列表是否为空
        return depth_values

    # 计算四分位差（IQR）
    Q1 = np.percentile(depth_values, 25)
    Q3 = np.percentile(depth_values, 75)
    IQR = Q3 - Q1

    # 设置异常值阈值
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # 去除异常值
    return [d for d in depth_values if lower_bound <= d <= upper_bound]

def find_depth_in_range(left_x, right_x, center_y):
    """
    优化版：向量化 + 内联异常值过滤 + 减少重复循环。
    测距结果与原函数完全一致。
    """
    h, w = depth_map.shape
    # === 坐标安全裁剪，防止越界 ===
    center_y = max(0, min(center_y, h - 1))
    left_x = max(0, min(left_x, w - 1))
    right_x = max(0, min(right_x, w - 1))
    center_x = (left_x + right_x) // 2

    def _filter_outliers(values):
        if len(values) == 0:
            return values
        mean = np.mean(values)
        std = np.std(values)
        return values[(values > mean - 3 * std) & (values < mean + 3 * std)]

    # === [1] 扫描 center_y 行 ===
    if 0 <= center_y < h:
        row_values = depth_map[center_y, max(0, left_x):min(w, right_x + 1)]
        valid_depths = row_values[row_values > 0]
    else:
        valid_depths = np.array([])

    if valid_depths.size > 0:
        valid_depths = _filter_outliers(valid_depths)
        if valid_depths.size > 0:
            median_depth = np.median(valid_depths)
            return median_depth, (center_x, center_y)

    # === [2] 相邻行搜索 ===
    for dy in (-1, 1):
        y = center_y + dy
        if 0 <= y < h:
            row_values = depth_map[y, max(0, left_x):min(w, right_x + 1)]
            valid_depths = row_values[row_values > 0]
            if valid_depths.size > 0:
                valid_depths = _filter_outliers(valid_depths)
                if valid_depths.size > 0:
                    median_depth = np.median(valid_depths)
                    return median_depth, (center_x, y)

    # === [3] 向外扩展搜索 ===
    max_extend = max(w - right_x, left_x)
    for step in range(1, max_extend):
        left = max(0, left_x - step)
        right = min(w, right_x + step + 1)
        row_values = depth_map[center_y, left:right]
        valid_depths = row_values[row_values > 0]

        if valid_depths.size > 0:
            valid_depths = _filter_outliers(valid_depths)
            if valid_depths.size > 0:
                median_depth = np.median(valid_depths)
                return median_depth, (center_x, center_y)

    return 0, (-1, -1)


def process_underpoints(underpoints):
    """
    处理每个 underpoint, 找到一个深度值并应用于三个像素坐标。
    """
    under_worldpoints = np.zeros((underpoints.shape[0], underpoints.shape[1], 3), dtype=np.float64)
    depth_values = []  # 存储所有深度值

    for j, underpoint in enumerate(underpoints):
        # [1] 提取当前点集坐标
        coords = np.array([underpoint[:, 0], underpoint[:, 1]]).T
        left_x, left_y = coords[0]
        right_x, right_y = coords[1]
        center_y = int(np.mean([left_y, right_y]))

        # [2] 查找符合条件的深度值
        depth, min_coord = find_depth_in_range(int(left_x), int(right_x), int(center_y))
        depth_values.append(depth)

        # [3] 像素坐标转世界坐标
        under_worldpoints[j] = pixel_to_world(coords, depth)

    return under_worldpoints, depth_values



def threeunderpoints(angpoints, height):
    """
    获得每个目标底部三个二维坐标及对应的三维坐标。
    """
    # === [1] 初始化并生成二维点 ===
    cenpoints = np.zeros((1, angpoints.shape[0], 2), dtype=np.float64)
    underpoints = np.zeros((angpoints.shape[0], angpoints.shape[1] + 1, angpoints.shape[2]))
    i = 0
    for angpoint in angpoints:
        cenpoints[0, i, 0] = angpoints[i, 0, 0] + (angpoints[i, 1, 0] - angpoints[i, 0, 0]) / 2
        cenpoints[0, i, 1] = angpoints[i, 0, 1]
        underpoints[i, :, :] = np.concatenate((angpoints[i], cenpoints[:, i, :]), axis=0)
        i += 1

    # === [2] 计算中心点像素坐标 ===
    cenpoints = np.squeeze(cenpoints, 0)
    cenpoints[:, 1] = np.subtract(cenpoints[:, 1], np.squeeze(height))

    # === [3] 获得底下的三维坐标 ===
    under_worldpoints, object_depth = process_underpoints(underpoints)

    return cenpoints, under_worldpoints, object_depth


def objectpoint(angpoints, height, width, _label):
    """
    获取图像中所有目标框的顶部三个点的三维坐标。
    """
    cenpoints_1, under_worldpoints_1, object_depth = threeunderpoints(angpoints, height)

    # [B] 计算所有框的实际 3D 宽度（ranging 循环）
    act_width_list = []
    for under_worldpoint in under_worldpoints_1:
        aw = ranging(under_worldpoint[0, :].reshape(1, 3),
                     under_worldpoint[1, :].reshape(1, 3), 1)
        act_width_list.append(aw.tolist())
    act_width = np.array(act_width_list)
    act_height = height * (act_width / width)

    # [D] 若含 jib：做配对替换 + 删 jib + 重新 threeunderpoints()
    if 'jib' in _label:
        location_num_jibs = [i for i, v in enumerate(_label) if v == 'jib']
        new_angpoints = angpoints

        if 'crane' in _label:
            location_num_cranes = [i for i, v in enumerate(_label) if v == 'crane']

            if np.size(location_num_cranes) < np.size(location_num_jibs):
                for location_num_crane in location_num_cranes:
                    i = 0
                    diff = np.zeros(np.array(location_num_jibs).shape)
                    for location_num_jib in location_num_jibs:
                        diff[i] = abs(new_angpoints[location_num_crane, 0, 1] - height[location_num_crane, 0]
                                      - new_angpoints[location_num_jib, 0, 1] + height[location_num_jib, 0])
                        i += 1
                    diff_min = np.argmin(diff)
                    new_angpoints[location_num_cranes, :, 0] = new_angpoints[location_num_jibs[diff_min], :, 0]

            if np.size(location_num_cranes) >= np.size(location_num_jibs):
                for location_num_jib in location_num_jibs:
                    # jib 越界 → 跳过
                    if location_num_jib >= new_angpoints.shape[0] or location_num_jib >= height.shape[0]:
                        continue

                    j = 0
                    diff = np.zeros(np.array(location_num_cranes).shape)

                    for location_num_crane in location_num_cranes:
                        # crane 越界 → 跳过
                        if location_num_crane >= new_angpoints.shape[0] or location_num_crane >= height.shape[0]:
                            continue

                        diff[j] = abs(new_angpoints[location_num_crane, 0, 1] - height[location_num_crane, 0]
                                    - new_angpoints[location_num_jib, 0, 1] + height[location_num_jib, 0])
                        j += 1

                    if j == 0:
                        # 无可匹配吊车 → 这个吊臂直接跳过（后面会删除）
                        continue

                    diff_min = np.argmin(diff[:j])
                    # 修正最佳匹配吊车的横坐标
                    new_angpoints[location_num_cranes[diff_min], :, 0] = new_angpoints[location_num_jib, :, 0]


        # 删除 jib 的数组项
        angpoints = np.delete(new_angpoints, location_num_jibs, axis=0)
        height = np.delete(height, location_num_jibs, axis=0)
        width = np.delete(width, location_num_jibs, axis=0)
        act_height = np.delete(act_height, location_num_jibs, axis=0)

        # 去掉 jib 后重新 threeunderpoints()
        cenpoints, under_worldpoints, object_depth = threeunderpoints(angpoints, height)
    else:
        cenpoints, under_worldpoints = cenpoints_1, under_worldpoints_1

    # [E] 构造 toppoints：把 3D 高度加到顶点 Z 上
    toppoints = under_worldpoints.copy()
    for c in range(act_height[:, 0].shape[0]):
        toppoints[c, :, 2] = under_worldpoints[c, :, 2] + act_height[c, :]

    return toppoints, cenpoints, object_depth, height


######线上的点与每个框顶部点的三维距离#####
# 用于计算每个分段的左、中、右代表点的距离
# 避免计算平均值，只需要计算每个分段的三个代表点的距离
def dist(toppoints, candidate_k=1):
    dist_allboxs, min_points_allboxs = [], []

    for toppoint in toppoints:
        min_distance = float("inf")
        min_point = None

        for world_point in toppoint:
            # === [1] 用代表点快速锁定候选分段 ===
            d_rep = np.linalg.norm(reps_concat - world_point, axis=1)
            rep_group = d_rep.reshape(-1, 3).min(axis=1)  # 每段取最近代表点距离
            top_idx = np.argsort(rep_group)[:candidate_k]  # 选前K个分段
            candidate_points = np.vstack([segments[i] for i in top_idx])

            # === [2] 在候选分段内计算精确最小距离 ===
            distances = np.linalg.norm(candidate_points - world_point, axis=1)

            # === [3] argmin ===
            min_idx = np.argmin(distances)

            # === [4] 更新最小值 ===
            current_min_distance = distances[min_idx]
            current_min_point = candidate_points[min_idx]
            if current_min_distance < min_distance:
                min_distance = current_min_distance
                min_point = current_min_point

        dist_allboxs.append(min_distance)
        min_points_allboxs.append(min_point)

    return np.array(dist_allboxs), np.array(min_points_allboxs)


######已知单张图片所有框到线点的距离，计算对应像素坐标#####
def mindist(allbox_mins, min_points_line, image_shape):
    """
    计算每个框的最小距离点的像素坐标和对应最小距离。

    参数:
        allbox_mins: 每个框到线的最小距离数组。
        min_points_line: 每个框对应最小距离的点云三维点数组。
        image_shape: 图像尺寸 (高度, 宽度)。

    返回:
        lineplist: 每个框的最小距离点的像素坐标。
        _min_list: 每个框的最小距离。
    """
    lineplist = []  # 存储像素坐标
    _min_list = []  # 存储最小距离

    for distance, point_3d in zip(allbox_mins, min_points_line):
        point_3d = np.array(point_3d)

        # 投影到2D图像坐标系
        linep_2d, _ = cv2.projectPoints(
            point_3d.reshape(-1, 1, 3),
            np.array(r), np.array(t),
            camera_intrinsic, 0
        )
        linep_2d = linep_2d.squeeze()

        lineplist.append(linep_2d)
        _min_list.append(distance)

    return np.array(lineplist), np.array(_min_list)


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


def _draw(cv_img, linep2ds, _mins, vis_cenpoints, vis_depth, height):
    """
    使用 OpenCV 在图像上绘制：
        1. 框到导线的虚线连接；
        2. 每个框的最小距离文本；
        3. 框底部中心下方的深度信息文本。
    Args:
        cv_img (numpy.ndarray): 输入图像（BGR格式）
        linep2ds (np.ndarray): 每个框到导线的最短点对应的像素坐标
        _mins (list): 每个框的最小距离（米）
        vis_cenpoints (np.ndarray): 每个框的上边中心点像素坐标
        vis_depth (list): 每个框对应的深度值（米）
        height (list): 每个框的像素高度
    Returns:
        cv_img (numpy.ndarray): 绘制完成后的图像
    """
    image_height, image_width = cv_img.shape[:2]
    font_scale = 0.9 if image_width < 2000 else 1.4  # 文本字体比例
    thickness = 2 if image_width < 2000 else 3       # 文本线宽
    dashed_width = 6 if image_width < 2000 else 8     # 虚线宽度

    # ---------- 定义内部函数：绘制虚线 ----------
    def draw_dashed_line(img, pt1, pt2, color, thickness=8, dash_len=10, gap=10):
        """在两点之间绘制虚线"""
        dist = int(np.linalg.norm(np.array(pt1) - np.array(pt2)))
        for j in range(0, dist, dash_len + gap):
            start = (
                int(pt1[0] + (pt2[0] - pt1[0]) * j / dist),
                int(pt1[1] + (pt2[1] - pt1[1]) * j / dist)
            )
            end = (
                int(pt1[0] + (pt2[0] - pt1[0]) * min(j + dash_len, dist) / dist),
                int(pt1[1] + (pt2[1] - pt1[1]) * min(j + dash_len, dist) / dist)
            )
            cv2.line(img, start, end, color, thickness)

    # ---------- 遍历绘制每个目标的测距信息 ----------
    for i, _min in enumerate(_mins):
        pt1 = (int(linep2ds[i, 0]), int(linep2ds[i, 1]))  # 线上的投影点
        pt2 = (int(vis_cenpoints[i, 0]), int(vis_cenpoints[i, 1]))  # 框顶部中心点

        # --- 距离阈值判断，颜色映射规则 ---
        if _min < 16:
            fill_1 = fill_2 = (0, 0, 255)     # 红色：告警
        elif _min < 16.5:
            fill_1 = fill_2 = (0, 165, 255)   # 橙色：预警
        elif _min < 17:
            fill_1 = fill_2 = (0, 255, 255)   # 黄色：注意
        else:
            fill_1 = (0, 255, 0)              # 绿色：正常
            fill_2 = (255, 0, 0)

        # ---------- 绘制虚线连接 ----------
        draw_dashed_line(cv_img, pt1, pt2, fill_1, thickness=dashed_width)

        # ---------- 绘制最小距离文本 ----------
        x1 = abs(pt1[0] - pt2[0]) / 2 + min(pt1[0], pt2[0])
        y1 = abs(pt1[1] - pt2[1]) / 2 + min(pt1[1], pt2[1])
        text = "{:.2f} m".format(_min)
        cv2.putText(cv_img, text, (int(x1 + 10), int(y1)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, fill_2, thickness + 1, lineType=cv2.LINE_AA)

        # ---------- 绘制深度信息文本 ----------
        depth_text = "d: {:.2f} m".format(vis_depth[i])
        (dw, dh), _ = cv2.getTextSize(depth_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        box_bottom_x = int(vis_cenpoints[i, 0] - dw // 2)
        box_bottom_y = int(vis_cenpoints[i, 1] + height[i] + dh + 10)  # 框底下方10像素
        cv2.putText(cv_img, depth_text, (box_bottom_x, box_bottom_y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, lineType=cv2.LINE_AA)

    return cv_img


def adjust_text_position(drawn_text_boxes, original_x1, original_y1, text_width, text_height, line_start, line_end):
    """
    动态调整文本位置，避免与已有文本框重叠超过 50%，并沿着线段方向移动。
    """
    margin = text_height  # 移动一个字体高度
    new_x1, new_y1 = original_x1, original_y1

    # 计算线段的方向
    dx, dy = line_end[0] - line_start[0], line_end[1] - line_start[1]
    line_length = (dx**2 + dy**2)**0.5
    dx /= line_length
    dy /= line_length

    # 检查是否重叠超过50%且是否需要调整位置
    def check_overlap(new_x1, new_y1):
        for (bx1, by1, bx2, by2) in drawn_text_boxes:
            if is_overlap(new_x1, new_y1, text_width, text_height, bx1, by1, bx2, by2):
                overlap_ratio = calculate_overlap_ratio(new_x1, new_y1, text_width, text_height, bx1, by1, bx2, by2)
                if overlap_ratio > 0.2:  # 如果重叠超过50%
                    return True
        return False

    # 计算重叠比例
    def calculate_overlap_ratio(x1, y1, width, height, bx1, by1, bx2, by2):
        # 计算重叠面积和总面积
        overlap_width = max(0, min(x1 + width, bx2) - max(x1, bx1))
        overlap_height = max(0, min(y1 + height, by2) - max(y1, by1))
        overlap_area = overlap_width * overlap_height
        total_area = width * height + (bx2 - bx1) * (by2 - by1) - overlap_area
        return overlap_area / total_area  # 重叠比例

    # 初始位置检查
    while check_overlap(new_x1, new_y1):
        # 如果有重叠，沿着线段方向移动一个字体高度
        new_x1 += int(dx * margin)
        new_y1 += int(dy * margin)

    return new_x1, new_y1


def adjust_depth_text_position(drawn_text_boxes, original_x1, original_y1, text_width, text_height):
    """
    调整深度文本位置，避免与已有文本框重叠超过 50%。
    """
    margin = int(text_height * 1.0)  # 字体高度的 100%
    new_x1, new_y1 = original_x1, original_y1

    # 检查重叠情况并调整位置
    for (bx1, by1, bx2, by2) in drawn_text_boxes:
        if is_overlap(original_x1, original_y1, text_width, text_height, bx1, by1, bx2, by2):
            # 如果有重叠，向下移动
            new_y1 += margin  # 先垂直向下移动
            break  # 只调整一次，避免过多移动

    return new_y1


def is_overlap(x1, y1, width, height, bx1, by1, bx2, by2):
    """
    判断两个矩形区域是否重叠。
    """
    # 计算两个矩形是否重叠（根据坐标和宽高）
    return not (x1 + width < bx1 or x1 > bx2 or y1 + height < by1 or y1 > by2)

      
    #############################
    ########2D-3D测距#############
    ##############################
import time
import numpy as np

# jinheng新增：定义全局计时器字典
_time_stats = {
    "draw_box": 0.0,
    "objectpoint": 0.0,
    "dist": 0.0,
    "mindist": 0.0,
    "_draw": 0.0
}


def transmission(image_file, im_results, labels, threshold):
    global _time_stats

    # Step 1: draw_box
    t0 = time.time()
    image, angpoints, height, width, _label = draw_box(image_file, im_results, labels, threshold=threshold)
    _time_stats["draw_box"] += (time.time() - t0)

    image_shape = (image.shape[0], image.shape[1])  # (height, width)

    angpoints = np.array(angpoints)
    height = np.array(height)
    width = np.array(width)

    if angpoints.shape[0] != 0 and (('crane' in _label) or ('car' in _label) or ('truck' in _label)):
        # Step 2: objectpoint
        t1 = time.time()
        toppoints, vis_cenpoints, vis_depth, height = objectpoint(angpoints, height, width, _label)
        _time_stats["objectpoint"] += (time.time() - t1)

        # Step 3: dist
        t2 = time.time()
        allbox_mins, min_points_line = dist(toppoints)
        _time_stats["dist"] += (time.time() - t2)

        # Step 4: mindist
        t3 = time.time()
        linep2ds, _mins = mindist(allbox_mins, min_points_line, image_shape)
        _mins = allbox_mins + optimization
        _time_stats["mindist"] += (time.time() - t3)

        # Step 5: _draw
        t4 = time.time()
        image = _draw(image, linep2ds, _mins, vis_cenpoints, vis_depth, height)
        _time_stats["_draw"] += (time.time() - t4)

    return image


# jinheng新增：视频处理结束后调用此函数输出总耗时统计
def print_transmission_time_summary():
    print("\n=========== Transmission Function Time Summary ===========")
    total = sum(_time_stats.values())
    for k, v in _time_stats.items():
        print(f"{k:12s}: {v:.3f} s  ({v/total*100:.1f}% of total)" if total > 0 else f"{k:12s}: {v:.3f} s")
    print(f"----------------------------------------------------------")
    print(f"Total transmission() time: {total:.3f} s")
    print("==========================================================\n")

