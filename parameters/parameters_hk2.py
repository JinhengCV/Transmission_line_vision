import numpy as np
import cv2
import struct  # 导入 struct 模块
import laspy

###需要自己实验，到底选取几个点，误差小####

point3s=np.array((
    [-17.7, -42.209, 42.228], [2.52, -50.92, 36.72], [-23.5, -40.46, 23.83],
    [4.5, -51.96, 23.7], [-21.8, -41.06, 11.78], [-1.66, -48.3, 6.59],
    [-56, 5.26, 43.2], [-37.9, -1.98, 41.398], [-47.8, 0.58, 32.296],
    [-35.8, -3, 23.5], [-54.3, 4.45, 18.5], [-37.878, -2.48, 12.9],
    [-51.4, -161, -27.5], [92.5, -126.8, 29.4], [-55.95, -2.998, -41]
    ),dtype=np.double)

point2s=np.array((
    [1127, 227], [1333, 283], [1063, 391],
    [1355, 402], [1076, 504], [1281, 561],
    [689, 290], [846, 305], [755, 374],
    [855, 447], [690, 487], [837, 538],  ###上面是两个电塔
    [1378, 873], [2412, 349], [658, 1016]
),dtype=np.double)



# def EPNP(point3s, point2s, camera_intrinsic, dist):
#     #dist=np.zeros((5,1))
#     found,r,t=cv2.solvePnP(point3s, point2s, camera_intrinsic, dist, None, None, False, cv2.SOLVEPNP_EPNP) #计算雷达相机外参,r-旋转向量，t-平移向量
#     # print(r)
#     r=cv2.Rodrigues(r)[0] #旋转向量转旋转矩阵####罗德里格斯(Rodrigues)旋转公式
#     # 使用计算出的R和T投影3D点
#     projected_points, _ = cv2.projectPoints(point3s, r, t, camera_intrinsic, dist)
#     projected_points = projected_points.reshape(-1, 2)
#     # 计算误差 (Mean Squared Error)
#     error = np.sqrt(np.mean(np.sum((projected_points - point2s) ** 2, axis=1)))
#     # 格式化输出 r 和 t
#     r = np.array(r).tolist()
#     t = t.flatten().tolist()
#     print('R=','\n',r,'\n','\n', 'T=','\n', t)
#     print('Projection Error (MSE):', error)

def EPNP(point3s, point2s, camera_intrinsic, dist):
# 使用 cv2.solvePnPRansac 来求解位姿
    found, r, t, inliers = cv2.solvePnPRansac(point3s, point2s, camera_intrinsic, dist
                                              , flags=cv2.SOLVEPNP_EPNP    # 
                                            )
    if found:
        r, t = cv2.solvePnPRefineLM(
            point3s, point2s, camera_intrinsic, dist, r, t,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-6)
        )


        # 将旋转向量转换为旋转矩阵
        r = cv2.Rodrigues(r)[0] #罗德里格斯(Rodrigues)旋转公式

        # 使用计算出的R和T投影3D点
        projected_points, _ = cv2.projectPoints(point3s, r, t, camera_intrinsic, dist)
        projected_points = projected_points.reshape(-1, 2)

        # 计算误差 (Mean Squared Error)
        error = np.sqrt(np.mean(np.sum((projected_points - point2s) ** 2, axis=1)))

        # 格式化输出 r 和 t
        r = np.array(r).tolist()
        t = t.flatten().tolist()

        print('R=','\n',r,'\n','\n', 'T=','\n', t)
        print('Inliers:', inliers.flatten())  # 展平内点数组以匹配期望输出格式
        print('Projection Error (MSE):', error)
        
        # return r, t, inliers
    else:
        print("solvePnPRansac failed to find a valid solution.")
        return None, None, None

def read_depth_map_from_binary(file_path):
    """
    从二进制文件中读取深度图和图像尺寸信息。
    
    参数:
    - file_path: 二进制文件的路径
    
    返回:
    - depth_map: 深度图数组
    """
    with open(file_path, 'rb') as f:
        # 读取图像尺寸信息
        height, width = struct.unpack('II', f.read(8))
        # 读取深度图数据并转换为 numpy 数组
        depth_map = np.fromfile(f, dtype=np.float32)
        depth_map = depth_map.reshape((height, width))
    return depth_map

def read_point_cloud(file_paths, shift):
    """
    从 LAS 文件中读取点云数据，并加上漂移量。
    """
    if isinstance(file_paths, str):
        file_paths = [file_paths]  # 如果是单个路径，转为列表

    all_points = []

    for file_path in file_paths:
        las = laspy.read(file_path)
        points = np.vstack((las.x, las.y, las.z)).T
        shifted_points = points + shift
        all_points.append(shifted_points)

    # 将所有点云数据合并到一个数组中
    combined_points = np.vstack(all_points)

    return combined_points

shift = np.array([-256914.10, -3365490.88, -53.18])  # 漂移量
##相机内参（matlab结果的转置）#1080P
# 对应图片地址"D:\Test_depth\images\50085.jpg"

camera_intrinsic = np.mat(np.array(([1.785072740345520e+03, 0, 1.251186131686387e+03],[0, 1.784492358648050e+03, 7.325446146606907e+02],[0,0,1]),dtype=float))
# 畸变系数 [k1,k2,p1,p2,k3](Radia, Tangen)
dist=np.array(([-0.415606568235958, 0.194606016994383, -2.715864007905328e-04, -2.943252670932223e-04, -0.047941656495039]),dtype=np.double).T


# EPNP(point3s, point2s, camera_intrinsic, dist)
depth_map = read_depth_map_from_binary('parameters/hk2_depth.bin')
# 读取并应用漂移量到点云
# file_paths = ['parameters/hk2_lines.las']
# linepoint = read_point_cloud(file_paths, shift)


segments_data = np.load('parameters/hk2_segments.npz', allow_pickle=True)
# 转换为标准浮点数组，避免 object 类型
segments = [np.asarray(seg, dtype=np.float64) for seg in segments_data["segments"]]
reps_all = [np.asarray(rep, dtype=np.float64) for rep in segments_data["reps_all"]]
reps_concat = np.vstack(reps_all).astype(np.float64)


r = [[0.8809881658564603, -0.4719836773531948, 0.033034223663120865], [0.07085604122766667, 0.0625806673381969, -0.9955215123225862], [0.4678026005083907, 0.8793833455255229, 0.08857572223767506]]

t = [[-23.96751928349953], [-7.7953361259304055], [252.65420403932868]]

optimization = 0  ####调节误差的参数
