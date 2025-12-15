import os, sys
import argparse
from queue import Queue
from concurrent.futures import ThreadPoolExecutor
from rknnlite.api import RKNNLite
import cv2

from utils.coco_utils import COCO_test_helper
import numpy as np

from trans_depth import transmission, get_alert_stats

# add path
realpath = os.path.abspath(__file__)
_sep = os.path.sep
realpath = realpath.split(_sep)
sys.path.append(os.path.join(realpath[0]+_sep, *realpath[1:realpath.index('rknn_model_zoo')+1]))

OBJ_THRESH = 0.25
NMS_THRESH = 0.45

# The follew two param is for map test
# OBJ_THRESH = 0.001
# NMS_THRESH = 0.65

IMG_SIZE = (640, 640)  # (width, height), such as (1280, 736)


CLASSES = ('crane', 'truck', 'car', 'jib', 'smoke', 'fire', 'nest', 'trash', 'kite', 'balloon')

coco_id_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

def _init_rknn_instance(rknn_path: str, core_id: int):
    rknn = RKNNLite()
    ret = rknn.load_rknn(rknn_path)
    if ret != 0:
        raise RuntimeError("Load RKNN failed")
    # 和 v5 样例一致：NPU 三核轮流绑定
    if core_id % 3 == 0:
        ret = rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0)
    elif core_id % 3 == 1:
        ret = rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_1)
    else:
        ret = rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_2)
    if ret != 0:
        raise RuntimeError("Init RKNN runtime failed")
    return rknn

class RKNNAsyncPool:
    """
    只做 NPU 推理（inputs=[img]），返回 outputs 列表。
    预处理/后处理/画图/写视频都还是走你原来的代码。
    """
    def __init__(self, rknn_path: str, tpes: int = 3):
        self.tpes = max(1, int(tpes))
        self.pool = ThreadPoolExecutor(max_workers=self.tpes)
        self.queue = Queue()
        # 初始化多个 RKNN 实例，绑定不同核心，提升并行度（同 v5）
        self.rknn_list = []
        for i in range(self.tpes):
            self.rknn_list.append(_init_rknn_instance(rknn_path, i))
        self._seq = 0

    def put(self, img_rgb_letterboxed):
        """
        异步提交一次 NPU 推理任务。img 要与原先 model.run([input_data]) 的输入一致：
        - RKNN 下本来就是 RGB + letterbox 后的 HxWx3（uint8）
        """
        rknn = self.rknn_list[self._seq % self.tpes]
        fut = self.pool.submit(lambda _rknn=rknn, _img=img_rgb_letterboxed: _rknn.inference(inputs=[_img]))
        self.queue.put(fut)
        self._seq += 1

    def get(self):
        """
        取回最早提交的一个推理结果：outputs(list)。
        """
        if self.queue.empty():
            return None, False
        fut = self.queue.get()
        return fut.result(), True

    def release(self):
        self.pool.shutdown(wait=True)
        for r in self.rknn_list:
            r.release()


def filter_boxes(boxes, box_confidences, box_class_probs):
    """Filter boxes with object threshold.
    """
    box_confidences = box_confidences.reshape(-1)
    candidate, class_num = box_class_probs.shape

    class_max_score = np.max(box_class_probs, axis=-1)
    classes = np.argmax(box_class_probs, axis=-1)

    if class_num==1:
        _class_pos = np.where(box_confidences >= OBJ_THRESH)
        scores = (box_confidences)[_class_pos]
    else:
        _class_pos = np.where(class_max_score* box_confidences >= OBJ_THRESH)
        scores = (class_max_score* box_confidences)[_class_pos]

    boxes = boxes[_class_pos]
    classes = classes[_class_pos]

    return boxes, classes, scores

def nms_boxes(boxes, scores):
    """Suppress non-maximal boxes.
    # Returns
        keep: ndarray, index of effective boxes.
    """
    x = boxes[:, 0]
    y = boxes[:, 1]
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]

    areas = w * h
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x[i], x[order[1:]])
        yy1 = np.maximum(y[i], y[order[1:]])
        xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
        yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])

        w1 = np.maximum(0.0, xx2 - xx1 + 0.00001)
        h1 = np.maximum(0.0, yy2 - yy1 + 0.00001)
        inter = w1 * h1

        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= NMS_THRESH)[0]
        order = order[inds + 1]
    keep = np.array(keep)
    return keep


def box_process(position, anchors):
    grid_h, grid_w = position.shape[2:4]
    col, row = np.meshgrid(np.arange(0, grid_w), np.arange(0, grid_h))
    col = col.reshape(1, 1, grid_h, grid_w)
    row = row.reshape(1, 1, grid_h, grid_w)
    grid = np.concatenate((col, row), axis=1)
    stride = np.array([IMG_SIZE[1]//grid_h, IMG_SIZE[0]//grid_w]).reshape(1,2,1,1)

    col = col.repeat(len(anchors), axis=0)
    row = row.repeat(len(anchors), axis=0)
    anchors = np.array(anchors)
    anchors = anchors.reshape(*anchors.shape, 1, 1)

    box_xy = position[:,:2,:,:]*2 - 0.5
    box_wh = pow(position[:,2:4,:,:]*2, 2) * anchors

    box_xy += grid
    box_xy *= stride
    box = np.concatenate((box_xy, box_wh), axis=1)

    # Convert [c_x, c_y, w, h] to [x1, y1, x2, y2]
    xyxy = np.copy(box)
    xyxy[:, 0, :, :] = box[:, 0, :, :] - box[:, 2, :, :]/ 2  # top left x
    xyxy[:, 1, :, :] = box[:, 1, :, :] - box[:, 3, :, :]/ 2  # top left y
    xyxy[:, 2, :, :] = box[:, 0, :, :] + box[:, 2, :, :]/ 2  # bottom right x
    xyxy[:, 3, :, :] = box[:, 1, :, :] + box[:, 3, :, :]/ 2  # bottom right y

    return xyxy

def post_process(input_data, anchors):
    boxes, scores, classes_conf = [], [], []
    # 1*255*h*w -> 3*85*h*w
    input_data = [_in.reshape([len(anchors[0]),-1]+list(_in.shape[-2:])) for _in in input_data]
    for i in range(len(input_data)):
        boxes.append(box_process(input_data[i][:,:4,:,:], anchors[i]))
        scores.append(input_data[i][:,4:5,:,:])
        classes_conf.append(input_data[i][:,5:,:,:])

    def sp_flatten(_in):
        ch = _in.shape[1]
        _in = _in.transpose(0,2,3,1)
        return _in.reshape(-1, ch)

    boxes = [sp_flatten(_v) for _v in boxes]
    classes_conf = [sp_flatten(_v) for _v in classes_conf]
    scores = [sp_flatten(_v) for _v in scores]

    boxes = np.concatenate(boxes)
    classes_conf = np.concatenate(classes_conf)
    scores = np.concatenate(scores)

    # filter according to threshold
    boxes, classes, scores = filter_boxes(boxes, scores, classes_conf)

    # nms
    nboxes, nclasses, nscores = [], [], []

    for c in set(classes):
        inds = np.where(classes == c)
        b = boxes[inds]
        c = classes[inds]
        s = scores[inds]
        keep = nms_boxes(b, s)

        if len(keep) != 0:
            nboxes.append(b[keep])
            nclasses.append(c[keep])
            nscores.append(s[keep])

    # print('AAAAAAAAAAA',boxes, classes, scores)
    if not nclasses and not nscores:
        return None, None, None

    boxes = np.concatenate(nboxes)
    classes = np.concatenate(nclasses)
    scores = np.concatenate(nscores)

    return boxes, classes, scores


def draw(image, boxes, scores, classes):
    # for box, score, cl in zip(boxes, scores, classes):
    #     top, left, right, bottom = [int(_b) for _b in box]
    #     # print("%s @ (%d %d %d %d) %.3f" % (CLASSES[cl], top, left, right, bottom, score))
    #     cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)
    #     cv2.putText(image, '{0} {1:.2f}'.format(CLASSES[cl], score),
    #                 (top, left - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    # return image
    npscores = scores[:,np.newaxis]
    npclasses = classes[:,np.newaxis]
    npboxes = np.concatenate((npclasses, npscores, boxes),axis=1)
    im = transmission(image, npboxes, CLASSES, threshold=0.5)
    # class_three = [x for x in label if x != 'jib']
    # print('class is:', class_three, ' corresponding distance is', mindis)
    return im

def setup_model(args):
    model_path = args.model_path
    if model_path.endswith('.pt') or model_path.endswith('.torchscript'):
        platform = 'pytorch'
        from common.framework_executor.pytorch_executor import Torch_model_container
        model = Torch_model_container(args.model_path)
    elif model_path.endswith('.rknn'):
        platform = 'rknn'
        from common.framework_executor.rknn_executor import RKNN_model_container 
        model = RKNN_model_container(args.model_path, args.target, args.device_id)
    
    elif model_path.endswith('onnx'):
        platform = 'onnx'
        from common.framework_executor.onnx_executor import ONNX_model_container
        model = ONNX_model_container(args.model_path)
    else:
        assert False, "{} is not rknn/pytorch/onnx model".format(model_path)
    print('Model-{} is {} model, starting val'.format(model_path, platform))
    return model, platform

def img_check(path):
    img_type = ['.jpg', '.jpeg', '.png', '.bmp']
    for _type in img_type:
        if path.endswith(_type) or path.endswith(_type.upper()):
            return True
    return False

import time



def process_video(model, video_path, output_path):
    prev_time = time.perf_counter()   # 用于每帧真实FPS计算
    fps_smooth = 0.0                  # 平滑显示用
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, 25, (2560, 1440))

    co_helper = COCO_test_helper(enable_letter_box=True)

    # ======== 创建异步推理池（多实例多核） ========
    TPEs = 3
    rknn_path_for_pool = args.model_path
    pool = RKNNAsyncPool(rknn_path=rknn_path_for_pool, tpes=TPEs)

    # ======== warmup ========
    frames = 0
    initTime = time.time()  # 与v5一致，用于打印总平均帧率
    warm_cnt = 0
    while cap.isOpened() and warm_cnt < (TPEs + 1):
        ok, frame = cap.read()
        if not ok:
            break
        img = co_helper.letter_box(im=frame.copy(), new_shape=(IMG_SIZE[1], IMG_SIZE[0]), pad_color=(0, 0, 0))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pool.put(img)
        warm_cnt += 1

    # ======== 统计变量 ========
    total_detection_time = 0.0
    total_ranged_time = 0.0
    wall_start = None

    # ======== 主循环 ========
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        frames += 1
        t_frame0 = time.perf_counter()
        if wall_start is None:
            wall_start = t_frame0  # 第一次帧的开始时间作为总时间起点

        # === 预处理 + 异步推理 ===
        img = co_helper.letter_box(im=frame.copy(), new_shape=(IMG_SIZE[1], IMG_SIZE[0]), pad_color=(0, 0, 0))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pool.put(img)

        # === 检测阶段 ===
        t_det0 = time.perf_counter()
        outputs, ok2 = pool.get()
        if not ok2:
            break
        boxes, classes, scores = post_process(outputs, anchors)
        det_dt = time.perf_counter() - t_det0
        total_detection_time += det_dt

        # === Range-D测距阶段 ===
        t_rd0 = time.perf_counter()
        if boxes is not None:
            real_boxes = co_helper.get_real_box(boxes)
            frame = np.array(draw(frame, real_boxes, scores, classes))  # draw里调用transmission
            detected_classes = [CLASSES[c] for c in classes]
        else:
            detected_classes = []
        rd_dt = time.perf_counter() - t_rd0
        total_ranged_time += rd_dt

        # === End-to-End耗时 ===
        e2e_dt = time.perf_counter() - t_frame0

        # === 实时FPS计算 ===
        now_time = time.perf_counter()
        frame_interval = now_time - prev_time
        prev_time = now_time
        frame_fps = 1.0 / (frame_interval + 1e-6)
        fps_smooth = (0.9 * fps_smooth + 0.1 * frame_fps) if fps_smooth > 0 else frame_fps

        # === 日志打印 ===
        if boxes is not None:
            print(f"[Frame {frames}] Detected: {detected_classes} | "
                  f"Det:{det_dt*1000:.1f} ms | RangeD:{rd_dt*1000:.1f} ms | "
                  f"End2End:{e2e_dt*1000:.1f} ms | FPS:{fps_smooth:.2f}")
        else:
            print(f"[Frame {frames}] No object | "
                  f"Det:{det_dt*1000:.1f} ms | RangeD:{rd_dt*1000:.1f} ms | "
                  f"End2End:{e2e_dt*1000:.1f} ms | FPS:{fps_smooth:.2f}")

        # === 在图像上叠加FPS ===
        fps_text = f"FPS: {fps_smooth:.2f}"
        cv2.putText(frame, fps_text, (frame.shape[1] - 220, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 4, cv2.LINE_AA)

        if args.img_save:
            writer.write(frame)

        if args.img_show:
            resized_frame = cv2.resize(frame, (2560, 1440))
            cv2.imshow("Processed Frame", resized_frame)
            
        # if args.img_show:
        #     resized_frame = cv2.resize(frame, (2560, 1440))
        #     window_name = "Processed Frame"
        #     cv2.imshow(window_name, resized_frame)

        #     # === 让窗口居中显示 ===
        #     # 获取屏幕尺寸（部分系统上可能返回0，用默认值兜底）
        #     screen_width  = cv2.getWindowImageRect(window_name)[2]
        #     screen_height = cv2.getWindowImageRect(window_name)[3]
        #     # 如果返回值异常（比如0），可手动设置屏幕分辨率
        #     if screen_width == 0 or screen_height == 0:
        #         screen_width, screen_height = 2560, 1440  # 可改成你实际的显示器分辨率

        #     # 计算窗口居中位置
        #     win_x = (screen_width  - resized_frame.shape[1]) // 2
        #     win_y = (screen_height - resized_frame.shape[0]) // 2
        #     cv2.moveWindow(window_name, win_x, win_y)
            
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # ======== 收尾与统计 ========
    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    pool.release()

    wall_end = time.perf_counter() if wall_start is not None else None

    # ======== 汇总打印 ========
    print("\n=========== Runtime Performance Summary ===========")
    if frames > 0 and wall_start and wall_end:
        wall_elapsed = wall_end - wall_start
        avg_det_ms = total_detection_time / frames * 1000
        avg_rd_ms = total_ranged_time / frames * 1000
        avg_e2e_ms = wall_elapsed / frames * 1000
        sustained_fps = frames / wall_elapsed

        print(f"Total processed frames : {frames}")
        print(f"Avg Detection time      : {avg_det_ms:.2f} ms")
        print(f"Avg Range-D time        : {avg_rd_ms:.2f} ms")
        print(f"Avg End-to-End latency  : {avg_e2e_ms:.2f} ms")
        print(f"Sustained FPS (video)   : {sustained_fps:.2f}")
    else:
        print("No frames processed.")
    print("===================================================\n")
    alert_count, avg_delay = get_alert_stats()
    print(f"\n=========== Alert Statistics ===========")
    print(f"Total alert frames       : {alert_count}")
    print(f"Average alert delay time : {avg_delay*1000:.2f} ms")
    print("========================================\n")

    # === 总平均帧率 ===
    # print("总平均帧率\t", frames / (time.time() - initTime))




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    # basic params
    parser.add_argument('--model_path', type=str, required= True, help='model path, could be .pt or .rknn file')
    parser.add_argument('--target', type=str, default='rk3566', help='target RKNPU platform')
    parser.add_argument('--device_id', type=str, default=None, help='device id')
    
    parser.add_argument('--img_show', action='store_true', default=False, help='draw the result and show')
    parser.add_argument('--img_save', action='store_true', default=False, help='save the result')

    # data params
    parser.add_argument('--anno_json', type=str, default='../../../datasets/COCO/annotations/instances_val2017.json', help='coco annotation path')
    # coco val folder: '../../../datasets/COCO//val2017'
    parser.add_argument('--img_folder', type=str, default='../model', help='img folder path')
    parser.add_argument('--coco_map_test', action='store_true', help='enable coco map test')
    parser.add_argument('--anchors', type=str, default='../model/anchors_yolov7.txt', help='target to anchor file, only yolov5, yolov7 need this param')

    args = parser.parse_args()

    with open(args.anchors, 'r') as f:
        values = [float(_v) for _v in f.readlines()]
        anchors = np.array(values).reshape(3,-1,2).tolist()
    print("use anchors from '{}', which is {}".format(args.anchors, anchors))
    
    # init model
    model, platform = setup_model(args)

    ## Open the video file
    # video_path = "/Disk500G/LJH/RKNN_toolkit2/examples/onnx/yolov5/demo/video20s-paper.mp4"
    video_path = "/Disk500G/LJH/YOLOv7/result/video/NCvideo20s.mp4"
    # 60.208.119.18_01_20220523160425784/60.208.119.18_01_20220524102753937/60.208.119.18_01_20220529165646236/60.208.119.18_01_20220620160208749
    # video_path = '/Disk500G/LJH/YOLOv7/result/video20s-paper_3s.mp4'
    output_path = '/Disk500G/LJH/YOLOv7/result/demoout.mp4'
    # Process the video

    # === 1) 在调用前记录起始时间 ===
    import time
    _start_all = time.perf_counter()

    # 原有调用（保持不变）
    process_video(model, video_path, output_path)
    # from trans_depth import print_transmission_time_summary
    # print_transmission_time_summary()

    # === 2) 结束后计算总耗时 ===
    _total_all = time.perf_counter() - _start_all

    # === 3) 读取输入视频的总帧数，用它来计算平均 FPS（整体端到端FPS）===
    import cv2
    _cap_tmp = cv2.VideoCapture(video_path)
    _frame_count = int(_cap_tmp.get(cv2.CAP_PROP_FRAME_COUNT))
    _cap_tmp.release()

    if _frame_count > 0 and _total_all > 0:
        _avg_fps = _frame_count / _total_all
    else:
        _avg_fps = 0.0

    print("\n=========== Average Inference FPS ===========")
    print(f"End-to-end average FPS: {_avg_fps:.2f}")
    print("=========================================================\n")




# python ./rknn_model_zoo/models/CV/object_detection/yolo/RKNN_python_demo/v7demo-mul.py --model_path ./model_cvt/RK3588/v7mul.rknn --target rk3588 --anchors ./model_cvt/RK3588/RK_anchors_mul.txt --img_show --img_save --img_folder ./rknn_model_zoo/common/rknn_converter/JPEGImages
# python ./rknn_model_zoo/models/CV/object_detection/yolo/RKNN_python_demo/yolo_map_test_rknn.py --model yolov7 --model_path ./runs/train/exp197/weights/best.onnx --target rk3588 --anchors RK_anchors.txt --img_show --img_folder ./rknn_model_zoo/common/rknn_converter/JPEGImages
# python ./rknn_model_zoo/models/CV/object_detection/yolo/RKNN_python_demo/yolo_map_test_rknn.py --model yolov7 --model_path ./weights/yolov7-tiny.onnx --target rk3588 --anchors RK_anchors.txt --img_show --img_folder ./rknn_model_zoo/common/rknn_converter/JPEGImages
#python ./rknn_model_zoo/models/CV/object_detection/yolo/RKNN_python_demo/mul-threaded/main.py 
