import mmcv
import torch
import cv2
from mmcv.runner import load_checkpoint
from mmdetection.mmdet.models import build_detector
from mmdetection.mmdet.apis import inference_detector, show_result,init_detector
from eval_kit.detector import FaceDetectorBase
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel

class MMDetector(FaceDetectorBase):

    def __init__(self):
        """
        This is an example face detector used for runtime evaluation in WIDER Challenge 2019.
        It takes one BGR image and output a numpy array of face boudning boxes in the format of
        [left, top, width, height, confidence]

        Your own face detector should also follow this interface and implement the FaceDetectorBase interface
        Please find the detailed requirement of the face detector in FaceDetectorBase
        """
        super(MMDetector, self).__init__()
        self.detectors = [None, None, None]
       # self.cfg = mmcv.Config.fromfile('mmdetection/configs/widerface_resnet50.py')
       # self.cfg = mmcv.Config.fromfile('mmdetection/configs/resnet18_cascadercnn.py')
        #self.cfg = mmcv.Config.fromfile('mmdetection/configs/widerface_fp16_ga_retinanet_x101_fpn_1x.py')
        #self.cfg.model.pretrained = None

       # self.model=init_detector('mmdetection/configs/widerface_resnet50.py', checkpoint='mmdetection/work_dirs/widerface_resnet50_epoch21.pth', device='cuda:0')
       # self.model=init_detector('mmdetection/configs/resnet18_cascadercnn.py', checkpoint='mmdetection/work_dirs/resnet18_cascadercnn.pth', device='cuda:0')
        #self.model=init_detector('mmdetection/configs/resnet18_retinanet.py', checkpoint='mmdetection/work_dirs/resnet18_retinanet_epoch_10.pth', device='cuda:0')
       # self.model=init_detector('mmdetection/configs/widerface_resnet50.py', checkpoint='mmdetection/work_dirs/resnet50_cascadercnn.pth', device='cuda:0')
        #self.model=init_detector('resnet50_cascadercnn_light/widerface_resnet50_light.py', checkpoint='resnet50_cascadercnn_light/epoch_21.pth', device='cuda:0')
        self.model=init_detector('ga_retina_r50/widerface_fp16_ga_retinanet_r50_fpn_128.py', checkpoint='ga_retina_r50/epoch_21.pth', device='cuda:0')
        #if self.cfg.get('cudnn_benchmark', False):
        #self.model=init_detector('mv2_addons/mobilenet_lite_retina.py', checkpoint='mv2_addons/epoch_2.pth', device='cuda:0')
        #     torch.backends.cudnn.benchmark = True

        # self.model = build_detector(self.cfg.model, test_cfg=self.cfg.test_cfg)
        # _ = load_checkpoint( self.model, 'mmdetection/work_dirs/ssd_vgg/epoch_24.pth')
        
        #self.model = MMDistributedDataParallel(self.model.cuda())
        use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda:0' if use_cuda else 'cpu')

        # self.model.cfg = self.cfg
        img = cv2.imread('test.jpg')
        # with torch.no_grad():
        #     img = torch.from_numpy(img)
        result = inference_detector(self.model, img)

    def process_image(self, image):
        """
        :param image: a numpy.array representing one image with BGR colors
        :return: numpy.array of detection results in the format of
            [
                [left, top, width, height, confidence],
                ...
            ], dtype=np.float32
        """
        # collect results from all ranks
        self.model.eval()
        #result = self.model(return_loss=False, rescale=True,image)
        # with torch.no_grad():
        #     image = torch.from_numpy(image)
        result = inference_detector(self.model, image)
        #for i in range(result[0].shape[0]):
        #    left_top = (result[0][i,0], result[0][i,1])
        #    right_bottom = (result[0][i,2], result[0][i,3])
        #    cv2.rectangle(
        #        image, left_top, right_bottom,(0,255,0),10)
        #cv2.imwrite('./result.jpg',image)
        result[0][:,2] = result[0][:,2] - result[0][:,0] 
        result[0][:,3] = result[0][:,3] - result[0][:,1]
        #show_result(image, result,('background','person'))
        return result[0]
