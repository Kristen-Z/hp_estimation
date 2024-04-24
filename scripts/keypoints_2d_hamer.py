import os
import cv2
import numpy as np

import json

def frame_preprocess(path):
    stream = cv2.VideoCapture(path)
    assert stream.isOpened(), 'Cannot capture source'

    datalen = int(stream.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_imgs = []
    im_names = []
    for k in range(datalen):
        (grabbed, frame) = stream.read()
        # if the `grabbed` boolean is `False`, then we have
        # reached the end of the video file
        if not grabbed:
            stream.release()
            break

        orig_imgs.append(frame[:, :, ::-1])
        im_names.append(f'{k:08d}' + '.jpg')

    stream.release()

    return im_names, orig_imgs

def write_direct_json(img_name, bboxes, kpts, outputpath):
    '''
    all_result: result dict of predictions
    outputpath: output directory
    '''
    name = os.path.basename(img_name)
    if not os.path.exists(outputpath):
        os.mkdir(outputpath)
    with open(os.path.join(outputpath,name.split('.')[0]+'.json'),'w') as json_file:
        json_file.write(json.dumps(kpts))#.reshape(-1, 3).tolist()))

    print('write json to', os.path.join(outputpath,name.split('.')[0]+'.json'))
    bbox_dir = outputpath.replace('keypoints_2d','bboxes')
    os.makedirs(bbox_dir, exist_ok=True)
    with open(os.path.join(bbox_dir,name.split('.')[0]+'.json'),'w') as json_file:
        json_file.write(json.dumps(bboxes))


def keypoints_2d(video, out_folder, cpm, detector, conf_threshold=0.3):

    # Make output directory if it does not exist
    os.makedirs(out_folder, exist_ok=True)
    im_names, orig_imgs = frame_preprocess(video)

    # Iterate over all images in folder
    for i, frame in enumerate(orig_imgs):
        if not os.path.exists(os.path.join(out_folder, im_names[i].split('.')[0]+'.json')):
            print('processing', os.path.join(out_folder, im_names[i].split('.')[0]+'.json'))
            # Detect humans in image
            det_out = detector(frame)
            img = frame.copy()[:, :, ::-1]

            det_instances = det_out['instances']
            valid_idx = (det_instances.pred_classes==0) & (det_instances.scores > 0.5)
            pred_bboxes=det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
            pred_scores=det_instances.scores[valid_idx].cpu().numpy()

            # Detect human keypoints for each person
            vitposes_out = cpm.predict_pose(
                img,
                [np.concatenate([pred_bboxes, pred_scores[:, None]], axis=1)],
            )

            bboxes = []

            # Use hands based on hand keypoint detections
            for vitposes in vitposes_out:
                left_hand_keyp = vitposes['keypoints'][-42:-21]
                right_hand_keyp = vitposes['keypoints'][-21:]

                # Rejecting not confident detections
                # keyp = left_hand_keyp
                # valid = keyp[:,2] > 0.5
                # if sum(valid) > 3:
                #     bbox = [keyp[valid,0].min(), keyp[valid,1].min(), keyp[valid,0].max(), keyp[valid,1].max()]
                #     bboxes.append(bbox)
                keyp = right_hand_keyp
                # valid = keyp[:,2] > conf_threshold
                # if sum(valid) > 3:
                #     bbox = [keyp[valid,0].min(), keyp[valid,1].min(), keyp[valid,0].max(), keyp[valid,1].max()]
                #     bboxes.append(bbox)
                bbox = [keyp[:,0].min(), keyp[:,1].min(), keyp[:,0].max(), keyp[:,1].max()]
                bboxes.append(bbox)
            
            if len(bboxes) == 0:
                write_direct_json(im_names[i], [[0.,0.,0.,0.]], np.zeros((21,3)).tolist(), out_folder)
                continue

            boxes = np.stack(bboxes)

            write_direct_json(im_names[i], boxes.tolist(), keyp.tolist(), out_folder)

        # print('boxes', boxes.shape, boxes, right) [1, 4]
        # print('hand kpt', right_hand_keyp.shape, left_hand_keyp.shape) [21, 3]
        # Run reconstruction on all detected hands