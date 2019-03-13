import matplotlib.pyplot as plt


def show(origin_map, gt_map, predict, index):
    figure, (origin, gt, pred) = plt.subplots(1, 3, figsize=(20, 4))
    origin.imshow(origin_map)
    origin.set_title("origin picture")
    gt.imshow(gt_map, cmap=plt.cm.jet)
    gt.set_title("gt map")
    pred.imshow(predict, cmap=plt.cm.jet)
    pred.set_title("prediction")
    plt.suptitle(str(index) + "th sample")
    plt.show()
    plt.close()


def show_phase2(origin_map, gt_map, predict_1, predict_2, index):
    figure, (origin, gt, pred_1, pred_2) = plt.subplots(1, 4, figsize=(20, 4))
    origin.imshow(origin_map)
    origin.set_title("origin picture")
    gt.imshow(gt_map, cmap=plt.cm.jet)
    gt.set_title("gt map")
    pred_1.imshow(predict_1, cmap=plt.cm.jet)
    pred_1.set_title("prediction_phase_1")
    pred_2.imshow(predict_2, cmap=plt.cm.jet)
    pred_2.set_title("prediction_phase_2")
    plt.suptitle(str(index) + "th sample")
    plt.show()
    plt.close()
