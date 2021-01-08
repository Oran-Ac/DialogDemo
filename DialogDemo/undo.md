1.我们编码是将字符串处理后进行编码
2 在UMS-ResSel/models/pretrained_common/tokenization_utils.py / 中有add_special_token的操作
3.在UMS-ResSel/post_train/post_training.py /中train 为什么要改写BertForPreTraining使其返回各自的loss，，在train里再先求平均再加起来


1. 测试是不是字符编码是不是对的
2. 检查模型构建有无错误
