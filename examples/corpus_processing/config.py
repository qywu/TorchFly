#!/usr/bin/env python3
#
import logging, time, os, pdb

class _Config():
    def __init__(self):
        self.reg_init()

    def reg_init(self):

        self.seed = 222
        self.lr = 1
        self.meta_lr = 0.01
        self.filter_lr = 0.01

        self.epoch_num = 100
        self.val_period = 500
        self.batch_size = 32
        self.sample_num = 1
        self.sample_num_val = 100
        self.domain_num = 15

        self.pre_val_loss = 0
        self.max_val_acc = 0
        self.ear_stop_num = 10
        self.ear_stop_num_test = 30
        self.max_adapt_num = 5000

        self.csz = 32
        self.n_way = 5
        self.k_spt = 1
        self.k_qry = 15
        self.task_num = 4
        self.update_step = 1
        self.update_step_test = 30
        self.imgsz = 84
        self.imgc = 3

        self.clip = 3
        self.optim = 'sgd'

        self.train_batchsz = 1001


        self.mode = 'train'
        self.model_dir = ''

        self.cuda = True
        self.cuda_device = 0
        self.restart_all = True

        # # # params to generate data points
        self.d_sour_num = 20      # number of source domains
        self.d_targ_num = 1       # number of target domain
        self.train_num = 200     # number of training point in each domain
        self.test_num = 100
        self.val_num = 100
        self.support_num = self.sample_num

        # # # log file
        self.save_log = True
        self.log_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())



    def _init_logging_handler(self, log_dir = './log'):
        stderr_handler = logging.StreamHandler()
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        if self.save_log:
            log_path = os.path.join(log_dir, 'log_{}_{}.txt'.format(self.mode, self.log_time))
            file_handler = logging.FileHandler(log_path)
            logging.basicConfig(handlers=[stderr_handler, file_handler])
        else:
            logging.basicConfig(handlers=[stderr_handler])
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)

global_config = _Config()