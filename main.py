import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import contextlib
import wandb
import warnings
import copy

from models.builder import MODEL_GETTER
from data.dataset import build_loader, TestImageDataset
from utils.costom_logger import timeLogger
from utils.config_utils import load_yaml, build_record_folder, get_args
from utils.lr_schedule import cosine_decay, adjust_lr, get_lr
from eval import evaluate, cal_train_metrics, suppression

warnings.simplefilter("ignore")


def eval_freq_schedule(args, epoch: int):
    if epoch >= args.max_epochs * 0.95:
        args.eval_freq = 1
    elif epoch >= args.max_epochs * 0.9:
        args.eval_freq = 1
    elif epoch >= args.max_epochs * 0.8:
        args.eval_freq = 2


def set_environment(args, tlogger):

    print("Setting Environment...")

    args.device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu")

    # = = = =  Dataset and Data Loader = = = =
    tlogger.print("Building Dataloader....")

    train_loader, val_loader = build_loader(args)

    if train_loader is None and val_loader is None:
        raise ValueError("Find nothing to train or evaluate.")

    if train_loader is not None:
        print("    Train Samples: {} (batch: {})".format(
            len(train_loader.dataset), len(train_loader)))
    else:
        # raise ValueError("Build train loader fail, please provide legal path.")
        print("    Train Samples: 0 ~~~~~> [Only Evaluation]")
    if val_loader is not None:
        print("    Validation Samples: {} (batch: {})".format(
            len(val_loader.dataset), len(val_loader)))
    else:
        print("    Validation Samples: 0 ~~~~~> [Only Training]")
    tlogger.print()

    # = = = =  Model = = = =
    tlogger.print("Building Model....")
    model = MODEL_GETTER[args.model_name](
        use_fpn=args.use_fpn,
        fpn_size=args.fpn_size,
        use_selection=args.use_selection,
        num_classes=args.num_classes,
        num_selects=args.num_selects,
        use_combiner=args.use_combiner,
    )  # about return_nodes, we use our default setting
    if args.pretrained is not None:
        checkpoint = torch.load(
            args.pretrained, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model'])
        start_epoch = checkpoint['epoch']
    else:
        start_epoch = 0

    # model = torch.nn.DataParallel(model, device_ids=None) # device_ids : None --> use all gpus.
    model.to(args.device)
    tlogger.print()

    """
    if you have multi-gpu device, you can use torch.nn.DataParallel in single-machine multi-GPU 
    situation and use torch.nn.parallel.DistributedDataParallel to use multi-process parallelism.
    more detail: https://pytorch.org/tutorials/beginner/dist_overview.html
    """

    if train_loader is None:
        return train_loader, val_loader, model, None, None, None, None

    # = = = =  Optimizer = = = =
    tlogger.print("Building Optimizer....")
    if args.optimizer == "SGD":
        optimizer = torch.optim.SGD(model.parameters(
        ), lr=args.max_lr, nesterov=True, momentum=0.9, weight_decay=args.wdecay)
    # elif args.optimizer == "AdamW":
    #     optimizer = torch.optim.AdamW(model.parameters(), lr=args.max_lr)
    elif args.optimizer == "AdamW":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.max_lr,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=args.wdecay
        )
    if args.pretrained is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])

    tlogger.print()

    schedule = cosine_decay(args, len(train_loader))

    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler()
        amp_context = torch.cuda.amp.autocast
    else:
        scaler = None
        amp_context = contextlib.nullcontext

    return train_loader, val_loader, model, optimizer, schedule, scaler, amp_context, start_epoch


def train(args, epoch, model, scaler, amp_context, optimizer, schedule, train_loader):

    optimizer.zero_grad()
    total_batchs = len(train_loader)  # just for log
    show_progress = [x/10 for x in range(11)]  # just for log
    progress_i = 0

    # temperature = 2 ** (epoch // 10 - 1)
    temperature = 0.5 ** (epoch // 10) * args.temperature
    # temperature = args.temperature

    n_left_batchs = len(train_loader) % args.update_freq

    for batch_id, (ids, datas, labels) in enumerate(train_loader):
        model.train()
        """ = = = = adjust learning rate = = = = """
        iterations = epoch * len(train_loader) + batch_id
        adjust_lr(iterations, optimizer, schedule)

        # temperature = (args.temperature - 1) * (get_lr(optimizer) / args.max_lr) + 1

        batch_size = labels.size(0)

        """ = = = = forward and calculate loss = = = = """
        datas, labels = datas.to(args.device), labels.to(args.device)

        with amp_context():
            """
            [Model Return]
                FPN + Selector + Combiner --> return 'layer1', 'layer2', 'layer3', 'layer4', ...(depend on your setting)
                    'preds_0', 'preds_1', 'comb_outs'
                FPN + Selector --> return 'layer1', 'layer2', 'layer3', 'layer4', ...(depend on your setting)
                    'preds_0', 'preds_1'
                FPN --> return 'layer1', 'layer2', 'layer3', 'layer4' (depend on your setting)
                ~ --> return 'ori_out'

            [Retuen Tensor]
                'preds_0': logit has not been selected by Selector.
                'preds_1': logit has been selected by Selector.
                'comb_outs': The prediction of combiner.
            """
            outs = model(datas)

            loss = 0.
            for name in outs:

                if "FPN1_" in name:
                    if args.lambda_b0 != 0:
                        aux_name = name.replace("FPN1_", "")
                        gt_score_map = outs[aux_name].detach()
                        thres = torch.Tensor(
                            model.selector.thresholds[aux_name])
                        gt_score_map = suppression(
                            gt_score_map, thres, temperature)
                        logit = F.log_softmax(outs[name] / temperature, dim=-1)
                        loss_b0 = nn.KLDivLoss()(logit, gt_score_map)
                        loss += args.lambda_b0 * loss_b0
                    else:
                        loss_b0 = 0.0

                elif "select_" in name:
                    if not args.use_selection:
                        raise ValueError("Selector not use here.")
                    if args.lambda_s != 0:
                        S = outs[name].size(1)
                        logit = outs[name].view(-1,
                                                args.num_classes).contiguous()
                        loss_s = nn.CrossEntropyLoss()(logit,
                                                       labels.unsqueeze(1).repeat(1, S).flatten(0))
                        loss += args.lambda_s * loss_s
                    else:
                        loss_s = 0.0

                elif "drop_" in name:
                    if not args.use_selection:
                        raise ValueError("Selector not use here.")

                    if args.lambda_n != 0:
                        S = outs[name].size(1)
                        logit = outs[name].view(-1,
                                                args.num_classes).contiguous()
                        n_preds = nn.Tanh()(logit)
                        labels_0 = torch.zeros(
                            [batch_size * S, args.num_classes]) - 1
                        labels_0 = labels_0.to(args.device)
                        loss_n = nn.MSELoss()(n_preds, labels_0)
                        loss += args.lambda_n * loss_n
                    else:
                        loss_n = 0.0

                elif "layer" in name:
                    if not args.use_fpn:
                        raise ValueError("FPN not use here.")
                    if args.lambda_b != 0:
                        # here using 'layer1'~'layer4' is default setting, you can change to your own
                        loss_b = nn.CrossEntropyLoss()(
                            outs[name].mean(1), labels)
                        loss += args.lambda_b * loss_b
                    else:
                        loss_b = 0.0

                elif "comb_outs" in name:
                    if not args.use_combiner:
                        raise ValueError("Combiner not use here.")

                    if args.lambda_c != 0:
                        loss_c = nn.CrossEntropyLoss()(outs[name], labels)
                        loss += args.lambda_c * loss_c

                elif "ori_out" in name:
                    loss_ori = F.cross_entropy(outs[name], labels)
                    loss += loss_ori

            if batch_id < len(train_loader) - n_left_batchs:
                loss /= args.update_freq
            else:
                loss /= n_left_batchs

        """ = = = = calculate gradient = = = = """
        if args.use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        """ = = = = update model = = = = """
        if (batch_id + 1) % args.update_freq == 0 or (batch_id + 1) == len(train_loader):
            if args.use_amp:
                scaler.step(optimizer)
                scaler.update()  # next batch
            else:
                optimizer.step()
            optimizer.zero_grad()

        """ log (MISC) """
        if args.use_wandb and ((batch_id + 1) % args.log_freq == 0):
            model.eval()
            msg = {}
            msg['info/epoch'] = epoch + 1
            msg['info/lr'] = get_lr(optimizer)
            cal_train_metrics(args, msg, outs, labels,
                              batch_size, model.selector.thresholds)
            wandb.log(msg)

        train_progress = (batch_id + 1) / total_batchs
        # print(train_progress, show_progress[progress_i])
        if train_progress > show_progress[progress_i]:
            print(
                ".."+str(int(show_progress[progress_i] * 100)) + "%", end='', flush=True)
            progress_i += 1


def main(args, tlogger):
    """
    save model last.pt and best.pt
    """

    train_loader, val_loader, model, optimizer, schedule, scaler, amp_context, start_epoch = set_environment(
        args, tlogger)

    best_acc = 0.0
    best_eval_name = "null"

    if args.use_wandb:
        wandb.init(entity=args.wandb_entity,
                   project=args.project_name,
                   name=args.exp_name,
                   config=args)
        wandb.run.summary["best_acc"] = best_acc
        wandb.run.summary["best_eval_name"] = best_eval_name
        wandb.run.summary["best_epoch"] = 0

    for epoch in range(start_epoch, args.max_epochs):

        """
        Train
        """
        if train_loader is not None:
            tlogger.print("Start Training {} Epoch".format(epoch+1))
            train(args, epoch, model, scaler, amp_context,
                  optimizer, schedule, train_loader)
            tlogger.print()
        else:
            from eval import eval_and_save
            eval_and_save(args, model, val_loader)
            break

        eval_freq_schedule(args, epoch)

        model_to_save = model.module if hasattr(model, "module") else model
        checkpoint = {"model": model_to_save.state_dict(
        ), "optimizer": optimizer.state_dict(), "epoch": epoch}
        torch.save(checkpoint, args.save_dir + "backup/last.pt")

        if epoch == 0 or (epoch + 1) % args.eval_freq == 0:
            """
            Evaluation
            """
            acc = -1
            if val_loader is not None:
                tlogger.print("Start Evaluating {} Epoch".format(epoch + 1))
                acc, eval_name, accs = evaluate(args, model, val_loader)
                tlogger.print("....BEST_ACC: {}% ({}%)".format(
                    max(acc, best_acc), acc))
                tlogger.print()

            if args.use_wandb:
                wandb.log(accs)

            if acc > best_acc:
                best_acc = acc
                best_eval_name = eval_name
                torch.save(checkpoint, args.save_dir + "backup/best.pt")
            if args.use_wandb:
                wandb.run.summary["best_acc"] = best_acc
                wandb.run.summary["best_eval_name"] = best_eval_name
                wandb.run.summary["best_epoch"] = epoch + 1


class_names = {0: '001.Black_footed_Albatross', 1: '002.Laysan_Albatross', 2: '003.Sooty_Albatross', 3: '004.Groove_billed_Ani', 4: '005.Crested_Auklet', 5: '006.Least_Auklet', 6: '007.Parakeet_Auklet', 7: '008.Rhinoceros_Auklet', 8: '009.Brewer_Blackbird', 9: '010.Red_winged_Blackbird', 10: '011.Rusty_Blackbird', 11: '012.Yellow_headed_Blackbird', 12: '013.Bobolink', 13: '014.Indigo_Bunting', 14: '015.Lazuli_Bunting', 15: '016.Painted_Bunting', 16: '017.Cardinal', 17: '018.Spotted_Catbird', 18: '019.Gray_Catbird', 19: '020.Yellow_breasted_Chat', 20: '021.Eastern_Towhee', 21: '022.Chuck_will_Widow', 22: '023.Brandt_Cormorant', 23: '024.Red_faced_Cormorant', 24: '025.Pelagic_Cormorant', 25: '026.Bronzed_Cowbird', 26: '027.Shiny_Cowbird', 27: '028.Brown_Creeper', 28: '029.American_Crow', 29: '030.Fish_Crow', 30: '031.Black_billed_Cuckoo', 31: '032.Mangrove_Cuckoo', 32: '033.Yellow_billed_Cuckoo', 33: '034.Gray_crowned_Rosy_Finch', 34: '035.Purple_Finch', 35: '036.Northern_Flicker', 36: '037.Acadian_Flycatcher', 37: '038.Great_Crested_Flycatcher', 38: '039.Least_Flycatcher', 39: '040.Olive_sided_Flycatcher', 40: '041.Scissor_tailed_Flycatcher', 41: '042.Vermilion_Flycatcher', 42: '043.Yellow_bellied_Flycatcher', 43: '044.Frigatebird', 44: '045.Northern_Fulmar', 45: '046.Gadwall', 46: '047.American_Goldfinch', 47: '048.European_Goldfinch', 48: '049.Boat_tailed_Grackle', 49: '050.Eared_Grebe', 50: '051.Horned_Grebe', 51: '052.Pied_billed_Grebe', 52: '053.Western_Grebe', 53: '054.Blue_Grosbeak', 54: '055.Evening_Grosbeak', 55: '056.Pine_Grosbeak', 56: '057.Rose_breasted_Grosbeak', 57: '058.Pigeon_Guillemot', 58: '059.California_Gull', 59: '060.Glaucous_winged_Gull', 60: '061.Heermann_Gull', 61: '062.Herring_Gull', 62: '063.Ivory_Gull', 63: '064.Ring_billed_Gull', 64: '065.Slaty_backed_Gull', 65: '066.Western_Gull', 66: '067.Anna_Hummingbird', 67: '068.Ruby_throated_Hummingbird', 68: '069.Rufous_Hummingbird', 69: '070.Green_Violetear', 70: '071.Long_tailed_Jaeger', 71: '072.Pomarine_Jaeger', 72: '073.Blue_Jay', 73: '074.Florida_Jay', 74: '075.Green_Jay', 75: '076.Dark_eyed_Junco', 76: '077.Tropical_Kingbird', 77: '078.Gray_Kingbird', 78: '079.Belted_Kingfisher', 79: '080.Green_Kingfisher', 80: '081.Pied_Kingfisher', 81: '082.Ringed_Kingfisher', 82: '083.White_breasted_Kingfisher', 83: '084.Red_legged_Kittiwake', 84: '085.Horned_Lark', 85: '086.Pacific_Loon', 86: '087.Mallard', 87: '088.Western_Meadowlark', 88: '089.Hooded_Merganser', 89: '090.Red_breasted_Merganser', 90: '091.Mockingbird', 91: '092.Nighthawk', 92: '093.Clark_Nutcracker', 93: '094.White_breasted_Nuthatch', 94: '095.Baltimore_Oriole', 95: '096.Hooded_Oriole', 96: '097.Orchard_Oriole', 97: '098.Scott_Oriole', 98: '099.Ovenbird', 99: '100.Brown_Pelican', 100: '101.White_Pelican', 101: '102.Western_Wood_Pewee',
               102: '103.Sayornis', 103: '104.American_Pipit', 104: '105.Whip_poor_Will', 105: '106.Horned_Puffin', 106: '107.Common_Raven', 107: '108.White_necked_Raven', 108: '109.American_Redstart', 109: '110.Geococcyx', 110: '111.Loggerhead_Shrike', 111: '112.Great_Grey_Shrike', 112: '113.Baird_Sparrow', 113: '114.Black_throated_Sparrow', 114: '115.Brewer_Sparrow', 115: '116.Chipping_Sparrow', 116: '117.Clay_colored_Sparrow', 117: '118.House_Sparrow', 118: '119.Field_Sparrow', 119: '120.Fox_Sparrow', 120: '121.Grasshopper_Sparrow', 121: '122.Harris_Sparrow', 122: '123.Henslow_Sparrow', 123: '124.Le_Conte_Sparrow', 124: '125.Lincoln_Sparrow', 125: '126.Nelson_Sharp_tailed_Sparrow', 126: '127.Savannah_Sparrow', 127: '128.Seaside_Sparrow', 128: '129.Song_Sparrow', 129: '130.Tree_Sparrow', 130: '131.Vesper_Sparrow', 131: '132.White_crowned_Sparrow', 132: '133.White_throated_Sparrow', 133: '134.Cape_Glossy_Starling', 134: '135.Bank_Swallow', 135: '136.Barn_Swallow', 136: '137.Cliff_Swallow', 137: '138.Tree_Swallow', 138: '139.Scarlet_Tanager', 139: '140.Summer_Tanager', 140: '141.Artic_Tern', 141: '142.Black_Tern', 142: '143.Caspian_Tern', 143: '144.Common_Tern', 144: '145.Elegant_Tern', 145: '146.Forsters_Tern', 146: '147.Least_Tern', 147: '148.Green_tailed_Towhee', 148: '149.Brown_Thrasher', 149: '150.Sage_Thrasher', 150: '151.Black_capped_Vireo', 151: '152.Blue_headed_Vireo', 152: '153.Philadelphia_Vireo', 153: '154.Red_eyed_Vireo', 154: '155.Warbling_Vireo', 155: '156.White_eyed_Vireo', 156: '157.Yellow_throated_Vireo', 157: '158.Bay_breasted_Warbler', 158: '159.Black_and_white_Warbler', 159: '160.Black_throated_Blue_Warbler', 160: '161.Blue_winged_Warbler', 161: '162.Canada_Warbler', 162: '163.Cape_May_Warbler', 163: '164.Cerulean_Warbler', 164: '165.Chestnut_sided_Warbler', 165: '166.Golden_winged_Warbler', 166: '167.Hooded_Warbler', 167: '168.Kentucky_Warbler', 168: '169.Magnolia_Warbler', 169: '170.Mourning_Warbler', 170: '171.Myrtle_Warbler', 171: '172.Nashville_Warbler', 172: '173.Orange_crowned_Warbler', 173: '174.Palm_Warbler', 174: '175.Pine_Warbler', 175: '176.Prairie_Warbler', 176: '177.Prothonotary_Warbler', 177: '178.Swainson_Warbler', 178: '179.Tennessee_Warbler', 179: '180.Wilson_Warbler', 180: '181.Worm_eating_Warbler', 181: '182.Yellow_Warbler', 182: '183.Northern_Waterthrush', 183: '184.Louisiana_Waterthrush', 184: '185.Bohemian_Waxwing', 185: '186.Cedar_Waxwing', 186: '187.American_Three_toed_Woodpecker', 187: '188.Pileated_Woodpecker', 188: '189.Red_bellied_Woodpecker', 189: '190.Red_cockaded_Woodpecker', 190: '191.Red_headed_Woodpecker', 191: '192.Downy_Woodpecker', 192: '193.Bewick_Wren', 193: '194.Cactus_Wren', 194: '195.Carolina_Wren', 195: '196.House_Wren', 196: '197.Marsh_Wren', 197: '198.Rock_Wren', 198: '199.Winter_Wren', 199: '200.Common_Yellowthroat'}


def evaluate_for_csv(args, model, test_loader, class_names):
    model.eval()
    predictions = []
    file_names = []
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    with torch.no_grad():
        for img_info, img_data in test_loader:
            # 将数据移动到相应的设备
            img_data = img_data.to(device)

            # 通过模型获取输出
            outs = model(img_data)

            # 如果模型输出包含'ori_out'键，处理这部分数据
            if "ori_out" in outs:
                outputs = outs["ori_out"]
                _, predicted = torch.max(outputs, 1)
                print(predicted)
                # 将预测结果和文件名添加到相应的列表中
                predictions.extend(predicted.cpu().numpy())
                file_names.extend(img_info)

    predicted_labels = [class_names[pred] for pred in predictions]
    return file_names, predicted_labels


if __name__ == "__main__":

    tlogger = timeLogger()

    tlogger.print("Reading Config...")
    args = get_args()

    assert args.c != "", "Please provide config file (.yaml)"
    load_yaml(args, args.c)
    build_record_folder(args)
    tlogger.print()

    test_dataset = TestImageDataset(
        root='/Users/maiwenjie/Desktop/大學課程/大三上/機器學習/final_project/test', data_size=args.data_size)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=8, shuffle=False)

    model = MODEL_GETTER["swin-t"](
        use_fpn=args.use_fpn,
        fpn_size=args.fpn_size,
        use_selection=args.use_selection,
        num_classes=args.num_classes,
        num_selects=args.num_selects,
        use_combiner=args.use_combiner,
    )
    model_path = '/Users/maiwenjie/Downloads/best.pt'
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model'])
    # 加載模型
    # model = torch.load(model_path, map_location=torch.device('cpu'))

    # 移動模型到適當的設備
    model.to('cuda' if torch.cuda.is_available() else 'cpu')

    # 將模型設置為評估模式
    model.eval()

    file_names, predicted_labels = evaluate_for_csv(
        args, model, test_loader, class_names)

    submission_df = pd.DataFrame({
        'id': file_names,
        'label': predicted_labels
    })

    submission_csv_path = '/Users/maiwenjie/Desktop/大學課程/大三上/機器學習/final_project/FGVC-HERBS-second'
    submission_df.to_csv(submission_csv_path, index=False)

    main(args, tlogger)
