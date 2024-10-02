import gymnasium as gym
from pymahjong.replayer import replay, worker, calculate
from base_module import *
import argparse, os 
from multiprocessing import Queue, Process
import pickle
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Mahjong AI BC Training")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--num_worker", type=int, default=64, help="Number of processes")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints", help="Checkpoint path")
    parser.add_argument("--device", type=str, default="cuda:3", help="Device")
    parser.add_argument("--datapath", type=str, default="/data/satori_hdd4/Ren/dataset", help="Data path")
    parser.add_argument("--year", type=list, default=[2011, 2012, 2013, 2014, 2015], help="Year")
    return parser.parse_args()


def main():
    args = parse_args()
    training_device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("device:", training_device)


    config = GPT2Config(n_embd=512, n_layer=8, n_head=8, n_positions=128)
    model = Policy_Network(config).to(training_device)
    model.train()

    input_collator = myCollator(device=training_device)
    label_collator = label_Collator(device=training_device)
    criteria = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    years = args.year
    dataset = []
    label = []
    for year in years:
        path = os.path.join(args.datapath, str(year))
        file_name_list = os.listdir(path)
        # 创建多进程
        num_worker = args.num_worker
        input_queue = Queue()
        output_queue = Queue()
        counter = {
        'success': 0,
        'failed': 0,
        'skip': 0,
        }
        for file_name in file_name_list:
            input_queue.put((replay, (path, file_name)))
        for i in range(num_worker):
            Process(target=worker, args=(input_queue, output_queue)).start()

        print(f"Start extracting {year}'s data")
        for i in tqdm(range(len(file_name_list))):
            data = output_queue.get()
            if data is None:
                counter['skip'] += 1
            else:
                if data['success']:
                    dataset.extend(data['data'])
                    label.extend(data['label'])
                    counter['success'] += 1
                else:
                    counter['failed'] += 1

        for i in range(num_worker):
            input_queue.put('STOP')
        
        print("Number of success:", counter['success'])
        print("Number of failed:", counter['failed'])
        print("Number of skip:", counter['skip'])


    print("Number of data:", len(dataset))
    # 保存数据集
    with open("dataset.pkl", "wb") as f:
        pickle.dump(dataset, f)
    with open("label.pkl", "wb") as f:
        pickle.dump(label, f)

    print("Start training")
    batch_size = args.batch_size
    for epoch in tqdm(range(args.epochs)):
        # shuffle
        index = torch.randperm(len(dataset))
        dataset = [dataset[i] for i in index]
        label = [label[i] for i in index]

        for i in range(0, len(dataset), batch_size):
            batch_input = dataset[i:i+batch_size]
            batch_label = label[i:i+batch_size]
            input = input_collator(batch_input)

            model.zero_grad()
            output = model(**input)

            batch_label = label_collator(batch_label)
            loss = criteria(output['action_logits'], batch_label)
            loss.backward()

            optimizer.step()

            print("epoch:", epoch, "batch:", i/batch_size + 1, "loss:", loss.item())
        torch.save(model.state_dict(), os.path.join(args.checkpoint, "model_{}_{}.pth".format(year, epoch)))

if __name__ == "__main__":
    main()


