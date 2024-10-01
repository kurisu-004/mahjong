import gymnasium as gym
from pymahjong.replayer import myReplayer
from base_module import *
import argparse, os 
import multiprocessing
from tqdm import tqdm

def extract_data(path, file_name_list:list):
    replayer = myReplayer(path, file_name_list)
    data, success, failed, failed_env = replayer.replay()
    print(data)

    # queue.put(data)
    return data

def parse_args():
    parser = argparse.ArgumentParser(description="Mahjong AI BC Training")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--num_envs", type=int, default=64, help="Number of environments")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints", help="Checkpoint path")
    parser.add_argument("--device", type=str, default="cuda:3", help="Device")
    parser.add_argument("--path", type=str, default="/data/satori_hdd4/Ren/dataset", help="Data path")
    parser.add_argument("--year", type=list, default=[2014, 2015], help="Year")
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

    for year in years:
        path = os.path.join(args.path, str(year))
        file_name_list = os.listdir(path)
        # ps = []
        # queue = Queue()

        # print("Extracting data from {}...".format(path))
        # for i in range(0, len(file_name_list), len(file_name_list)//args.num_envs):
        #     p = multiprocessing.Process(target=extract_data, args=(path, file_name_list[i:i+len(file_name_list)//args.num_envs], queue))
        #     p.start()
        #     ps.append(p)
        # for p in ps:
        #     p.join()

        # all_data = []
        # all_label = []
        # while not queue.empty():
        #     data = queue.get()
        #     if len(data['obs']) == 0:
        #         continue
        #     all_data.extend(data['obs'][0]) # TODO: 这里为了方便，只取了第一局的数据
        #     all_label.extend(data['label'][0])
        data = extract_data(path, file_name_list)
        all_data = []
        all_label = []
        for kyoku, label in zip(data['obs'], data['label']):
            all_data.extend(kyoku)
            all_label.extend(label) 

        print("Number of data:", len(all_data))
        for epoch in tqdm(range(args.epochs)):
            batch_size = args.batch_size
            for i in range(0, len(all_data), batch_size):
                batch_input = all_data[i:i+batch_size]
                batch_label = all_label[i:i+batch_size]
                input = input_collator(batch_input)

                model.zero_grad()
                output = model(**input)

                label = label_collator(batch_label)
                loss = criteria(output['action_logits'], label)
                loss.backward()

                optimizer.step()

                print("epoch:", epoch, "batch:", i/batch_size + 1, "loss:", loss.item())
            torch.save(model.state_dict(), os.path.join(args.checkpoint, "model_{}_{}.pth".format(year, epoch)))

if __name__ == "__main__":
    main()


