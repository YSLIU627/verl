import os
import datasets
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dirs', nargs="+", required=True)
    parser.add_argument('--output_dir', default='./data/combined')
    parser.add_argument('--split', default='test')
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Combining datasets from {args.data_dirs} to {args.output_dir}")

    dataset_list = []
    for data_dir in args.data_dirs:
        data_file = os.path.join(data_dir, f"{args.split}.parquet")
        dataset = datasets.load_dataset("parquet", data_files={f'{args.split}': data_file})
        dataset_list.append(dataset[args.split])

    combined_dataset = datasets.concatenate_datasets(dataset_list, split=args.split)
    combined_dataset.to_parquet(os.path.join(args.output_dir, f"{args.split}.parquet"))
    print(f"Combined dataset saved to {args.output_dir}")


if __name__ == "__main__":
    main()