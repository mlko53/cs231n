import copy
import json

from args import TrainArgParser


def write_args(args):
    save_dir = args.save_dir
    copy_args = copy.deepcopy(args)
    copy_args.save_dir = save_dir.as_posix()
    """Save args as a JSON file"""
    with (save_dir / 'args.json').open('w') as f:
        args_dict = vars(copy_args)
        print(args_dict)
        json.dump(args_dict, f, indent=4, sort_keys=True)
        f.write('/n')

    print("Saved Arguments")

def main(args):
    write_args(args)


if __name__ == '__main__':
    parser = TrainArgParser()
    main(parser.parse_args())
