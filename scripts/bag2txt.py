#!/usr/bin/python

#BuildIn
import os
from io import BytesIO
from argparse import ArgumentParser

#Installed
import numpy as np
from PIL import Image #Pillow
import rosbag

def init_argparser(parents=[]):
    '''
    Initialize an ArgumentParser for this script.

    Args:
        parents: A list of ArgumentParsers of other scripts, if any.

    Returns:
        parser: The ArgumentParsers.
    '''
    parser = ArgumentParser(
        description='Script for unpacking BAG-files.',
        parents=parents
        )

    parser.add_argument(
        'BAG',
        help='The one or more BAG-files to be unpacked. Note: Also wildcards are accepted.',
        nargs='+'
        )

    parser.add_argument(
        '--outdir', '-o',
        help='The output folder',
        default='./'
        )

    parser.add_argument(
        '--testsplit', '-t',
        help="Seperates each (-t)'s to a test set. 0 = No test-set will be generated.",
        type=int,
        default=0
        )
    
    parser.add_argument(
        '--delimiter', '-D',
        help="Delimiter that separates the values in the TXT-files.",
        default=' '
        )
    
    parser.add_argument(
        '--verbose',
        help="Set to 'False' for quite mode. (default=True)",
        type=bool,
        default=True
        )

    return parser

def unpack_bag(bag_files, out_dir, test_split=0, delimiter=' ', verbose=True):
    def verbose_print(*args):
        if not verbose:
            pass
        else:
            print ' '.join(args)

    for bag_file in bag_files:
        basename = os.path.splitext(os.path.basename(bag_file))[0]
        curr_dir = os.path.join(out_dir, basename)
        if not os.path.exists(curr_dir):
            verbose_print("Create output folder:", curr_dir)
            os.makedirs(curr_dir)
        train = open(os.path.join(curr_dir, 'train.csv'), 'w')
        test = open(os.path.join(curr_dir, 'test.csv'), 'w')
        headers = delimiter.join(('topic', 'num', 'omega', 'gain', 'timestamp', 'image')) + '\r\n'
        test.write(headers)
        train.write(headers)
        
        with rosbag.Bag(bag_file, 'r') as bag:
            verbose_print("Unpack:", bag_file , "to:", curr_dir)
            topic = None
            img_path = None
            omega = 0
            v = 0
            
            for num, (topic, msg, t) in enumerate(bag.read_messages()):
                if 'CompressedImage' in type(msg).__name__:
                    img_path = os.path.join(curr_dir, "{}.jpg".format(t))
                    img_data = np.fromstring(msg.data, np.uint8).tobytes()
                    try:
                        image = Image.open(BytesIO(img_data))
                        image.save(img_path)
                        verbose_print("Wrote image to: {}".format(img_path))
                    except KeyboardInterrupt:
                        return
                    except:
                        verbose_print("Writing image to: {} failed!".format(img_path))
                    
                    entry = delimiter.join((topic, str(num), str(omega), str(v), str(t), img_path)) + '\r\n'
                    if test_split and num % test_split is 0:
			            test.write(entry)
			            verbose_print("Wrote: '{}' to test".format(entry))
                    else:
                        train.write(entry)
                        verbose_print("Wrote: '{}' to train".format(entry))
                      
                elif 'Twist2DStamped' in type(msg).__name__: 
                    omega = msg.omega
                    v = msg.v
                else:
                    verbose_print("Unsupported type in topic:", topic, type(msg).__name__)
                    continue
                    
        train.close()
        test.close()
    pass 

if __name__ == '__main__':
    parser = init_argparser()
    args, _ = parser.parse_known_args()
    unpack_bag(args.BAG, args.outdir, args.testsplit, args.delimiter, args.verbose)

