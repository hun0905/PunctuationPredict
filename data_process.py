import argparse
parser = argparse.ArgumentParser(description="Text process.")
parser.add_argument('--in_path',default='/home/yunghuan/Desktop/nlp_punctuation/data/text_noprocess', type=str, required=True, help='Text data to be punctuated')
parser.add_argument('--out_path', default='/home/yunghuan/Desktop/nlp_punctuation/data/train20_nopunc', type=str, required=True, help='Input vocab. (Don\'t include <UNK> and <END>)')
def remove_punc(args):
    in_path = args.in_path
    out_path = args.out_path
    in_file = open(in_path,'r',encoding='utf8')
    out = in_file.read().replace('<questionmark>','<period>').split()
    out = out[0:50000]#for non_punc
    count = 1
    for i in out:
        count+=1
        if count % 50 == 0:
            out.insert(count,'\n')
        if i == '<period>' or i == '<comma>':#for non_punc
            out.remove(i)#for non_punc
    out_file = open(out_path,'w+',encoding='utf8')
    out = ' '.join(out)
    #print(out)
    out_file.write(out)
if __name__ == '__main__':
    args = parser.parse_args()
    print(args)
    remove_punc(args)
