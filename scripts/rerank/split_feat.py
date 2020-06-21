from __future__ import print_function
import sys, os

nbest_list_file = sys.argv[1]

print('reading features from %s' % nbest_list_file, sys.stderr)

f_recon = open(nbest_list_file + '.+recon', 'w')
f_para = open(nbest_list_file + '.+para', 'w')
f_recon_para = open(nbest_list_file + '.+recon+para', 'w')
f_recon_wc = open(nbest_list_file + '.+recon+wc', 'w')
f_para_wc = open(nbest_list_file + '.+para+wc', 'w')
f_recon_np = open(nbest_list_file + '.+recon+np', 'w')
f_para_np = open(nbest_list_file + '.+para+np', 'w')
f_recon_para_wc = open(nbest_list_file + '.+recon+para+wc', 'w')
f_recon_para_np = open(nbest_list_file + '.+recon+para+np', 'w')

for line in open(nbest_list_file):
    line = line.strip()
    d = line.split(' ||| ')
    features = d[3].strip().split(' ')

    parser_score = features[0]
    reconstructor = features[1]
    paraphrase_identifier = features[2]
    normalized_parser_score = features[3]
    wc = features[4]

    prefix = ' ||| '.join([d[0], d[1], d[2]])

    f_recon.write(prefix + ' ||| ' + ' '.join([parser_score, reconstructor]) + '\n')
    f_para.write(prefix + ' ||| ' + ' '.join([parser_score, paraphrase_identifier]) + '\n')
    f_recon_para.write(prefix + ' ||| ' + ' '.join([parser_score, reconstructor, paraphrase_identifier]) + '\n')
    f_recon_wc.write(prefix + ' ||| ' + ' '.join([parser_score, reconstructor, wc]) + '\n')
    f_para_wc.write(prefix + ' ||| ' + ' '.join([parser_score, paraphrase_identifier, wc]) + '\n')
    f_recon_np.write(prefix + ' ||| ' + ' '.join([parser_score, reconstructor, normalized_parser_score]) + '\n')
    f_para_np.write(prefix + ' ||| ' + ' '.join([parser_score, paraphrase_identifier, normalized_parser_score]) + '\n')
    f_recon_para_wc.write(prefix + ' ||| ' + ' '.join([parser_score, reconstructor, paraphrase_identifier, wc]) + '\n')
    f_recon_para_np.write(prefix + ' ||| ' + ' '.join([parser_score, reconstructor, paraphrase_identifier, normalized_parser_score]) + '\n')

f_recon.close()
f_para.close()
f_recon_para.close()
f_recon_wc.close()
f_para_wc.close()
f_recon_np.close()
f_para_np.close()
f_recon_para_wc.close()
f_recon_para_np.close()
