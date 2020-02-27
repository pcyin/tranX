# def self_training(args):
#     """Perform self-training
#
#     First load decoding results on disjoint data
#     also load pre-trained model and perform supervised
#     training on both existing training data and the
#     decoded results
#     """
#
#     print('load pre-trained model from [%s]' % args.load_model, file=sys.stderr)
#     params = torch.load(args.load_model, map_location=lambda storage, loc: storage)
#     vocab = params['vocab']
#     transition_system = params['transition_system']
#     saved_args = params['args']
#     saved_state = params['state_dict']
#
#     # transfer arguments
#     saved_args.cuda = args.cuda
#     saved_args.save_to = args.save_to
#     saved_args.train_file = args.train_file
#     saved_args.unlabeled_file = args.unlabeled_file
#     saved_args.dev_file = args.dev_file
#     saved_args.load_decode_results = args.load_decode_results
#     args = saved_args
#
#     update_args(args, arg_parser)
#
#     model = Parser(saved_args, vocab, transition_system)
#     model.load_state_dict(saved_state)
#
#     if args.cuda: model = model.cuda()
#     model.train()
#     optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
#
#     print('load unlabeled data [%s]' % args.unlabeled_file, file=sys.stderr)
#     unlabeled_data = Dataset.from_bin_file(args.unlabeled_file)
#
#     print('load decoding results of unlabeled data [%s]' % args.load_decode_results, file=sys.stderr)
#     decode_results = pickle.load(open(args.load_decode_results))
#
#     labeled_data = Dataset.from_bin_file(args.train_file)
#     dev_set = Dataset.from_bin_file(args.dev_file)
#
#     print('Num. examples in unlabeled data: %d' % len(unlabeled_data), file=sys.stderr)
#     assert len(unlabeled_data) == len(decode_results)
#     self_train_examples = []
#     for example, hyps in zip(unlabeled_data, decode_results):
#         if hyps:
#             hyp = hyps[0]
#             sampled_example = Example(idx='self_train-%s' % example.idx,
#                                       src_sent=example.src_sent,
#                                       tgt_code=hyp.code,
#                                       tgt_actions=hyp.action_infos,
#                                       tgt_ast=hyp.tree)
#             self_train_examples.append(sampled_example)
#     print('Num. self training examples: %d, Num. labeled examples: %d' % (len(self_train_examples), len(labeled_data)),
#           file=sys.stderr)
#
#     train_set = Dataset(examples=labeled_data.examples + self_train_examples)
#
#     print('begin training, %d training examples, %d dev examples' % (len(train_set), len(dev_set)), file=sys.stderr)
#     print('vocab: %s' % repr(vocab), file=sys.stderr)
#
#     epoch = train_iter = 0
#     report_loss = report_examples = 0.
#     history_dev_scores = []
#     num_trial = patience = 0
#     while True:
#         epoch += 1
#         epoch_begin = time.time()
#
#         for batch_examples in train_set.batch_iter(batch_size=args.batch_size, shuffle=True):
#             batch_examples = [e for e in batch_examples if len(e.tgt_actions) <= args.decode_max_time_step]
#
#             train_iter += 1
#             optimizer.zero_grad()
#
#             loss = -model.score(batch_examples)
#             # print(loss.data)
#             loss_val = torch.sum(loss).data[0]
#             report_loss += loss_val
#             report_examples += len(batch_examples)
#             loss = torch.mean(loss)
#
#             loss.backward()
#
#             # clip gradient
#             if args.clip_grad > 0.:
#                 grad_norm = torch.nn.utils.clip_grad_norm(model.parameters(), args.clip_grad)
#
#             optimizer.step()
#
#             if train_iter % args.log_every == 0:
#                 print('[Iter %d] encoder loss=%.5f' %
#                       (train_iter,
#                        report_loss / report_examples),
#                       file=sys.stderr)
#
#                 report_loss = report_examples = 0.
#
#         print('[Epoch %d] epoch elapsed %ds' % (epoch, time.time() - epoch_begin), file=sys.stderr)
#         # model_file = args.save_to + '.iter%d.bin' % train_iter
#         # print('save model to [%s]' % model_file, file=sys.stderr)
#         # model.save(model_file)
#
#         # perform validation
#         print('[Epoch %d] begin validation' % epoch, file=sys.stderr)
#         eval_start = time.time()
#         eval_results = evaluation.evaluate(dev_set.examples, model, args, verbose=True)
#         dev_acc = eval_results['accuracy']
#         print('[Epoch %d] code generation accuracy=%.5f took %ds' % (epoch, dev_acc, time.time() - eval_start), file=sys.stderr)
#         is_better = history_dev_scores == [] or dev_acc > max(history_dev_scores)
#         history_dev_scores.append(dev_acc)
#
#         if is_better:
#             patience = 0
#             model_file = args.save_to + '.bin'
#             print('save currently the best model ..', file=sys.stderr)
#             print('save model to [%s]' % model_file, file=sys.stderr)
#             model.save(model_file)
#             # also save the optimizers' state
#             torch.save(optimizer.state_dict(), args.save_to + '.optim.bin')
#         elif epoch == args.max_epoch:
#             print('reached max epoch, stop!', file=sys.stderr)
#             exit(0)
#         elif patience < args.patience:
#             patience += 1
#             print('hit patience %d' % patience, file=sys.stderr)
#
#         if patience == args.patience:
#             num_trial += 1
#             print('hit #%d trial' % num_trial, file=sys.stderr)
#             if num_trial == args.max_num_trial:
#                 print('early stop!', file=sys.stderr)
#                 exit(0)
#
#             # decay lr, and restore from previously best checkpoint
#             lr = optimizer.param_groups[0]['lr'] * args.lr_decay
#             print('load previously best model and decay learning rate to %f' % lr, file=sys.stderr)
#
#             # load model
#             params = torch.load(args.save_to + '.bin', map_location=lambda storage, loc: storage)
#             model.load_state_dict(params['state_dict'])
#             if args.cuda: model = model.cuda()
#
#             # load optimizers
#             if args.reset_optimizer:
#                 print('reset optimizer', file=sys.stderr)
#                 optimizer = torch.optim.Adam(model.inference_model.parameters(), lr=lr)
#             else:
#                 print('restore parameters of the optimizers', file=sys.stderr)
#                 optimizer.load_state_dict(torch.load(args.save_to + '.optim.bin'))
#
#             # set new lr
#             for param_group in optimizer.param_groups:
#                 param_group['lr'] = lr
#
#             # reset patience
#             patience = 0
#
