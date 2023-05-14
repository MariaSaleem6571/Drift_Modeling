import argparse

import os
import torch
from torch.utils.data import DataLoader
from torch import nn

from core import model
from core.model import loss
from core import util

from scripts import helper


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluates a model that predicts density map snapshots.',
        formatter_class=helper.ArgParseFormatter)
    
    parser.add_argument('directory', type=str,
                        help='root directory')
    parser.add_argument('dataset', type=str, nargs='+',
                        help='dataset name(s)')
    parser.add_argument('runid', type=str, default=0,
                        help='run id of the model')
    parser.add_argument('--subset', type=str, default='test',
                        help='evaluation subset')
    
    parser.add_argument('--fieldname', type=str, default='velocity',
                        help='name of the field to use as input')
    parser.add_argument('--channels', type=int, default=5,
                        help='total number of input channels')
    
    parser.add_argument('--one-day-only', action='store_false',
                        help='do not input the field at t+1')
    parser.add_argument('--interp', type=float, default=0,
                        help='given a value of 0-1, the input field is '
                        'interpolated from t and t+1')
    
    parser.add_argument('--residual', action='store_true',
                        help='evaluate residual maps')
    parser.add_argument('--threshold', action='store_true',
                        help='compute the loss only on values above 0')
    
    parser.add_argument('--batchsize', type=int, default=24,
                        help='batch size')
    
    parser.add_argument('--nw', type=int, default=8,
                        help='number of dataloader workers')
    parser.add_argument('--nmp', action='store_false',
                        help='do not use mixed precision GPU operations')
    parser.add_argument('--debug', action='store_true',
                        help='run the script in debug mode')
    
    # Parsers for Multi-Day Predictions
    parser.add_argument('--saving_name', type=str, default='model-test',
                        help='name to save model evaluation result')
    parser.add_argument('--no_of_days', type=int, default=1,
                        help='no of days ahead to be predicted')
    
    # Parsers for Plotting
    parser.add_argument('--plotting_result', action='store_true',
                        help='plot a given result')
    parser.add_argument('--plot_input_id', type=int, default=1,
                        help='id of the image to serve as base for prediction')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    
    for ds in args.dataset:
        paths = helper.PathIndex(args.directory, ds) 
        loader = helper.Loader(paths)
        checkpoint_path = paths.model_dir / args.runid
        
        # fetch evaluation set and create dataloader
        dataset = loader.snapshot_dataset(
            args.fieldname, subset=args.subset, input_map=args.residual,
            field_interp=args.interp, next_field=args.one_day_only, no_of_days=args.no_of_days)
        dataloader = DataLoader(
            dataset, args.batchsize, pin_memory=True, num_workers=args.nw)
        
        # select GPU if available
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # initialisations
        out_modifier = util.model.MaskedOutput(loader.glazure64_mesh.mask)
        net = model.models.unet(n_channels=args.channels, n_classes=1,
                                out_modifier=out_modifier)
        
        # EVALUATION OF ALL DATA WITHOUT PLOTTING
        
        # loss function
        land_mask = ~loader.glazure64_mesh.mask
        loss_fns = [loss.MAE(land_mask, batch_mean=False), loss.MSE(land_mask, batch_mean=False), 
                loss.MAEProbDistrLoss(land_mask, batch_mean=False), loss.MSEProbDistrLoss(land_mask, batch_mean=False)]
        loss_names = ["MAE", "MSE", "MAEwProba", "MSEwProba"]
        
        if not args.plotting_result:
            # evaluate
            evaluator = model.Evaluator(
                device, net, dataloader, loss_fns, checkpoint_path, args.nmp)
            try:
                evaluator.load_best_checkpoint()
            except:
                evaluator.load_last_checkpoint()
            
            # NAme for saving the output of the model evaluation result
            #prefix = '{}_{}'.format(ds, args.subset)
            # prefix = args.saving_name
            prefix = f'{ds}_{args.subset}_{args.no_of_days}'
            
            evaluator.save_results(
                prefix=prefix, residual=args.residual, clip=True, loss_names=loss_names, loss_fns=loss_fns)
        
        # PLOTTING A SPECIFIC INPUT, TRUE OUTPUT AND PREDICTED OUTPUT
        else:
            # evaluate
            evaluator = model.plotEvaluator(
                device, net, dataloader, loss_fns, checkpoint_path, args.nmp)
            try:
                evaluator.load_best_checkpoint()
            except:
                evaluator.load_last_checkpoint()
                
            plotting_dir = os.path.join(paths.plotting_dir, args.runid)
            
            evaluator.plot_results(
                plotting_dir, clip=True, groundtruth_id = args.plot_input_id)
                
            
