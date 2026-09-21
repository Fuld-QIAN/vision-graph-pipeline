"""Untrained GNN/LSTM interface test. Outputs are NOT validated representations."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn


class SceneEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_layers = nn.ModuleList([nn.Linear(37,64),nn.Linear(64,64)])
        self.neighbor_layers = nn.ModuleList([nn.Linear(37,64),nn.Linear(64,64)])
        self.lstm = nn.LSTM(128,64,num_layers=2,batch_first=True)

    def spatial(self, graph):
        device = next(self.parameters()).device
        x = torch.tensor(graph['x'],dtype=torch.float32,device=device)
        edges = torch.tensor(graph['edges'],dtype=torch.long,device=device).T
        if x.ndim != 2 or x.shape[1] != 37 or not torch.isfinite(x).all():
            raise ValueError('Invalid graph features')
        src,dst = edges
        for own,neighbor in zip(self.self_layers,self.neighbor_layers):
            aggregated = torch.zeros_like(x)
            aggregated.index_add_(0,dst,x[src])
            count = torch.bincount(dst,minlength=len(x)).clamp(min=1).unsqueeze(1)
            x = torch.relu(own(x)+neighbor(aggregated/count))
        pooled = x[1:].mean(0) if len(x)>1 else torch.zeros_like(x[0])
        return torch.cat([x[0],pooled])


def select_past_frames(graphs, step=.1):
    times = np.array([g['time_s'] for g in graphs])
    if len(times)==0 or np.any(np.diff(times)<=0): raise ValueError('Empty/nonmonotonic sequence')
    grid = np.arange(times[0],times[-1]+1e-8,step)
    indexes = np.searchsorted(times,grid+1e-9,side='right')-1
    return grid,indexes


def main(args):
    output = args.input/'encoder_smoke'
    output.mkdir(exist_ok=False)
    torch.manual_seed(33)
    model = SceneEncoder().to(args.device).eval()
    manifest = {'status':'untrained_forward_test','trained':False,'seed':33,
                'warning':'Random weights. Do not use these embeddings as validated cognitive predictors.',
                'graph_encoder':'two-layer mean-neighbor message passing with node type features',
                'sampling_seconds':.1,'window_steps':20,'events':{},
                'processed_events':0,'skipped_events':0}
    requested = {value.strip() for value in args.event_id.split(',') if value.strip()}
    event_dirs = sorted(
        path for path in args.input.iterdir()
        if path.is_dir() and (path/'graphs.jsonl').exists()
        and (not requested or path.name in requested)
    )
    if not event_dirs:
        raise FileNotFoundError(f'No event graphs found under {args.input}')
    with torch.no_grad():
        for event_dir in event_dirs:
            event = event_dir.name
            graphs = [json.loads(s) for s in (event_dir/'graphs.jsonl').read_text(encoding='utf-8').splitlines() if s.strip()]
            if not graphs:
                manifest['events'][event] = {'graphs':0,'windows':0,'status':'skipped_no_graphs'}
                manifest['skipped_events'] += 1
                continue
            grid,indexes = select_past_frames(graphs)
            # Reject windows across missing frames; don't compress failed times away.
            times = np.array([g['time_s'] for g in graphs])
            valid = grid-times[indexes] <= .1+1e-8
            vectors = torch.stack([model.spatial(graphs[i]) for i in indexes])
            results,ends,end_frames = [],[],[]
            for end in range(19,len(grid)):
                if not valid[end-19:end+1].all(): continue
                sequence,_ = model.lstm(vectors[end-19:end+1].unsqueeze(0))
                results.append(sequence[0,-1].cpu().numpy())
                ends.append(grid[end]); end_frames.append(graphs[indexes[end]]['frame'])
            if not results:
                manifest['events'][event] = {
                    'graphs':len(graphs),'windows':0,
                    'status':'skipped_no_complete_20_step_window'
                }
                manifest['skipped_events'] += 1
                continue
            encoded = np.stack(results)
            if not np.isfinite(encoded).all(): raise ValueError('Non-finite encoding')
            np.savez_compressed(output/f'{event}_UNTRAINED.npz',embeddings=encoded,
                                window_end_s=ends,window_end_frame=end_frames,trained=False)
            manifest['events'][event] = {'graphs':len(graphs),'windows':len(results),
                                         'output_shape':list(encoded.shape),'status':'completed'}
            manifest['processed_events'] += 1
    if manifest['processed_events'] == 0:
        raise ValueError('No event produced a complete 20-step sequence')
    if manifest['skipped_events']:
        manifest['status'] = 'untrained_forward_test_with_skipped_short_events'
    torch.save(model.state_dict(),output/'UNTRAINED_initial_weights.pt')
    (output/'summary.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(manifest,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--event-id',default='',help='Optional comma-separated event IDs; default discovers all graph event directories')
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    main(p.parse_args())
